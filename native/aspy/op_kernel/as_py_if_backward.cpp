#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kHalfPi = 1.57079632679489661923f;

class KernelAsPyIfBackward {
 public:
  __aicore__ inline KernelAsPyIfBackward() = default;

  __aicore__ inline void Init(
      GM_ADDR h_seq,
      GM_ADDR spike_seq,
      GM_ADDR grad_spike_seq,
      GM_ADDR grad_v_seq,
      GM_ADDR grad_v_final,
      GM_ADDR grad_x_seq,
      GM_ADDR grad_v_init,
      const AsPyIfBackwardTilingData& tiling) {
    time_steps_ = tiling.timeSteps;
    neuron_count_ = tiling.neuronCount;
    tile_length_ = tiling.tileLength;
    tile_count_ = tiling.tileCount;
    threshold_ = tiling.vThreshold;
    reset_ = tiling.vReset;
    surrogate_alpha_ = tiling.surrogateAlpha;
    hard_reset_ = tiling.hardReset != 0;
    detach_reset_ = tiling.detachReset != 0;
    block_index_ = GetBlockIdx();
    block_count_ = GetBlockNum();
    h_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_seq));
    spike_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(spike_seq));
    grad_spike_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_spike_seq));
    grad_v_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_seq));
    grad_v_final_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_final));
    grad_x_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_x_seq));
    grad_v_init_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_init));

    const uint32_t buffer_bytes = tile_length_ * sizeof(float);
    pipe_.InitBuffer(h_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(spike_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(grad_spike_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(grad_v_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(output_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(calc_buffer_, 3U * buffer_bytes);
    LocalTensor<uint8_t> ub = calc_buffer_.Get<uint8_t>();
    carry_ = ub[0].ReinterpretCast<float>();
    surrogate_ = ub[buffer_bytes].ReinterpretCast<float>();
    scratch_ = ub[2U * buffer_bytes].ReinterpretCast<float>();
  }

  __aicore__ inline void Process() {
    for (uint32_t tile = block_index_; tile < tile_count_; tile += block_count_) {
      const uint32_t begin = tile * tile_length_;
      const uint32_t count =
          begin + tile_length_ <= neuron_count_
              ? tile_length_
              : neuron_count_ - begin;
      ProcessTile(begin, count);
    }
  }

 private:
  __aicore__ inline void CopyOut(
      GlobalTensor<float>& destination,
      uint64_t offset,
      const LocalTensor<float>& source,
      uint32_t count) {
    LocalTensor<float> output = output_queue_.AllocTensor<float>();
    Adds(output, source, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    output_queue_.EnQue(output);
    output = output_queue_.DeQue<float>();
    DataCopy(destination[offset], output, count);
    output_queue_.FreeTensor(output);
  }

  __aicore__ inline void ProcessTile(uint32_t begin, uint32_t count) {
    // The public adapter pads each flattened row to eight FP32 values. Every
    // tile and tail is therefore 32-byte aligned for disjoint multicore DMA.
    LocalTensor<float> initial = h_queue_.AllocTensor<float>();
    DataCopy(initial, grad_v_final_gm_[begin], count);
    h_queue_.EnQue(initial);
    initial = h_queue_.DeQue<float>();
    Adds(carry_, initial, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    h_queue_.FreeTensor(initial);

    const float surrogate_scale = kHalfPi * surrogate_alpha_;
    const float surrogate_numerator = 0.5f * surrogate_alpha_;
    for (uint32_t reverse = 0; reverse < time_steps_; ++reverse) {
      const uint32_t time = time_steps_ - 1U - reverse;
      const uint64_t offset =
          static_cast<uint64_t>(time) * neuron_count_ + begin;

      LocalTensor<float> h = h_queue_.AllocTensor<float>();
      DataCopy(h, h_seq_gm_[offset], count);
      h_queue_.EnQue(h);
      LocalTensor<float> spike = spike_queue_.AllocTensor<float>();
      DataCopy(spike, spike_seq_gm_[offset], count);
      spike_queue_.EnQue(spike);
      LocalTensor<float> grad_spike = grad_spike_queue_.AllocTensor<float>();
      DataCopy(grad_spike, grad_spike_seq_gm_[offset], count);
      grad_spike_queue_.EnQue(grad_spike);
      LocalTensor<float> grad_v = grad_v_queue_.AllocTensor<float>();
      DataCopy(grad_v, grad_v_seq_gm_[offset], count);
      grad_v_queue_.EnQue(grad_v);

      h = h_queue_.DeQue<float>();
      spike = spike_queue_.DeQue<float>();
      grad_spike = grad_spike_queue_.DeQue<float>();
      grad_v = grad_v_queue_.DeQue<float>();

      // carry += grad_v_seq[t]
      Add(carry_, carry_, grad_v, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      // ATan'(h - threshold) = (0.5 * alpha) /
      //     (1 + ((pi / 2) * alpha * (h - threshold))^2)
      Adds(scratch_, h, -threshold_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Muls(scratch_, scratch_, surrogate_scale, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Mul(surrogate_, scratch_, scratch_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Adds(surrogate_, surrogate_, 1.0f, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      // Use the native FP32 divide instruction rather than the approximate
      // vrec instruction: one vrec pass is not accurate enough for the
      // existing eager-gradient qualification tolerance on 910B.
      Duplicate(scratch_, surrogate_numerator, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Div(surrogate_, scratch_, surrogate_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      // Reuse scratch_ for reset_grad after the surrogate has been formed.
      if (hard_reset_) {
        Muls(scratch_, spike, -1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Adds(scratch_, scratch_, 1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        if (!detach_reset_) {
          // (reset - h) * surrogate is accumulated without another UB tensor.
          Muls(spike, h, -1.0f, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Adds(spike, spike, reset_, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Mul(spike, spike, surrogate_, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Add(scratch_, scratch_, spike, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
        }
      } else if (detach_reset_) {
        Duplicate(scratch_, 1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
      } else {
        Muls(scratch_, surrogate_, -threshold_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Adds(scratch_, scratch_, 1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
      }

      // grad_h = grad_spike * surrogate + carry * reset_grad
      Mul(grad_spike, grad_spike, surrogate_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Mul(carry_, carry_, scratch_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Add(carry_, carry_, grad_spike, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      CopyOut(grad_x_seq_gm_, offset, carry_, count);

      h_queue_.FreeTensor(h);
      spike_queue_.FreeTensor(spike);
      grad_spike_queue_.FreeTensor(grad_spike);
      grad_v_queue_.FreeTensor(grad_v);
    }

    CopyOut(grad_v_init_gm_, begin, carry_, count);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> h_queue_;
  TQue<QuePosition::VECIN, 1> spike_queue_;
  TQue<QuePosition::VECIN, 1> grad_spike_queue_;
  TQue<QuePosition::VECIN, 1> grad_v_queue_;
  TQue<QuePosition::VECOUT, 1> output_queue_;
  TBuf<QuePosition::VECCALC> calc_buffer_;
  LocalTensor<float> carry_;
  LocalTensor<float> surrogate_;
  LocalTensor<float> scratch_;
  GlobalTensor<float> h_seq_gm_;
  GlobalTensor<float> spike_seq_gm_;
  GlobalTensor<float> grad_spike_seq_gm_;
  GlobalTensor<float> grad_v_seq_gm_;
  GlobalTensor<float> grad_v_final_gm_;
  GlobalTensor<float> grad_x_seq_gm_;
  GlobalTensor<float> grad_v_init_gm_;
  uint32_t time_steps_;
  uint32_t neuron_count_;
  uint32_t tile_length_;
  uint32_t tile_count_;
  uint32_t block_index_;
  uint32_t block_count_;
  float threshold_;
  float reset_;
  float surrogate_alpha_;
  bool hard_reset_;
  bool detach_reset_;
};
}  // namespace

extern "C" __global__ __aicore__ void as_py_if_backward(
    GM_ADDR hSeq,
    GM_ADDR spikeSeq,
    GM_ADDR gradSpikeSeq,
    GM_ADDR gradVSeq,
    GM_ADDR gradVFinal,
    GM_ADDR gradXSeq,
    GM_ADDR gradVInit,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  KernelAsPyIfBackward kernel;
  kernel.Init(
      hSeq,
      spikeSeq,
      gradSpikeSeq,
      gradVSeq,
      gradVFinal,
      gradXSeq,
      gradVInit,
      tiling_data);
  kernel.Process();
}
