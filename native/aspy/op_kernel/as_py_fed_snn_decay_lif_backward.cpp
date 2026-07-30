#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kHalfPi = 1.57079632679489661923f;

class KernelAsPyFedSNNDecayLifBackward {
 public:
  __aicore__ inline KernelAsPyFedSNNDecayLifBackward() = default;

  __aicore__ inline void Init(
      GM_ADDR h_seq,
      GM_ADDR grad_spike_seq,
      GM_ADDR grad_current_seq,
      const AsPyFedSNNDecayLifBackwardTilingData& tiling) {
    time_steps_ = tiling.timeSteps;
    neuron_count_ = tiling.neuronCount;
    tile_length_ = tiling.tileLength;
    tile_count_ = tiling.tileCount;
    membrane_decay_ = tiling.membraneDecay;
    threshold_ = tiling.vThreshold;
    surrogate_alpha_ = tiling.surrogateAlpha;
    block_index_ = GetBlockIdx();
    block_count_ = GetBlockNum();
    h_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_seq));
    grad_spike_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_spike_seq));
    grad_current_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_current_seq));

    const uint32_t buffer_bytes = tile_length_ * sizeof(float);
    pipe_.InitBuffer(h_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(grad_spike_queue_, 1, buffer_bytes);
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
    Duplicate(carry_, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();

    const float surrogate_scale = kHalfPi * surrogate_alpha_;
    const float surrogate_numerator = 0.5f * surrogate_alpha_;
    for (uint32_t reverse = 0; reverse < time_steps_; ++reverse) {
      const uint32_t time = time_steps_ - 1U - reverse;
      const uint64_t offset =
          static_cast<uint64_t>(time) * neuron_count_ + begin;

      LocalTensor<float> h = h_queue_.AllocTensor<float>();
      DataCopy(h, h_seq_gm_[offset], count);
      h_queue_.EnQue(h);
      LocalTensor<float> grad_spike = grad_spike_queue_.AllocTensor<float>();
      DataCopy(grad_spike, grad_spike_seq_gm_[offset], count);
      grad_spike_queue_.EnQue(grad_spike);
      h = h_queue_.DeQue<float>();
      grad_spike = grad_spike_queue_.DeQue<float>();

      Adds(scratch_, h, -threshold_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Muls(scratch_, scratch_, surrogate_scale, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Mul(surrogate_, scratch_, scratch_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Adds(surrogate_, surrogate_, 1.0f, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Duplicate(scratch_, surrogate_numerator, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Div(surrogate_, scratch_, surrogate_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      // grad_h = grad_spike * ATan'(h-threshold) + carry.
      Mul(grad_spike, grad_spike, surrogate_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Add(carry_, grad_spike, carry_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      CopyOut(grad_current_seq_gm_, offset, carry_, count);
      // The detached soft reset contributes identity; charge contributes decay.
      Muls(carry_, carry_, membrane_decay_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      h_queue_.FreeTensor(h);
      grad_spike_queue_.FreeTensor(grad_spike);
    }
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> h_queue_;
  TQue<QuePosition::VECIN, 1> grad_spike_queue_;
  TQue<QuePosition::VECOUT, 1> output_queue_;
  TBuf<QuePosition::VECCALC> calc_buffer_;
  LocalTensor<float> carry_;
  LocalTensor<float> surrogate_;
  LocalTensor<float> scratch_;
  GlobalTensor<float> h_seq_gm_;
  GlobalTensor<float> grad_spike_seq_gm_;
  GlobalTensor<float> grad_current_seq_gm_;
  uint32_t time_steps_;
  uint32_t neuron_count_;
  uint32_t tile_length_;
  uint32_t tile_count_;
  uint32_t block_index_;
  uint32_t block_count_;
  float membrane_decay_;
  float threshold_;
  float surrogate_alpha_;
};
}  // namespace

extern "C" __global__ __aicore__ void as_py_fed_snn_decay_lif_backward(
    GM_ADDR hSeq,
    GM_ADDR gradSpikeSeq,
    GM_ADDR gradCurrentSeq,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  KernelAsPyFedSNNDecayLifBackward kernel;
  kernel.Init(hSeq, gradSpikeSeq, gradCurrentSeq, tiling_data);
  kernel.Process();
}
