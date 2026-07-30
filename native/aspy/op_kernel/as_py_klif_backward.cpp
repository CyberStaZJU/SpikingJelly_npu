#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr float kHalfPi = 1.57079632679489661923f;
constexpr uint32_t kVectorRepeat = 64;

class KernelAsPyKlifBackward {
 public:
  __aicore__ inline KernelAsPyKlifBackward() = default;

  __aicore__ inline void Init(
      GM_ADDR x_seq,
      GM_ADDR v_prev_seq,
      GM_ADDR h_seq,
      GM_ADDR spike_seq,
      GM_ADDR grad_spike_seq,
      GM_ADDR grad_v_seq,
      GM_ADDR grad_v_final,
      GM_ADDR k,
      GM_ADDR grad_x_seq,
      GM_ADDR grad_v_init,
      GM_ADDR grad_k_partial,
      const AsPyKlifBackwardTilingData& tiling) {
    time_steps_ = tiling.timeSteps;
    neuron_count_ = tiling.neuronCount;
    tile_length_ = tiling.tileLength;
    tile_count_ = tiling.tileCount;
    threshold_ = tiling.vThreshold;
    reset_ = tiling.vReset;
    surrogate_alpha_ = tiling.surrogateAlpha;
    reciprocal_tau_ = tiling.reciprocalTau;
    decay_ = 1.0f - reciprocal_tau_;
    input_scale_ = tiling.decayInput != 0 ? reciprocal_tau_ : 1.0f;
    hard_reset_ = tiling.hardReset != 0;
    detach_reset_ = tiling.detachReset != 0;
    decay_input_ = tiling.decayInput != 0;
    scale_reset_ = tiling.scaleReset != 0;
    block_index_ = GetBlockIdx();
    block_count_ = GetBlockNum();
    x_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_seq));
    v_prev_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(v_prev_seq));
    h_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_seq));
    spike_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(spike_seq));
    grad_spike_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_spike_seq));
    grad_v_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_seq));
    grad_v_final_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_final));
    k_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(k));
    grad_x_seq_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_x_seq));
    grad_v_init_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_v_init));
    grad_k_partial_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(grad_k_partial));

    const uint32_t buffer_bytes = tile_length_ * sizeof(float);
    pipe_.InitBuffer(x_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(v_prev_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(h_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(spike_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(grad_spike_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(grad_v_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(output_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(scalar_queue_, 1, 32);
    pipe_.InitBuffer(calc_buffer_, 5U * buffer_bytes);
    LocalTensor<uint8_t> ub = calc_buffer_.Get<uint8_t>();
    carry_ = ub[0].ReinterpretCast<float>();
    surrogate_ = ub[buffer_bytes].ReinterpretCast<float>();
    scratch_ = ub[2U * buffer_bytes].ReinterpretCast<float>();
    partial_ = ub[3U * buffer_bytes].ReinterpretCast<float>();
    mask_ = ub[4U * buffer_bytes];
  }

  __aicore__ inline void Process() {
    LocalTensor<float> scalar = scalar_queue_.AllocTensor<float>();
    DataCopy(scalar, k_gm_[0], 8);
    scalar_queue_.EnQue(scalar);
    scalar = scalar_queue_.DeQue<float>();
    k_ = scalar.GetValue(0);
    scalar_queue_.FreeTensor(scalar);
    if (scale_reset_) {
      reciprocal_k_ = 1.0f / k_;
      reciprocal_k_squared_ = reciprocal_k_ * reciprocal_k_;
      effective_threshold_ = threshold_ * reciprocal_k_;
    } else {
      reciprocal_k_ = 0.0f;
      reciprocal_k_squared_ = 0.0f;
      effective_threshold_ = threshold_;
    }
    for (uint32_t tile = block_index_; tile < tile_count_; tile += block_count_) {
      const uint32_t begin = tile * tile_length_;
      const uint32_t count = begin + tile_length_ <= neuron_count_
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
    LocalTensor<float> initial = h_queue_.AllocTensor<float>();
    DataCopy(initial, grad_v_final_gm_[begin], count);
    h_queue_.EnQue(initial);
    initial = h_queue_.DeQue<float>();
    Adds(carry_, initial, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Duplicate(partial_, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    h_queue_.FreeTensor(initial);

    const float surrogate_scale = kHalfPi * surrogate_alpha_;
    const float surrogate_numerator = 0.5f * surrogate_alpha_;
    for (uint32_t reverse = 0; reverse < time_steps_; ++reverse) {
      const uint32_t time = time_steps_ - 1U - reverse;
      const uint64_t offset =
          static_cast<uint64_t>(time) * neuron_count_ + begin;

      LocalTensor<float> input = x_queue_.AllocTensor<float>();
      DataCopy(input, x_seq_gm_[offset], count);
      x_queue_.EnQue(input);
      LocalTensor<float> v_prev = v_prev_queue_.AllocTensor<float>();
      DataCopy(v_prev, v_prev_seq_gm_[offset], count);
      v_prev_queue_.EnQue(v_prev);
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

      input = x_queue_.DeQue<float>();
      v_prev = v_prev_queue_.DeQue<float>();
      h = h_queue_.DeQue<float>();
      spike = spike_queue_.DeQue<float>();
      grad_spike = grad_spike_queue_.DeQue<float>();
      grad_v = grad_v_queue_.DeQue<float>();

      // carry is dL/dv after the reset, including explicit v_seq loss.
      Add(carry_, carry_, grad_v, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Adds(grad_v, carry_, 0.0f, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

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

      if (scale_reset_) {
        // Accumulate the reset path's explicit k derivative before forming dL/dh.
        if (hard_reset_) {
          Muls(scratch_, spike, -1.0f, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Adds(scratch_, scratch_, 1.0f, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Mul(scratch_, scratch_, h, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Muls(scratch_, scratch_, -reciprocal_k_squared_, static_cast<int32_t>(count));
        } else {
          Muls(scratch_, spike, threshold_, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Sub(scratch_, scratch_, h, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Muls(scratch_, scratch_, reciprocal_k_squared_, static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
        Mul(scratch_, scratch_, grad_v, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(partial_, partial_, scratch_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();

        if (hard_reset_) {
          Muls(scratch_, spike, -1.0f, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Adds(scratch_, scratch_, 1.0f, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          Muls(scratch_, scratch_, reciprocal_k_, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          if (!detach_reset_) {
            Muls(spike, h, -reciprocal_k_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Adds(spike, spike, reset_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Mul(spike, spike, surrogate_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Add(scratch_, scratch_, spike, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
          }
        } else {
          Duplicate(scratch_, reciprocal_k_, static_cast<int32_t>(count));
          PipeBarrier<PIPE_V>();
          if (!detach_reset_) {
            Muls(spike, surrogate_, -effective_threshold_, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Add(scratch_, scratch_, spike, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
          }
        }
      } else if (hard_reset_) {
        Muls(scratch_, spike, -1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Adds(scratch_, scratch_, 1.0f, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        if (!detach_reset_) {
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

      Mul(grad_spike, grad_spike, surrogate_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Mul(carry_, carry_, scratch_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Add(carry_, carry_, grad_spike, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      // h=relu(k*h_pre): PyTorch's ReLU derivative is zero at h==0.
      const uint32_t compare_count =
          (count + kVectorRepeat - 1U) / kVectorRepeat * kVectorRepeat;
      CompareScalar(mask_, h, 0.0f, CMPMODE::GT, compare_count);
      PipeBarrier<PIPE_V>();
      Duplicate(grad_spike, 1.0f, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Select(
          grad_spike, mask_, grad_spike, 0.0f,
          SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
      PipeBarrier<PIPE_V>();
      Mul(carry_, carry_, grad_spike, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      // Recreate h_pre in public charge order for the k partial.
      if (decay_input_) {
        Adds(scratch_, v_prev, -reset_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(scratch_, input, scratch_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Muls(scratch_, scratch_, reciprocal_tau_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(scratch_, v_prev, scratch_, static_cast<int32_t>(count));
      } else {
        Adds(scratch_, v_prev, -reset_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Muls(scratch_, scratch_, reciprocal_tau_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(scratch_, v_prev, scratch_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(scratch_, scratch_, input, static_cast<int32_t>(count));
      }
      PipeBarrier<PIPE_V>();
      Mul(scratch_, scratch_, carry_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Add(partial_, partial_, scratch_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      Muls(carry_, carry_, k_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Muls(scratch_, carry_, input_scale_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      CopyOut(grad_x_seq_gm_, offset, scratch_, count);
      Muls(carry_, carry_, decay_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();

      x_queue_.FreeTensor(input);
      v_prev_queue_.FreeTensor(v_prev);
      h_queue_.FreeTensor(h);
      spike_queue_.FreeTensor(spike);
      grad_spike_queue_.FreeTensor(grad_spike);
      grad_v_queue_.FreeTensor(grad_v);
    }

    CopyOut(grad_v_init_gm_, begin, carry_, count);
    CopyOut(grad_k_partial_gm_, begin, partial_, count);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> x_queue_;
  TQue<QuePosition::VECIN, 1> v_prev_queue_;
  TQue<QuePosition::VECIN, 1> h_queue_;
  TQue<QuePosition::VECIN, 1> spike_queue_;
  TQue<QuePosition::VECIN, 1> grad_spike_queue_;
  TQue<QuePosition::VECIN, 1> grad_v_queue_;
  TQue<QuePosition::VECIN, 1> scalar_queue_;
  TQue<QuePosition::VECOUT, 1> output_queue_;
  TBuf<QuePosition::VECCALC> calc_buffer_;
  LocalTensor<float> carry_;
  LocalTensor<float> surrogate_;
  LocalTensor<float> scratch_;
  LocalTensor<float> partial_;
  LocalTensor<uint8_t> mask_;
  GlobalTensor<float> x_seq_gm_;
  GlobalTensor<float> v_prev_seq_gm_;
  GlobalTensor<float> h_seq_gm_;
  GlobalTensor<float> spike_seq_gm_;
  GlobalTensor<float> grad_spike_seq_gm_;
  GlobalTensor<float> grad_v_seq_gm_;
  GlobalTensor<float> grad_v_final_gm_;
  GlobalTensor<float> k_gm_;
  GlobalTensor<float> grad_x_seq_gm_;
  GlobalTensor<float> grad_v_init_gm_;
  GlobalTensor<float> grad_k_partial_gm_;
  uint32_t time_steps_;
  uint32_t neuron_count_;
  uint32_t tile_length_;
  uint32_t tile_count_;
  uint32_t block_index_;
  uint32_t block_count_;
  float threshold_;
  float reset_;
  float surrogate_alpha_;
  float reciprocal_tau_;
  float decay_;
  float input_scale_;
  float k_;
  float reciprocal_k_;
  float reciprocal_k_squared_;
  float effective_threshold_;
  bool hard_reset_;
  bool detach_reset_;
  bool decay_input_;
  bool scale_reset_;
};
}  // namespace

extern "C" __global__ __aicore__ void as_py_klif_backward(
    GM_ADDR xSeq,
    GM_ADDR vPrevSeq,
    GM_ADDR hSeq,
    GM_ADDR spikeSeq,
    GM_ADDR gradSpikeSeq,
    GM_ADDR gradVSeq,
    GM_ADDR gradVFinal,
    GM_ADDR k,
    GM_ADDR gradXSeq,
    GM_ADDR gradVInit,
    GM_ADDR gradKPartial,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  KernelAsPyKlifBackward kernel;
  kernel.Init(
      xSeq, vPrevSeq, hSeq, spikeSeq, gradSpikeSeq, gradVSeq, gradVFinal, k,
      gradXSeq, gradVInit, gradKPartial, tiling_data);
  kernel.Process();
}
