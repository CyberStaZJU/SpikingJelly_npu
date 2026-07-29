#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kVectorRepeat = 64;
constexpr uint32_t kCalcUbBytes = 128 * 1024;

class KernelAsPyLifForward {
 public:
  __aicore__ inline KernelAsPyLifForward() = default;

  __aicore__ inline void Init(
      GM_ADDR x_seq,
      GM_ADDR v_init,
      GM_ADDR spike_seq,
      GM_ADDR v_seq,
      GM_ADDR v_final,
      GM_ADDR h_seq,
      const AsPyLifForwardTilingData& tiling) {
    time_steps_ = tiling.timeSteps;
    neuron_count_ = tiling.neuronCount;
    tile_length_ = tiling.tileLength;
    tile_count_ = tiling.tileCount;
    threshold_ = tiling.vThreshold;
    reset_ = tiling.vReset;
    reciprocal_tau_ = tiling.reciprocalTau;
    hard_reset_ = tiling.hardReset != 0;
    decay_input_ = tiling.decayInput != 0;
    block_index_ = GetBlockIdx();
    block_count_ = GetBlockNum();
    x_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x_seq));
    v_init_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v_init));
    spike_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(spike_seq));
    v_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v_seq));
    v_final_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(v_final));
    h_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_seq));

    const uint32_t buffer_bytes = tile_length_ * sizeof(float);
    pipe_.InitBuffer(input_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(output_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(calc_buffer_, kCalcUbBytes);
    LocalTensor<uint8_t> ub = calc_buffer_.Get<uint8_t>();
    voltage_ = ub[0].ReinterpretCast<float>();
    spike_ = ub[buffer_bytes].ReinterpretCast<float>();
    scratch_ = ub[2U * buffer_bytes].ReinterpretCast<float>();
    mask_ = ub[3U * buffer_bytes];
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
    LocalTensor<float> initial = input_queue_.AllocTensor<float>();
    DataCopy(initial, v_init_gm_[begin], count);
    input_queue_.EnQue(initial);
    initial = input_queue_.DeQue<float>();
    Adds(voltage_, initial, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    input_queue_.FreeTensor(initial);

    for (uint32_t time = 0; time < time_steps_; ++time) {
      const uint64_t offset =
          static_cast<uint64_t>(time) * neuron_count_ + begin;
      LocalTensor<float> input = input_queue_.AllocTensor<float>();
      DataCopy(input, x_seq_gm_[offset], count);
      input_queue_.EnQue(input);
      input = input_queue_.DeQue<float>();

      // Match eager LIF operation ordering rather than using an algebraically
      // simplified expression, which can change FP32 spike decisions.
      Adds(scratch_, voltage_, -reset_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      if (decay_input_) {
        Sub(scratch_, input, scratch_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Muls(scratch_, scratch_, reciprocal_tau_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(voltage_, voltage_, scratch_, static_cast<int32_t>(count));
      } else {
        Muls(scratch_, scratch_, reciprocal_tau_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(voltage_, voltage_, scratch_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Add(voltage_, voltage_, input, static_cast<int32_t>(count));
      }
      PipeBarrier<PIPE_V>();
      input_queue_.FreeTensor(input);

      CopyOut(h_seq_gm_, offset, voltage_, count);

      const uint32_t compare_count =
          (count + kVectorRepeat - 1U) / kVectorRepeat * kVectorRepeat;
      CompareScalar(mask_, voltage_, threshold_, CMPMODE::GE, compare_count);
      PipeBarrier<PIPE_V>();
      Duplicate(spike_, 1.0f, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Select(
          spike_,
          mask_,
          spike_,
          0.0f,
          SELMODE::VSEL_TENSOR_SCALAR_MODE,
          count);
      PipeBarrier<PIPE_V>();
      CopyOut(spike_seq_gm_, offset, spike_, count);

      if (hard_reset_) {
        Duplicate(scratch_, reset_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Select(
            voltage_,
            mask_,
            scratch_,
            voltage_,
            SELMODE::VSEL_TENSOR_TENSOR_MODE,
            count);
      } else {
        Muls(scratch_, spike_, threshold_, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(voltage_, voltage_, scratch_, static_cast<int32_t>(count));
      }
      PipeBarrier<PIPE_V>();
      CopyOut(v_seq_gm_, offset, voltage_, count);
    }

    CopyOut(v_final_gm_, begin, voltage_, count);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> input_queue_;
  TQue<QuePosition::VECOUT, 1> output_queue_;
  TBuf<QuePosition::VECCALC> calc_buffer_;
  LocalTensor<float> voltage_;
  LocalTensor<float> spike_;
  LocalTensor<float> scratch_;
  LocalTensor<uint8_t> mask_;
  GlobalTensor<float> x_seq_gm_;
  GlobalTensor<float> v_init_gm_;
  GlobalTensor<float> spike_seq_gm_;
  GlobalTensor<float> v_seq_gm_;
  GlobalTensor<float> v_final_gm_;
  GlobalTensor<float> h_seq_gm_;
  uint32_t time_steps_;
  uint32_t neuron_count_;
  uint32_t tile_length_;
  uint32_t tile_count_;
  uint32_t block_index_;
  uint32_t block_count_;
  float threshold_;
  float reset_;
  float reciprocal_tau_;
  bool hard_reset_;
  bool decay_input_;
};
}  // namespace

extern "C" __global__ __aicore__ void as_py_lif_forward(
    GM_ADDR xSeq,
    GM_ADDR vInit,
    GM_ADDR spikeSeq,
    GM_ADDR vSeq,
    GM_ADDR vFinal,
    GM_ADDR hSeq,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  KernelAsPyLifForward kernel;
  kernel.Init(xSeq, vInit, spikeSeq, vSeq, vFinal, hSeq, tiling_data);
  kernel.Process();
}
