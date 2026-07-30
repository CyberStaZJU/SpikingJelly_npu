#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kVectorRepeat = 64;
constexpr uint32_t kCalcUbBytes = 128 * 1024;

class KernelAsPyFedSNNDecayLifForward {
 public:
  __aicore__ inline KernelAsPyFedSNNDecayLifForward() = default;

  __aicore__ inline void Init(
      GM_ADDR current_seq,
      GM_ADDR spike_seq,
      GM_ADDR h_seq,
      const AsPyFedSNNDecayLifForwardTilingData& tiling) {
    time_steps_ = tiling.timeSteps;
    neuron_count_ = tiling.neuronCount;
    tile_length_ = tiling.tileLength;
    tile_count_ = tiling.tileCount;
    membrane_decay_ = tiling.membraneDecay;
    threshold_ = tiling.vThreshold;
    block_index_ = GetBlockIdx();
    block_count_ = GetBlockNum();
    current_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(current_seq));
    spike_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(spike_seq));
    h_seq_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(h_seq));

    const uint32_t buffer_bytes = tile_length_ * sizeof(float);
    pipe_.InitBuffer(input_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(output_queue_, 1, buffer_bytes);
    pipe_.InitBuffer(calc_buffer_, kCalcUbBytes);
    LocalTensor<uint8_t> ub = calc_buffer_.Get<uint8_t>();
    membrane_ = ub[0].ReinterpretCast<float>();
    charged_ = ub[buffer_bytes].ReinterpretCast<float>();
    spike_ = ub[2U * buffer_bytes].ReinterpretCast<float>();
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
    Duplicate(membrane_, 0.0f, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();

    for (uint32_t time = 0; time < time_steps_; ++time) {
      const uint64_t offset =
          static_cast<uint64_t>(time) * neuron_count_ + begin;
      LocalTensor<float> current = input_queue_.AllocTensor<float>();
      DataCopy(current, current_seq_gm_[offset], count);
      input_queue_.EnQue(current);
      current = input_queue_.DeQue<float>();

      // Preserve FedSNN's exact FP32 operation order.
      Muls(charged_, membrane_, membrane_decay_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Add(charged_, charged_, current, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      input_queue_.FreeTensor(current);
      CopyOut(h_seq_gm_, offset, charged_, count);

      const uint32_t compare_count =
          (count + kVectorRepeat - 1U) / kVectorRepeat * kVectorRepeat;
      CompareScalar(mask_, charged_, threshold_, CMPMODE::GE, compare_count);
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

      Muls(membrane_, spike_, threshold_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
      Sub(membrane_, charged_, membrane_, static_cast<int32_t>(count));
      PipeBarrier<PIPE_V>();
    }
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> input_queue_;
  TQue<QuePosition::VECOUT, 1> output_queue_;
  TBuf<QuePosition::VECCALC> calc_buffer_;
  LocalTensor<float> membrane_;
  LocalTensor<float> charged_;
  LocalTensor<float> spike_;
  LocalTensor<uint8_t> mask_;
  GlobalTensor<float> current_seq_gm_;
  GlobalTensor<float> spike_seq_gm_;
  GlobalTensor<float> h_seq_gm_;
  uint32_t time_steps_;
  uint32_t neuron_count_;
  uint32_t tile_length_;
  uint32_t tile_count_;
  uint32_t block_index_;
  uint32_t block_count_;
  float membrane_decay_;
  float threshold_;
};
}  // namespace

extern "C" __global__ __aicore__ void as_py_fed_snn_decay_lif_forward(
    GM_ADDR currentSeq,
    GM_ADDR spikeSeq,
    GM_ADDR hSeq,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tiling_data, tiling);
  KernelAsPyFedSNNDecayLifForward kernel;
  kernel.Init(currentSeq, spikeSeq, hSeq, tiling_data);
  kernel.Process();
}
