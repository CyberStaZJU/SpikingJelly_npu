#include <algorithm>
#include <cmath>
#include <cstdint>

#include "as_py_fed_snn_decay_lif_forward_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kTileLength = 4096;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* current_shape = context->GetInputShape(0);
  const gert::Shape& shape = current_shape->GetStorageShape();
  if (shape.GetDimNum() < 2) {
    return ge::GRAPH_FAILED;
  }

  uint64_t time_steps = static_cast<uint64_t>(shape.GetDim(0));
  uint64_t neuron_count = 1;
  for (size_t index = 1; index < shape.GetDimNum(); ++index) {
    neuron_count *= static_cast<uint64_t>(shape.GetDim(index));
  }
  if (time_steps == 0 || neuron_count == 0 ||
      time_steps > UINT32_MAX || neuron_count > UINT32_MAX ||
      (neuron_count & 7U) != 0U) {
    return ge::GRAPH_FAILED;
  }

  const float membrane_decay = *context->GetAttrs()->GetFloat(0);
  if (!(membrane_decay >= 0.0f && membrane_decay <= 1.0f)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tile_count =
      static_cast<uint32_t>((neuron_count + kTileLength - 1) / kTileLength);
  AsPyFedSNNDecayLifForwardTilingData tiling;
  tiling.set_timeSteps(static_cast<uint32_t>(time_steps));
  tiling.set_neuronCount(static_cast<uint32_t>(neuron_count));
  tiling.set_tileLength(kTileLength);
  const float v_threshold = *context->GetAttrs()->GetFloat(1);
  if (!std::isfinite(v_threshold)) {
    return ge::GRAPH_FAILED;
  }
  tiling.set_tileCount(tile_count);
  tiling.set_membraneDecay(membrane_decay);
  tiling.set_vThreshold(v_threshold);

  constexpr uint32_t kAscend910BAivCores = 20U;
  context->SetBlockDim(std::min(tile_count, kAscend910BAivCores));
  tiling.SaveToBuffer(
      context->GetRawTilingData()->GetData(),
      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  *context->GetOutputShape(1) = *context->GetInputShape(0);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);
  context->SetOutputDataType(1, dtype);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AsPyFedSNNDecayLifForward : public OpDef {
 public:
  explicit AsPyFedSNNDecayLifForward(const char* name) : OpDef(name) {
    this->Input("currentSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("spikeSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("hSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("membraneDecay").Float();
    this->Attr("vThreshold").Float();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AsPyFedSNNDecayLifForward);
}  // namespace ops
