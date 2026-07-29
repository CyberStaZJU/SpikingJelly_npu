#include <algorithm>
#include <cstdint>

#include "as_py_lif_forward_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kTileLength = 4096;
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  const gert::Shape& shape = x_shape->GetStorageShape();
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

  const float tau = *context->GetAttrs()->GetFloat(3);
  if (!(tau > 1.0f)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tile_count =
      static_cast<uint32_t>((neuron_count + kTileLength - 1) / kTileLength);
  AsPyLifForwardTilingData tiling;
  tiling.set_timeSteps(static_cast<uint32_t>(time_steps));
  tiling.set_neuronCount(static_cast<uint32_t>(neuron_count));
  tiling.set_tileLength(kTileLength);
  tiling.set_tileCount(tile_count);
  tiling.set_vThreshold(*context->GetAttrs()->GetFloat(0));
  tiling.set_vReset(*context->GetAttrs()->GetFloat(1));
  tiling.set_reciprocalTau(1.0f / tau);
  tiling.set_hardReset(*context->GetAttrs()->GetBool(2) ? 1U : 0U);
  tiling.set_decayInput(*context->GetAttrs()->GetBool(4) ? 1U : 0U);

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
  const gert::Shape* x_shape = context->GetInputShape(0);
  const gert::Shape* v_shape = context->GetInputShape(1);
  *context->GetOutputShape(0) = *x_shape;
  *context->GetOutputShape(1) = *x_shape;
  *context->GetOutputShape(2) = *v_shape;
  *context->GetOutputShape(3) = *x_shape;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);
  context->SetOutputDataType(1, dtype);
  context->SetOutputDataType(2, dtype);
  context->SetOutputDataType(3, dtype);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AsPyLifForward : public OpDef {
 public:
  explicit AsPyLifForward(const char* name) : OpDef(name) {
    this->Input("xSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("vInit").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("spikeSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("vSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("vFinal").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("hSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("vThreshold").Float();
    this->Attr("vReset").Float();
    this->Attr("hardReset").Bool();
    this->Attr("tau").Float();
    this->Attr("decayInput").Bool();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AsPyLifForward);
}  // namespace ops
