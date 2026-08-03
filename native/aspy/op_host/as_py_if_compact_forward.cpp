#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "as_py_if_compact_forward_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kTileLength = 4096;

bool MatchesTrailingShape(
    const gert::Shape& sequence_shape,
    const gert::Shape& state_shape) {
  if (sequence_shape.GetDimNum() < 2 ||
      state_shape.GetDimNum() + 1 != sequence_shape.GetDimNum()) {
    return false;
  }
  for (std::size_t index = 1; index < sequence_shape.GetDimNum(); ++index) {
    if (sequence_shape.GetDim(index) < 0 || state_shape.GetDim(index - 1) < 0 ||
        sequence_shape.GetDim(index) != state_shape.GetDim(index - 1)) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* x_storage_shape = context->GetInputShape(0);
  const gert::StorageShape* v_storage_shape = context->GetInputShape(1);
  if (x_storage_shape == nullptr || v_storage_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape& shape = x_storage_shape->GetStorageShape();
  const gert::Shape& v_shape = v_storage_shape->GetStorageShape();
  if (!MatchesTrailingShape(shape, v_shape)) {
    return ge::GRAPH_FAILED;
  }

  const int64_t time_steps_dim = shape.GetDim(0);
  if (time_steps_dim <= 0) {
    return ge::GRAPH_FAILED;
  }
  uint64_t neuron_count = 1;
  for (std::size_t index = 1; index < shape.GetDimNum(); ++index) {
    const int64_t dimension = shape.GetDim(index);
    if (dimension <= 0 ||
        neuron_count > UINT32_MAX / static_cast<uint64_t>(dimension)) {
      return ge::GRAPH_FAILED;
    }
    neuron_count *= static_cast<uint64_t>(dimension);
  }
  const uint64_t time_steps = static_cast<uint64_t>(time_steps_dim);
  if (time_steps > UINT32_MAX || neuron_count > UINT32_MAX ||
      (neuron_count & 7U) != 0U) {
    return ge::GRAPH_FAILED;
  }

  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetFloat(0) == nullptr ||
      attrs->GetFloat(1) == nullptr || attrs->GetBool(2) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float v_threshold = *attrs->GetFloat(0);
  const float v_reset = *attrs->GetFloat(1);
  if (!std::isfinite(v_threshold) || !std::isfinite(v_reset)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tile_count =
      static_cast<uint32_t>((neuron_count + kTileLength - 1) / kTileLength);
  AsPyIfCompactForwardTilingData tiling;
  tiling.set_timeSteps(static_cast<uint32_t>(time_steps));
  tiling.set_neuronCount(static_cast<uint32_t>(neuron_count));
  tiling.set_tileLength(kTileLength);
  tiling.set_tileCount(tile_count);
  tiling.set_vThreshold(v_threshold);
  tiling.set_vReset(v_reset);
  tiling.set_hardReset(*attrs->GetBool(2) ? 1U : 0U);

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
  *context->GetOutputShape(1) = *v_shape;
  *context->GetOutputShape(2) = *x_shape;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);
  context->SetOutputDataType(1, dtype);
  context->SetOutputDataType(2, dtype);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AsPyIfCompactForward : public OpDef {
 public:
  explicit AsPyIfCompactForward(const char* name) : OpDef(name) {
    this->Input("xSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("vInit").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("spikeSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("vFinal").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("hSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("vThreshold").Float();
    this->Attr("vReset").Float();
    this->Attr("hardReset").Bool();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AsPyIfCompactForward);
}  // namespace ops
