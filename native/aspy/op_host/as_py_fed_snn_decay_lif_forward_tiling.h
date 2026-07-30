#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsPyFedSNNDecayLifForwardTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, timeSteps);
  TILING_DATA_FIELD_DEF(uint32_t, neuronCount);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileCount);
  TILING_DATA_FIELD_DEF(float, membraneDecay);
  TILING_DATA_FIELD_DEF(float, vThreshold);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    AsPyFedSNNDecayLifForward,
    AsPyFedSNNDecayLifForwardTilingData)
}  // namespace optiling
