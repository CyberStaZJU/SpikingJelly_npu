#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsPyKlifForwardTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, timeSteps);
  TILING_DATA_FIELD_DEF(uint32_t, neuronCount);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileCount);
  TILING_DATA_FIELD_DEF(float, vThreshold);
  TILING_DATA_FIELD_DEF(float, vReset);
  TILING_DATA_FIELD_DEF(float, reciprocalTau);
  TILING_DATA_FIELD_DEF(uint32_t, hardReset);
  TILING_DATA_FIELD_DEF(uint32_t, decayInput);
  TILING_DATA_FIELD_DEF(uint32_t, scaleReset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsPyKlifForward, AsPyKlifForwardTilingData)
}  // namespace optiling
