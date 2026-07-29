#!/usr/bin/env bash
# Source a user-provided or discoverable CANN environment without assuming a
# project-specific checkout. This file is intended to be sourced.

export PYTHONNOUSERSITE=1
export ASCEND_DEVICE_ID="${ASCEND_DEVICE_ID:-0}"
export DEVICE_ID="$ASCEND_DEVICE_ID"

if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}" ]]; then
  return 0 2>/dev/null || exit 0
fi

CANN_ENV="${SPIKINGJELLY_NPU_CANN_ENV:-}"
if [[ -z "$CANN_ENV" ]]; then
  candidates=(
    "${CONDA_PREFIX:-}/Ascend/cann/set_env.sh"
    "${HOME}/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      CANN_ENV="$candidate"
      break
    fi
  done
fi

if [[ -z "$CANN_ENV" || ! -f "$CANN_ENV" ]]; then
  echo "CANN environment was not found. Source CANN first or set" >&2
  echo "SPIKINGJELLY_NPU_CANN_ENV=/path/to/set_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi

_had_nounset=0
case "$-" in *u*) _had_nounset=1; set +u ;; esac
# shellcheck disable=SC1090
source "$CANN_ENV"
if [[ "$_had_nounset" == 1 ]]; then set -u; fi
unset _had_nounset CANN_ENV candidates candidate

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  echo "CANN set_env.sh did not define ASCEND_TOOLKIT_HOME" >&2
  return 2 2>/dev/null || exit 2
fi
