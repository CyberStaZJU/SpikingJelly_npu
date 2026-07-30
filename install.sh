#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="${SPIKINGJELLY_NPU_REPOSITORY:-CyberStaZJU/SpikingJelly_npu}"
VERSION="${SPIKINGJELLY_NPU_VERSION:-v0.1.0-alpha.1}"
PYTHON_BIN="${PYTHON:-python3}"
PREFIX="${SPIKINGJELLY_NPU_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/spikingjelly_npu}"
INSTALL_NATIVE=auto
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Install SpikingJelly_npu from a GitHub Release.

Usage: install.sh [options]
  --version TAG          Release tag (default: v0.1.0-alpha.1)
  --python PATH          Python interpreter (default: python3)
  --prefix PATH          Native bundle root
  --fallback-only        Install the pure-Python wheel only
  --require-native       Fail unless the native bundle includes packed_aspy
  --check                Inspect Python, CANN, and Ascend compatibility only
  -h, --help             Show this help

Environment overrides:
  SPIKINGJELLY_NPU_REPOSITORY  GitHub owner/repository
  SPIKINGJELLY_NPU_VERSION     Release tag
  SPIKINGJELLY_NPU_PREFIX      Native installation root
  PYTHON                       Python interpreter
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --fallback-only) INSTALL_NATIVE=never; shift ;;
    --require-native) INSTALL_NATIVE=require; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }

PYTHON_INFO="$($PYTHON_BIN - <<'PY'
import json
import platform
import sys

result = {
    "implementation": platform.python_implementation(),
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "machine": platform.machine().lower(),
    "platform": sys.platform,
}
try:
    import torch
    result["torch"] = torch.__version__.split("+")[0]
except Exception as error:
    result["torch_error"] = str(error)
try:
    import torch_npu
    result["torch_npu"] = getattr(torch_npu, "__version__", "unknown").split("+")[0]
except Exception as error:
    result["torch_npu_error"] = str(error)
print(json.dumps(result, sort_keys=True))
PY
)"
printf 'Environment: %s\n' "$PYTHON_INFO"
if ! "$PYTHON_BIN" - "$PYTHON_INFO" <<'PY'
import json
import sys

info = json.loads(sys.argv[1])
raise SystemExit(0 if "torch" in info else 1)
PY
then
  echo "PyTorch must be installed before SpikingJelly_npu; the installer never replaces it." >&2
  exit 3
fi

TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-}}"
CANN_VERSION_TEXT="$({
  if [[ -n "$TOOLKIT_HOME" && -d "$TOOLKIT_HOME" ]]; then
    find "$TOOLKIT_HOME" -maxdepth 3 -type f \
      \( -name 'version.info' -o -name 'version.cfg' -o -name 'ascend_toolkit_install.info' \) \
      -print0 2>/dev/null | xargs -0 cat 2>/dev/null || true
    printf '%s\n' "$TOOLKIT_HOME"
  fi
} | grep -Eo '8\.5(\.0)?' | head -1 || true)"
ASCEND_PRODUCT="$(
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>/dev/null | grep -Eo '910B[^ |]*' | head -1 || true
  fi
)"

NATIVE_OK="$($PYTHON_BIN - "$PYTHON_INFO" "$CANN_VERSION_TEXT" "$ASCEND_PRODUCT" <<'PY'
import json
import sys

info = json.loads(sys.argv[1])
cann = sys.argv[2]
product = sys.argv[3]
ok = (
    info.get("implementation") == "CPython"
    and info.get("major") == 3
    and info.get("minor") == 10
    and info.get("machine") in {"aarch64", "arm64"}
    and info.get("platform") == "linux"
    and info.get("torch") == "2.9.0"
    and str(info.get("torch_npu", "")).startswith("2.9.0")
    and cann.startswith("8.5")
    and product.startswith("910B")
)
print("yes" if ok else "no")
PY
)"
printf 'CANN toolkit: %s (detected version: %s)\n' "${TOOLKIT_HOME:-not sourced}" "${CANN_VERSION_TEXT:-unknown}"
printf 'Ascend product: %s\n' "${ASCEND_PRODUCT:-not detected}"
printf 'Qualified native matrix: %s\n' "$NATIVE_OK"

if [[ "$CHECK_ONLY" == 1 ]]; then
  if [[ "$INSTALL_NATIVE" == require && "$NATIVE_OK" != yes ]]; then
    exit 3
  fi
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
BASE_URL="${SPIKINGJELLY_NPU_BASE_URL:-https://github.com/${REPOSITORY}/releases/download/${VERSION}}"
WHEEL="spikingjelly_npu-0.1.0-py3-none-any.whl"
BUNDLE="spikingjelly_npu_aspy-0.1.0-cann8.5-torch2.9-torchnpu2.9-cp310-linux_aarch64.tar.gz"
CHECKSUMS="SHA256SUMS"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/spikingjelly-npu-install.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

curl -fL --retry 5 --retry-all-errors --retry-delay 2 \
  -o "$TMP/$CHECKSUMS" "$BASE_URL/$CHECKSUMS"
curl -fL --retry 5 --retry-all-errors --retry-delay 2 \
  -o "$TMP/$WHEEL" "$BASE_URL/$WHEEL"
(
  cd "$TMP"
  grep "  $WHEEL\$" "$CHECKSUMS" | shasum -a 256 -c -
)

CAPABILITIES_JSON='{"fedsnn_decay_lif": false}'
STAGED_BUNDLE="$TMP/aspy-bundle"
if [[ "$INSTALL_NATIVE" != never && "$NATIVE_OK" == yes ]]; then
  curl -fL --retry 5 --retry-all-errors --retry-delay 2 \
    -o "$TMP/$BUNDLE" "$BASE_URL/$BUNDLE"
  (
    cd "$TMP"
    grep "  $BUNDLE\$" "$CHECKSUMS" | shasum -a 256 -c -
  )
  mkdir -p "$STAGED_BUNDLE"
  tar -xzf "$TMP/$BUNDLE" -C "$STAGED_BUNDLE" --strip-components=1
  CAPABILITIES_JSON="$($PYTHON_BIN - "$STAGED_BUNDLE" <<'PY'
import ctypes
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
library = root / "lib" / "libcust_opapi.so"
candidates = []
for suffix in importlib.machinery.EXTENSION_SUFFIXES:
    candidates.extend(sorted((root / "python").glob(f"_spikingjelly_npu_aspy*{suffix}")))
candidates = list(dict.fromkeys(candidates))
if not library.is_file() or len(candidates) != 1:
    raise SystemExit(
        f"invalid AsPy bundle layout: library={library.is_file()}, extensions={len(candidates)}"
    )

import torch  # noqa: F401
import torch_npu  # noqa: F401

ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
spec = importlib.util.spec_from_file_location("_spikingjelly_npu_aspy", candidates[0])
if spec is None or spec.loader is None:
    raise SystemExit(f"could not load AsPy extension spec from {candidates[0]}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
required_generic = (
    "if_forward",
    "if_backward",
    "lif_forward",
    "lif_backward",
    "plif_forward",
    "plif_backward",
)
missing_generic = [name for name in required_generic if not callable(getattr(module, name, None))]
if missing_generic:
    raise SystemExit(f"native bundle is missing required generic symbols: {missing_generic}")
print(
    json.dumps(
        {
            "klif": all(
                callable(getattr(module, name, None))
                for name in ("klif_forward", "klif_backward")
            ),
            "fedsnn_decay_lif": all(
                callable(getattr(module, name, None))
                for name in (
                    "fedsnn_decay_lif_forward",
                    "fedsnn_decay_lif_backward",
                )
            )
        },
        sort_keys=True,
    )
)
PY
)"
  printf 'AsPy capabilities: %s\n' "$CAPABILITIES_JSON"
  if ! "$PYTHON_BIN" - "$CAPABILITIES_JSON" <<'PY'
import json
import sys

raise SystemExit(0 if json.loads(sys.argv[1]).get("klif") is True else 1)
PY
  then
    echo "Warning: the selected release bundle lacks the KLIF forward/backward symbols." >&2
    echo "KLIF backend='aspy' will fall back or fail in strict mode; build current main from source for KLIF." >&2
  fi
  if ! "$PYTHON_BIN" - "$CAPABILITIES_JSON" <<'PY'
import json
import sys

raise SystemExit(0 if json.loads(sys.argv[1]).get("fedsnn_decay_lif") is True else 1)
PY
  then
    echo "Warning: the selected release bundle lacks the FedSNN decay-LIF symbols required by packed_aspy." >&2
    echo "The generic AsPy routes remain available, but packed_aspy will fall back or fail in strict mode." >&2
    echo "Build current main from source or select a newer release that advertises this capability." >&2
    if [[ "$INSTALL_NATIVE" == require ]]; then
      exit 4
    fi
  fi
elif [[ "$INSTALL_NATIVE" == require ]]; then
  echo "The native bundle requires Linux aarch64, CPython 3.10, torch 2.9.0, torch-npu 2.9.0, CANN 8.5, and Ascend 910B." >&2
  exit 3
fi

"$PYTHON_BIN" -m pip install --upgrade --no-deps "$TMP/$WHEEL"

if [[ -d "$STAGED_BUNDLE" ]]; then
  DEST="$PREFIX/$VERSION/aspy"
  mkdir -p "$DEST"
  cp -R "$STAGED_BUNDLE/." "$DEST/"
  "$PYTHON_BIN" - "$DEST" <<'PY'
from pathlib import Path
import os
import site
import sys

root = Path(sys.argv[1]).resolve()
configured_target = os.environ.get("PIP_TARGET")
targets = [configured_target] if configured_target else site.getsitepackages()
if not configured_target and site.ENABLE_USER_SITE:
    targets.append(site.getusersitepackages())
target = next(
    (Path(path) for path in targets if path and Path(path).is_dir() and os.access(path, os.W_OK)),
    None,
)
if target is None:
    raise SystemExit("no writable site-packages directory for AsPy bundle registration")
(target / "spikingjelly_npu_aspy_bundle.pth").write_text(
    f"import os; os.environ.setdefault('SPIKINGJELLY_NPU_ASPY_BUNDLE', {str(root)!r})\n",
    encoding="utf-8",
)
package = target / "spikingjelly_npu"
if package.is_dir():
    (package / "_aspy_bundle_path.txt").write_text(str(root) + "\n", encoding="utf-8")
print(f"Registered AsPy bundle: {root}")
PY
elif [[ "$INSTALL_NATIVE" != never ]]; then
  echo "Installed the pure-Python package; this environment does not match the qualified native matrix."
fi

PYTHONPATH="${PIP_TARGET:-}${PYTHONPATH:+:$PYTHONPATH}" \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" - <<'PY'
import sys
import spikingjelly_npu

assert "torch_npu" not in sys.modules
print("SpikingJelly_npu", spikingjelly_npu.__version__, "installed successfully")
PY

cat <<'EOF'

Python usage:
  import spikingjelly_npu

For unchanged imports from the qualified SpikingJelly subset:
  from spikingjelly_npu import enable_compat
  enable_compat()

For a dedicated Ascend process that still uses common CUDA conveniences:
  enable_compat(cuda=True)
This uses torch-npu's official transfer_to_npu compatibility module and is not
an emulator for CuPy arrays, custom CUDA extensions, or arbitrary CUDA APIs.
EOF
