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
  --require-native       Fail unless the qualified AsPy bundle can be installed
  --check                Inspect compatibility without installing
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

command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }

PYTHON_INFO="$($PYTHON_BIN - <<'PY'
import json, platform, sys
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
if "$PYTHON_BIN" - "$PYTHON_INFO" <<'PY'
import json, sys
info = json.loads(sys.argv[1])
raise SystemExit(0 if "torch" in info else 1)
PY
then
  :
else
  echo "PyTorch must be installed before SpikingJelly_npu; the installer never replaces it." >&2
  exit 3
fi

if [[ "$CHECK_ONLY" == 1 ]]; then
  exit 0
fi

BASE_URL="${SPIKINGJELLY_NPU_BASE_URL:-https://github.com/${REPOSITORY}/releases/download/${VERSION}}"
WHEEL="spikingjelly_npu-0.1.0-py3-none-any.whl"
BUNDLE="spikingjelly_npu_aspy-0.1.0-cann8.5-torch2.9-torchnpu2.9-cp310-linux_aarch64.tar.gz"
CHECKSUMS="SHA256SUMS"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/spikingjelly-npu-install.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

curl -fL --retry 3 -o "$TMP/$CHECKSUMS" "$BASE_URL/$CHECKSUMS"
curl -fL --retry 3 -o "$TMP/$WHEEL" "$BASE_URL/$WHEEL"
(
  cd "$TMP"
  grep "  $WHEEL\$" "$CHECKSUMS" | shasum -a 256 -c -
)
"$PYTHON_BIN" -m pip install --upgrade --no-deps "$TMP/$WHEEL"

NATIVE_OK="$($PYTHON_BIN - "$PYTHON_INFO" <<'PY'
import json, sys
info = json.loads(sys.argv[1])
ok = (
    info.get("implementation") == "CPython"
    and info.get("major") == 3
    and info.get("minor") == 10
    and info.get("machine") in {"aarch64", "arm64"}
    and info.get("platform") == "linux"
    and info.get("torch") == "2.9.0"
    and str(info.get("torch_npu", "")).startswith("2.9.0")
)
print("yes" if ok else "no")
PY
)"

if [[ "$INSTALL_NATIVE" != never && "$NATIVE_OK" == yes ]]; then
  curl -fL --retry 3 -o "$TMP/$BUNDLE" "$BASE_URL/$BUNDLE"
  (
    cd "$TMP"
    grep "  $BUNDLE\$" "$CHECKSUMS" | shasum -a 256 -c -
  )
  DEST="$PREFIX/$VERSION/aspy"
  mkdir -p "$DEST"
  tar -xzf "$TMP/$BUNDLE" -C "$DEST" --strip-components=1
  "$PYTHON_BIN" - "$DEST" <<'PY'
from pathlib import Path
import os, site, sys
root = Path(sys.argv[1]).resolve()
configured_target = os.environ.get("PIP_TARGET")
targets = [configured_target] if configured_target else site.getsitepackages()
if not configured_target and site.ENABLE_USER_SITE:
    targets.append(site.getusersitepackages())
target = next((Path(p) for p in targets if p and Path(p).is_dir() and os.access(p, os.W_OK)), None)
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
elif [[ "$INSTALL_NATIVE" == require ]]; then
  echo "The native bundle requires Linux aarch64, CPython 3.10, torch 2.9.0 and torch-npu 2.9.0." >&2
  exit 3
else
  echo "Installed the pure-Python package; this environment does not match the qualified native bundle."
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
