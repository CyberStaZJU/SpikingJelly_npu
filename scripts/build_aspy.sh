#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/native/aspy"
PYTHON_BIN="${PYTHON:-python3}"
STATE_ROOT="${SPIKINGJELLY_NPU_ASPY_BUILD_ROOT:-}"
if [[ -z "$STATE_ROOT" ]]; then
  echo "SPIKINGJELLY_NPU_ASPY_BUILD_ROOT must name an external, empty build directory" >&2
  exit 2
fi
mkdir -p "$STATE_ROOT"
STATE_ROOT="$(cd "$STATE_ROOT" && pwd)"
case "$STATE_ROOT/" in
  "$ROOT/"*)
    echo "AsPy build state must remain outside the source repository" >&2
    exit 2
    ;;
esac
if find "$STATE_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty AsPy build directory: $STATE_ROOT" >&2
  exit 2
fi

: "${ASCEND_TOOLKIT_HOME:?source the qualified CANN environment first}"
MSOPGEN="${MSOPGEN:-$ASCEND_TOOLKIT_HOME/python/site-packages/bin/msopgen}"
if [[ ! -x "$MSOPGEN" ]]; then
  MSOPGEN="$(command -v msopgen || true)"
fi
if [[ -z "$MSOPGEN" || ! -x "$MSOPGEN" ]]; then
  echo "msopgen was not found in the qualified CANN environment" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter was not found: $PYTHON_BIN" >&2
  exit 2
fi

RUNTIME_JSON="$($PYTHON_BIN - <<'PY'
import json
import platform
import sys

result = {
    "architecture": platform.machine().lower(),
    "implementation": platform.python_implementation(),
    "python": platform.python_version(),
    "python_major": sys.version_info.major,
    "python_minor": sys.version_info.minor,
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
if ! "$PYTHON_BIN" - "$RUNTIME_JSON" <<'PY'
import json
import sys

info = json.loads(sys.argv[1])
ok = (
    info.get("platform") == "linux"
    and info.get("architecture") in {"aarch64", "arm64"}
    and info.get("implementation") == "CPython"
    and info.get("python_major") == 3
    and info.get("python_minor") == 10
    and info.get("torch") == "2.9.0"
    and str(info.get("torch_npu", "")).startswith("2.9.0")
)
raise SystemExit(0 if ok else 1)
PY
then
  echo "AsPy source build requires Linux aarch64, CPython 3.10, torch 2.9.0, and torch-npu 2.9.0; observed: $RUNTIME_JSON" >&2
  exit 3
fi
CANN_VERSION_TEXT="$(
  {
    find "$ASCEND_TOOLKIT_HOME" -maxdepth 3 -type f \
      \( -name 'version.info' -o -name 'version.cfg' -o -name 'ascend_toolkit_install.info' \) \
      -print0 2>/dev/null | xargs -0 cat 2>/dev/null
    printf '%s\n' "$ASCEND_TOOLKIT_HOME"
  } | grep -Eo '8\.5(\.0)?' | head -1 || true
)"
if [[ -z "$CANN_VERSION_TEXT" ]]; then
  echo "could not verify qualified CANN 8.5 from ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME" >&2
  exit 3
fi

PROJECT="$STATE_ROOT/project"
BUILD="$STATE_ROOT/op-build"
INSTALL="$STATE_ROOT/op-install"
BRIDGE_SOURCE="$STATE_ROOT/bridge-source"
BRIDGE_BUILD="$STATE_ROOT/bridge-build"
ENV_FILE="$STATE_ROOT/activate_aspy.sh"
MANIFEST_FILE="$STATE_ROOT/build-manifest.json"

SOURCE_HEAD="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
if git -C "$ROOT" diff --quiet --ignore-submodules HEAD -- 2>/dev/null && \
   [[ -z "$(git -C "$ROOT" ls-files --others --exclude-standard 2>/dev/null)" ]]; then
  SOURCE_DIRTY=false
else
  SOURCE_DIRTY=true
fi
SOURCE_DIGEST_JSON="$($PYTHON_BIN - "$ROOT" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
try:
    output = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout
    relative_paths = sorted(
        relative
        for encoded in output.split(b"\0")
        if encoded
        for relative in [encoded.decode("utf-8", "surrogateescape")]
        if (root / relative).is_file()
    )
    scope = "git tracked plus untracked non-ignored files"
except (FileNotFoundError, subprocess.CalledProcessError):
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
    }
    relative_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in excluded_parts or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
        and not path.name.startswith("._")
        and not path.name.endswith((".pyc", ".pyo"))
    )
    scope = "source snapshot files excluding VCS, caches, environments, and build outputs"

aggregate = hashlib.sha256()
for relative in relative_paths:
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    aggregate.update(relative.encode("utf-8", "surrogateescape"))
    aggregate.update(b"\0")
    aggregate.update(str(size).encode())
    aggregate.update(b"\0")
    aggregate.update(digest.encode())
    aggregate.update(b"\n")
print(
    json.dumps(
        {
            "algorithm": "sha256(path\\0size\\0sha256\\n)",
            "file_count": len(relative_paths),
            "scope": scope,
            "tree_sha256": aggregate.hexdigest(),
        },
        sort_keys=True,
    )
)
PY
)"
CMAKE_VERSION="$(cmake --version | head -1)"
COMPILER_VERSION="$(${CXX:-c++} --version 2>/dev/null | head -1 || printf unavailable)"
MSOPGEN_VERSION="$($MSOPGEN --version 2>&1 | head -1 || printf unavailable)"

"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_fedsnn_decay_lif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -op AsPyFedSNNDecayLifForward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_fedsnn_decay_lif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyFedSNNDecayLifBackward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_if_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyIfForward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_if_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyIfBackward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_lif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyLifForward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_lif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyLifBackward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_klif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyKlifForward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_klif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyKlifBackward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_plif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyPlifForward \
  -lan cpp
"$MSOPGEN" gen \
  -i "$SOURCE/definition/aspy_plif_forward.json" \
  -f aclnn \
  -c ai_core-ascend910b \
  -out "$PROJECT" \
  -m 1 \
  -op AsPyPlifBackward \
  -lan cpp
install -m 0644 "$SOURCE/op_host/as_py_fed_snn_decay_lif_forward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_fed_snn_decay_lif_forward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_fed_snn_decay_lif_backward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_fed_snn_decay_lif_backward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_if_forward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_if_forward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_if_backward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_if_backward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_lif_forward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_lif_forward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_lif_backward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_lif_backward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_klif_forward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_klif_forward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_klif_backward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_klif_backward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_plif_forward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_plif_forward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_plif_backward.cpp" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_host/as_py_plif_backward_tiling.h" "$PROJECT/op_host/"
install -m 0644 "$SOURCE/op_kernel/as_py_fed_snn_decay_lif_forward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_fed_snn_decay_lif_backward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_if_forward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_if_backward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_lif_forward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_lif_backward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_klif_forward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_klif_backward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_plif_forward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/as_py_plif_backward.cpp" "$PROJECT/op_kernel/"
install -m 0644 "$SOURCE/op_kernel/CMakeLists.txt" "$PROJECT/op_kernel/"

cmake -S "$PROJECT" -B "$BUILD" \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DCMAKE_INSTALL_PREFIX="$INSTALL" \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_TOOLKIT_HOME"
cmake --build "$BUILD" --target install -- -j

mkdir -p "$BRIDGE_SOURCE"
install -m 0644 "$SOURCE/bridge/aspy_bridge.cpp" "$BRIDGE_SOURCE/"
install -m 0644 "$SOURCE/bridge/setup.py" "$BRIDGE_SOURCE/"
(
  cd "$BRIDGE_SOURCE"
  SPIKINGJELLY_NPU_ASPY_OP_API="$INSTALL/op_api" \
    "$PYTHON_BIN" setup.py build --build-base "$BRIDGE_BUILD"
)
EXTENSION_FILE="$(find "$BRIDGE_BUILD" -type f -name '_spikingjelly_npu_aspy*.so' -print -quit)"
if [[ -z "$EXTENSION_FILE" || ! -f "$EXTENSION_FILE" ]]; then
  echo "AsPy bridge build did not produce the expected extension" >&2
  exit 1
fi
EXTENSION_DIR="$(dirname "$EXTENSION_FILE")"

cat >"$ENV_FILE" <<EOF
# Generated by scripts/build_aspy.sh; source after scripts/cann_env.sh.
export PYTHONPATH="$ROOT/src:$EXTENSION_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="$INSTALL/op_api/lib:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export SPIKINGJELLY_NPU_ASPY_EXPECT_NATIVE=1
EOF
chmod 0600 "$ENV_FILE"
CAPABILITIES_JSON="$(
  PYTHONPATH="$ROOT/src:$EXTENSION_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH="$INSTALL/op_api/lib:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PYTHON_BIN" - <<'PY'
import importlib
import json

import torch  # noqa: F401
import torch_npu  # noqa: F401

module = importlib.import_module("_spikingjelly_npu_aspy")
required = (
    "if_forward",
    "if_backward",
    "lif_forward",
    "lif_backward",
    "klif_forward",
    "klif_backward",
    "plif_forward",
    "plif_backward",
    "fedsnn_decay_lif_forward",
    "fedsnn_decay_lif_backward",
)
missing = [name for name in required if not callable(getattr(module, name, None))]
if missing:
    raise SystemExit(f"built AsPy extension is missing required symbols: {missing}")
print(json.dumps({"required_symbols": list(required)}, sort_keys=True))
PY
)"
"$PYTHON_BIN" - \
  "$MANIFEST_FILE" \
  "$ROOT" \
  "$SOURCE_HEAD" \
  "$SOURCE_DIRTY" \
  "$SOURCE_DIGEST_JSON" \
  "$CAPABILITIES_JSON" \
  "$RUNTIME_JSON" \
  "$ASCEND_TOOLKIT_HOME" \
  "$CANN_VERSION_TEXT" \
  "$MSOPGEN" \
  "$MSOPGEN_VERSION" \
  "$CMAKE_VERSION" \
  "$COMPILER_VERSION" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    source_root,
    source_head,
    source_dirty,
    source_digest_json,
    capabilities_json,
    runtime_json,
    ascend_toolkit_home,
    cann_version,
    msopgen,
    msopgen_version,
    cmake_version,
    compiler_version,
) = sys.argv[1:]
manifest = {
    "ascend_compute_unit": "ascend910b",
    "capabilities": json.loads(capabilities_json),
    "ascend_toolkit_home": ascend_toolkit_home,
    "cann_version": cann_version,
    "cmake": cmake_version,
    "compiler": compiler_version,
    "msopgen": {"path": msopgen, "version": msopgen_version},
    "python_interpreter": sys.executable,
    "runtime": json.loads(runtime_json),
    "source": {
        "dirty": source_dirty == "true",
        "git_head": source_head,
        "input_digest": json.loads(source_digest_json),
        "root": source_root,
    },
}
Path(output).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
chmod 0600 "$MANIFEST_FILE"
printf 'AsPy build completed. Source %s after the CANN environment.\n' "$ENV_FILE"
printf 'Build identity written to %s\n' "$MANIFEST_FILE"
