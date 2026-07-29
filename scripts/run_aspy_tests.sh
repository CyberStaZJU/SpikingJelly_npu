#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/cann_env.sh"
: "${SPIKINGJELLY_NPU_ASPY_BUILD_ROOT:?set the external AsPy build root first}"
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"
cd "$ROOT"

python - <<'PY'
import importlib.util
import os
import subprocess

selected = int(os.environ["ASCEND_DEVICE_ID"])
output = subprocess.run(
    ["npu-smi", "info"],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout
in_process_table = False
for line in output.splitlines():
    if "Process id" in line:
        in_process_table = True
        continue
    if not in_process_table:
        continue
    columns = [column.strip() for column in line.strip().strip("|").split("|")]
    if columns and columns[0].isdigit() and int(columns[0]) == selected:
        raise SystemExit(f"refusing to run: npu:{selected} already has a process")

for module_name in ("spikingjelly_npu_aspy", "_spikingjelly_npu_aspy"):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise SystemExit(f"required AsPy module is unavailable: {module_name}")
    print(f"{module_name}: {spec.origin}")
PY

python -m pytest -q tests/test_aspy_npu.py -m "npu and aspy"
