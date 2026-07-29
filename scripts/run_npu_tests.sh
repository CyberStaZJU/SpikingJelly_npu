#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/cann_env.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import os
import subprocess

try:
    selected = int(os.environ["ASCEND_DEVICE_ID"])
except ValueError as error:
    raise SystemExit("ASCEND_DEVICE_ID must be an integer device index") from error
output = subprocess.run(
    ["npu-smi", "info"],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
).stdout
in_process_table = False
busy = False
for line in output.splitlines():
    if "Process id" in line:
        in_process_table = True
        continue
    if not in_process_table:
        continue
    columns = [column.strip() for column in line.strip().strip("|").split("|")]
    if columns and columns[0].isdigit() and int(columns[0]) == selected:
        busy = True
        break
if busy:
    raise SystemExit(
        f"refusing to run: npu:{selected} already has a process in the npu-smi table"
    )
PY

python -m pytest -q tests -m "npu and not aspy"
