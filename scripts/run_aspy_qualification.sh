#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/cann_env.sh"
: "${SPIKINGJELLY_NPU_ASPY_BUILD_ROOT:?set the external AsPy build root first}"
: "${SPIKINGJELLY_NPU_ASPY_QUALIFICATION_ROOT:?set an external qualification root first}"
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"

QUALIFICATION_ROOT="$SPIKINGJELLY_NPU_ASPY_QUALIFICATION_ROOT"
mkdir -p "$QUALIFICATION_ROOT"
QUALIFICATION_ROOT="$(cd "$QUALIFICATION_ROOT" && pwd)"
case "$QUALIFICATION_ROOT/" in
  "$ROOT/"*)
    echo "AsPy qualification evidence must remain outside the source repository" >&2
    exit 2
    ;;
esac
if find "$QUALIFICATION_ROOT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty qualification directory: $QUALIFICATION_ROOT" >&2
  exit 2
fi

cd "$ROOT"
python - <<'PY'
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
PY

DEVICE="npu:${ASCEND_DEVICE_ID}"
IF_ROOT="$QUALIFICATION_ROOT/if"
PLIF_ROOT="$QUALIFICATION_ROOT/plif"
mkdir -p "$IF_ROOT" "$PLIF_ROOT"

for RUN in 1 2 3; do
  python scripts/aspy_if_qualification.py \
    --device "$DEVICE" \
    --mode determinism \
    --seed 20260729 \
    >"$IF_ROOT/determinism-$RUN.json"
done

python - "$IF_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = [
    json.loads((root / f"determinism-{run}.json").read_text())["determinism"]
    for run in (1, 2, 3)
]
if records[1:] != records[:-1]:
    raise SystemExit(f"fresh-process AsPy IF determinism failed: {records}")
(root / "determinism-summary.json").write_text(
    json.dumps(
        {
            "fresh_processes": 3,
            "passed": True,
            "seed": 20260729,
            "hashes": records[0],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

for SEED in 20260729 20260730 20260731; do
  python scripts/aspy_if_qualification.py \
    --device "$DEVICE" \
    --mode trajectory \
    --seed "$SEED" \
    --trajectory-steps 20 \
    >"$IF_ROOT/trajectory-$SEED.json"
done

python - "$IF_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
results = {}
for seed in (20260729, 20260730, 20260731):
    result = json.loads((root / f"trajectory-{seed}.json").read_text())["trajectory"]
    results[str(seed)] = result
    if not result["passed"]:
        raise SystemExit(f"AsPy IF SGD trajectory failed for seed {seed}: {result}")
(root / "trajectory-summary.json").write_text(
    json.dumps(
        {"fresh_processes": 3, "passed": True, "results": results},
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

python scripts/aspy_if_qualification.py \
  --device "$DEVICE" \
  --mode graph \
  --seed 20260729 \
  >"$IF_ROOT/npugraph.json"

for RUN in 1 2 3; do
  python scripts/aspy_if_qualification.py \
    --device "$DEVICE" \
    --mode performance \
    --seed 20260729 \
    --warmup "${ASPY_PERFORMANCE_WARMUP:-10}" \
    --iterations "${ASPY_PERFORMANCE_ITERATIONS:-50}" \
    >"$IF_ROOT/performance-$RUN.json"
done

python - "$IF_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
runs = [json.loads((root / f"performance-{run}.json").read_text()) for run in (1, 2, 3)]
paths = ("torch_stepwise", "aspy_eager", "aspy_npugraph")
medians = {
    path: statistics.median(run["performance"]["paths"][path]["median_ms"] for run in runs)
    for path in paths
}
speedups = {
    path: statistics.median(
        run["performance"]["speedups_vs_torch_stepwise"][path] for run in runs
    )
    for path in ("aspy_eager", "aspy_npugraph")
}
for run in runs:
    route = run["performance"]["paths"]["aspy_npugraph"]["route"]
    if route["backend"] != "npugraph" or not route["captured"]:
        raise SystemExit(f"AsPy IF performance path did not use NPUGraph: {route}")
(root / "performance-summary.json").write_text(
    json.dumps(
        {
            "fresh_processes": 3,
            "passed": True,
            "median_of_run_medians_ms": medians,
            "median_run_speedups_vs_torch": speedups,
            "runs": runs,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

for RUN in 1 2 3; do
  python scripts/aspy_plif_qualification.py \
    --device "$DEVICE" \
    --mode determinism \
    --seed 20260729 \
    >"$PLIF_ROOT/determinism-$RUN.json"
done

python - "$PLIF_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = [
    json.loads((root / f"determinism-{run}.json").read_text())["determinism"]
    for run in (1, 2, 3)
]
hashes = [record["hashes"] for record in records]
if hashes[1:] != hashes[:-1]:
    raise SystemExit(f"fresh-process AsPy PLIF determinism failed: {hashes}")
(root / "determinism-summary.json").write_text(
    json.dumps(
        {
            "fresh_processes": 3,
            "passed": True,
            "seed": 20260729,
            "hashes": hashes[0],
            "w_gradient_value": records[0]["w_gradient_value"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

for SEED in 20260729 20260730 20260731; do
  python scripts/aspy_plif_qualification.py \
    --device "$DEVICE" \
    --mode trajectory \
    --seed "$SEED" \
    --steps 20 \
    >"$PLIF_ROOT/trajectory-$SEED.json"
done

python - "$PLIF_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
results = {}
for seed in (20260729, 20260730, 20260731):
    result = json.loads((root / f"trajectory-{seed}.json").read_text())["trajectory"]
    results[str(seed)] = result
    if not result["passed"]:
        raise SystemExit(f"AsPy PLIF trajectory failed for seed {seed}: {result}")
(root / "trajectory-summary.json").write_text(
    json.dumps(
        {"fresh_processes": 3, "passed": True, "results": results},
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

python scripts/aspy_plif_qualification.py \
  --device "$DEVICE" \
  --mode graph \
  --seed 20260729 \
  >"$PLIF_ROOT/npugraph.json"

for OFFSET in 0 1 2; do
  SEED=$((20260729 + OFFSET))
  RUN=$((OFFSET + 1))
  python scripts/aspy_plif_qualification.py \
    --device "$DEVICE" \
    --mode performance \
    --seed "$SEED" \
    --warmup "${ASPY_PERFORMANCE_WARMUP:-10}" \
    --iterations "${ASPY_PERFORMANCE_ITERATIONS:-50}" \
    >"$PLIF_ROOT/performance-$RUN.json"
done

python - "$PLIF_ROOT" <<'PY'
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
runs = [json.loads((root / f"performance-{run}.json").read_text()) for run in (1, 2, 3)]
speedups = [run["performance"]["speedup"] for run in runs]
(root / "performance-summary.json").write_text(
    json.dumps(
        {
            "fresh_processes": 3,
            "passed": True,
            "median_speedup": statistics.median(speedups),
            "speedups": speedups,
            "runs": runs,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

python - "$QUALIFICATION_ROOT" <<'PY'
import hashlib
import json
import pathlib
import platform
import sys

root = pathlib.Path(sys.argv[1])
source_root = pathlib.Path.cwd()
evidence_files = sorted(path for path in root.rglob("*") if path.is_file())
source_files = sorted(
    path
    for relative in ("src", "native", "scripts", "tests")
    for path in (source_root / relative).rglob("*")
    if path.is_file() and "__pycache__" not in path.parts
)
source_files.extend(
    path
    for path in (source_root / "pyproject.toml", source_root / "uv.lock")
    if path.is_file()
)
manifest = {
    "host": platform.node(),
    "evidence_files": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_files
    },
    "source_files": {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_files)
    },
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"AsPy IF/PLIF qualification completed: {root}")
PY
