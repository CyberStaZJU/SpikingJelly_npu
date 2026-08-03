import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks import _protocol

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = "benchmarks.benchmark_sequence_recurrent"


def _smoke_args(*extra):
    entrypoint = _protocol._resolve_entrypoint(ENTRYPOINT)
    parser = _protocol._parser(ENTRYPOINT, entrypoint)
    return parser.parse_args(
        [
            "--case",
            "gru",
            "--time-steps",
            "3",
            "--batch-size",
            "2",
            "--input-size",
            "4",
            "--hidden-size",
            "4",
            "--fresh-processes",
            "1",
            "--warmup-iterations",
            "1",
            "--minimum-measured-work-seconds",
            "0",
            "--minimum-measured-iterations",
            "1",
            "--maximum-measured-iterations",
            "2",
            "--smoke",
            *extra,
        ]
    )


def _worker_result(process_index=0, order=("candidate", "baseline")):
    measurements = {}
    for offset, implementation in enumerate(_protocol.IMPLEMENTATIONS):
        measurements[implementation] = {
            "implementation": implementation,
            "cold_latency_ms": 10.0 + offset,
            "warmup_iterations": 1,
            "measured_iterations": 3,
            "measured_work_seconds": 0.1,
            "minimum_measured_work_seconds": 0.0,
            "median_ms": float(process_index * 10 + offset + 1),
            "mean_ms": float(process_index * 10 + offset + 2),
            "p90_ms": float(process_index * 10 + offset + 3),
            "peak_allocated_device_memory_bytes": None,
            "peak_reserved_device_memory_bytes": None,
            "metadata": {
                "provider_routes": [],
                "native_region_count": 0,
                "format_conversion_bytes": None,
            },
        }
    return {
        "schema_version": _protocol.SCHEMA_VERSION,
        "kind": "fresh_process_result",
        "entrypoint": ENTRYPOINT,
        "case": "gru",
        "process_index": process_index,
        "pid": 1000 + process_index,
        "order": list(order),
        "seed": 7,
        "mode": "eval",
        "dtype": "float32",
        "smoke": False,
        "evidence_eligible": True,
        "input_hash_sha256": "a" * 64,
        "state_hash_sha256": "b" * 64,
        "workload": {"shape": [3, 2, 4]},
        "runtime": {"device": "cpu"},
        "source_and_build_identity": {
            "git_commit": "deadbeef",
            "git_tree": "feedface",
            "git_dirty": False,
        },
        "synchronization": "synchronous CPU execution",
        "measurements": measurements,
    }


def test_worker_schema_and_cpu_optional_memory_fields():
    result = _protocol.run_worker(
        ENTRYPOINT,
        _smoke_args(),
        process_index=0,
        order=("candidate", "baseline"),
    )

    _protocol.validate_worker_result(
        result,
        expected_entrypoint=ENTRYPOINT,
        expected_process_index=0,
        expected_order=("candidate", "baseline"),
    )
    assert result["schema_version"] == 1
    assert result["kind"] == "fresh_process_result"
    assert result["smoke"] is True
    assert result["evidence_eligible"] is False
    assert set(result["measurements"]) == {"candidate", "baseline"}
    for measurement in result["measurements"].values():
        assert measurement["cold_latency_ms"] >= 0.0
        assert measurement["warmup_iterations"] == 1
        assert measurement["measured_iterations"] >= 1
        assert measurement["median_ms"] >= 0.0
        assert measurement["mean_ms"] >= 0.0
        assert measurement["p90_ms"] >= 0.0
        assert measurement["peak_allocated_device_memory_bytes"] is None
        assert measurement["peak_reserved_device_memory_bytes"] is None
        assert "provider_routes" in measurement["metadata"]


def test_stable_hash_is_deterministic_and_sensitive_to_input_and_state():
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = {"input": first, "state": (torch.ones(2), {"seed": 9})}

    assert _protocol.stable_hash(payload) == _protocol.stable_hash(
        {"state": (torch.ones(2), {"seed": 9}), "input": first.clone()}
    )
    changed_input = {"input": first.clone(), "state": (torch.ones(2), {"seed": 9})}
    changed_input["input"][0, 0] = 1.0
    assert _protocol.stable_hash(payload) != _protocol.stable_hash(changed_input)
    assert _protocol.stable_hash(payload) != _protocol.stable_hash(
        {"input": first, "state": (torch.zeros(2), {"seed": 9})}
    )
    transposed = first.transpose(0, 1)
    assert torch.equal(transposed, transposed.contiguous())
    assert _protocol.stable_hash(transposed) != _protocol.stable_hash(
        transposed.contiguous()
    )


def test_recurrent_reference_defaults_preserve_explicit_time_steps():
    entrypoint = _protocol._resolve_entrypoint(ENTRYPOINT)
    parser = _protocol._parser(ENTRYPOINT, entrypoint)
    common = [
        "--case",
        "spiking-rnn",
        "--batch-size",
        "1",
        "--input-size",
        "2",
        "--hidden-size",
        "2",
        "--smoke",
    ]

    default_case = entrypoint.build_case(parser.parse_args(common), torch.device("cpu"))
    explicit_case = entrypoint.build_case(
        parser.parse_args([*common, "--time-steps", "128"]),
        torch.device("cpu"),
    )

    assert default_case.workload["shape"]["T"] == 64
    assert explicit_case.workload["shape"]["T"] == 128


def test_balanced_interleaved_order_alternates_first_position():
    orders = _protocol.balanced_interleaved_orders(5)

    assert orders == [
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ]
    first_positions = [order[0] for order in orders]
    assert abs(first_positions.count("candidate") - first_positions.count("baseline")) == 1


def test_percentile_and_median_of_five_aggregation():
    assert _protocol.percentile([0.0, 10.0], 0.9) == pytest.approx(9.0)
    results = [
        _worker_result(process_index, order)
        for process_index, order in enumerate(_protocol.balanced_interleaved_orders(5))
    ]

    aggregate = _protocol.aggregate_worker_results(ENTRYPOINT, results, smoke=False)

    assert aggregate["fresh_processes"] == 5
    assert aggregate["evidence_eligible"] is True
    assert aggregate["aggregate"]["candidate"]["raw_process_medians_ms"] == [
        1.0,
        11.0,
        21.0,
        31.0,
        41.0,
    ]
    assert (
        aggregate["aggregate"]["candidate"]["median_of_process_medians_ms"]
        == 21.0
    )
    assert aggregate["aggregate"]["baseline"]["median_of_process_p90_ms"] == 24.0


def test_subprocess_orchestrator_uses_fresh_pid_and_balanced_order():
    args = _smoke_args("--fresh-processes", "2")

    aggregate = _protocol.run_orchestrator(ENTRYPOINT, args)

    raw = aggregate["raw_process_results"]
    assert aggregate["fresh_processes"] == 2
    assert aggregate["orders"] == [
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]
    assert len({result["pid"] for result in raw}) == 2
    assert all(result["pid"] != os.getpid() for result in raw)
    assert len({result["input_hash_sha256"] for result in raw}) == 1
    assert len({result["state_hash_sha256"] for result in raw}) == 1


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda result: result.update(schema_version=999),
            "unsupported schema_version",
        ),
        (
            lambda result: result.update(order=["candidate", "candidate"]),
            "order must contain candidate and baseline",
        ),
        (
            lambda result: result["measurements"]["candidate"].update(
                median_ms=float("nan")
            ),
            "median_ms must be finite",
        ),
    ],
)
def test_worker_validation_errors(mutate, match):
    result = _worker_result()
    mutate(result)

    with pytest.raises(_protocol.BenchmarkProtocolError, match=match):
        _protocol.validate_worker_result(result)


def test_aggregate_rejects_hash_and_order_mismatches():
    results = [
        _worker_result(process_index, order)
        for process_index, order in enumerate(_protocol.balanced_interleaved_orders(2))
    ]
    bad_hash = copy.deepcopy(results)
    bad_hash[1]["state_hash_sha256"] = "c" * 64
    with pytest.raises(_protocol.BenchmarkProtocolError, match="identical initial state"):
        _protocol.aggregate_worker_results(ENTRYPOINT, bad_hash, smoke=True)

    bad_order = copy.deepcopy(results)
    bad_order[1]["order"] = ["candidate", "baseline"]
    with pytest.raises(_protocol.BenchmarkProtocolError, match="balanced/interleaved"):
        _protocol.aggregate_worker_results(ENTRYPOINT, bad_order, smoke=True)

    duplicate_pid = copy.deepcopy(results)
    duplicate_pid[1]["pid"] = duplicate_pid[0]["pid"]
    with pytest.raises(_protocol.BenchmarkProtocolError, match="distinct fresh processes"):
        _protocol.aggregate_worker_results(ENTRYPOINT, duplicate_pid, smoke=True)

    duplicate_index = copy.deepcopy(results)
    duplicate_index[1]["process_index"] = duplicate_index[0]["process_index"]
    with pytest.raises(_protocol.BenchmarkProtocolError, match="process_index"):
        _protocol.aggregate_worker_results(ENTRYPOINT, duplicate_index, smoke=True)


def test_formal_settings_and_repository_output_are_rejected(tmp_path):
    formal = _smoke_args()
    formal.smoke = False
    formal.fresh_processes = 1
    with pytest.raises(_protocol.BenchmarkProtocolError, match="exactly five"):
        _protocol._validate_common_args(formal)

    output_inside_repository = _smoke_args("--output", str(ROOT / "result.json"))
    with pytest.raises(_protocol.BenchmarkProtocolError, match="outside the source repository"):
        _protocol._validate_common_args(output_inside_repository)

    output_outside_repository = _smoke_args("--output", str(tmp_path / "result.json"))
    _protocol._validate_common_args(output_outside_repository)


def test_protocol_and_entrypoints_do_not_import_torch_npu():
    code = (
        "import sys; "
        "from benchmarks import _protocol; "
        "import benchmarks.benchmark_sequence_recurrent; "
        "import benchmarks.benchmark_sequence_transformer; "
        "import benchmarks.benchmark_spikformer; "
        "assert 'torch_npu' not in sys.modules; "
        "print(json.dumps({'ok': True}))"
    )
    environment = os.environ.copy()
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", "import json; " + code],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=environment,
    )

    assert json.loads(completed.stdout) == {"ok": True}


def test_worker_rejects_shared_payload_mutation_between_implementations(monkeypatch):
    args = _smoke_args()
    payload = torch.zeros(1)

    def build_case(_args, device):
        def mutate():
            payload.add_(1)

        return _protocol.BenchmarkCase(
            name="mutating-payload",
            device=device,
            candidate_step=mutate,
            baseline_step=lambda: None,
            input_hash_payload={"input": payload},
            state_hash_payload={"state": torch.zeros(1)},
            workload={"case": "mutation-guard"},
        )

    monkeypatch.setattr(
        _protocol,
        "_resolve_entrypoint",
        lambda _module: _protocol.Entrypoint(build_case=build_case),
    )

    with pytest.raises(
        _protocol.BenchmarkProtocolError,
        match="input payload mutated",
    ):
        _protocol.run_worker(
            "benchmarks.mutating_test_entrypoint",
            args,
            process_index=0,
            order=("candidate", "baseline"),
        )
