"""Reusable fresh-process benchmark protocol for sequence workloads.

The default settings mirror ``docs/evidence/sequence_acceptance_policy.json``:
five fresh processes per implementation, balanced interleaving, a cold call,
warmup, at least five seconds of measured work per implementation/process, and
device synchronization around every measured region. Tests and developer smoke
runs may explicitly opt into shorter durations; such output is marked as smoke
and is not acceptance or performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

if __name__ == "__main__":
    sys.modules.setdefault("benchmarks._protocol", sys.modules[__name__])

SCHEMA_VERSION = 1
DEFAULT_FRESH_PROCESSES = 5
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_MINIMUM_MEASURED_WORK_SECONDS = 5.0
DEFAULT_MINIMUM_MEASURED_ITERATIONS = 1
DEFAULT_MAXIMUM_MEASURED_ITERATIONS = 1_000_000
IMPLEMENTATIONS = ("candidate", "baseline")

JsonValue = Any
Step = Callable[[], object]
MetadataHook = Callable[[], Mapping[str, object] | None]


class BenchmarkProtocolError(ValueError):
    """Raised when a benchmark request or worker result violates the protocol."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkProtocolError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkProtocolError(f"{name} must be a non-negative integer")
    return value


def _non_negative_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkProtocolError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BenchmarkProtocolError(f"{name} must be a non-negative finite number")
    return result


def _nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BenchmarkProtocolError(f"{name} must be a non-empty, trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class MeasurementSettings:
    """Validated timing settings for one implementation in one worker."""

    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    minimum_measured_work_seconds: float = DEFAULT_MINIMUM_MEASURED_WORK_SECONDS
    minimum_measured_iterations: int = DEFAULT_MINIMUM_MEASURED_ITERATIONS
    maximum_measured_iterations: int = DEFAULT_MAXIMUM_MEASURED_ITERATIONS

    def __post_init__(self) -> None:
        _non_negative_int("warmup_iterations", self.warmup_iterations)
        _non_negative_float(
            "minimum_measured_work_seconds", self.minimum_measured_work_seconds
        )
        _positive_int("minimum_measured_iterations", self.minimum_measured_iterations)
        _positive_int("maximum_measured_iterations", self.maximum_measured_iterations)
        if self.maximum_measured_iterations < self.minimum_measured_iterations:
            raise BenchmarkProtocolError(
                "maximum_measured_iterations must be greater than or equal to "
                "minimum_measured_iterations"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One candidate/baseline comparison prepared inside a fresh worker."""

    name: str
    device: torch.device
    candidate_step: Step
    baseline_step: Step
    input_hash_payload: object
    state_hash_payload: object
    workload: Mapping[str, object]
    candidate_metadata: MetadataHook | None = None
    baseline_metadata: MetadataHook | None = None
    synchronize: Callable[[], None] | None = None
    reset_peak_memory: Callable[[], None] | None = None
    peak_memory: Callable[[], tuple[int | None, int | None]] | None = None

    def __post_init__(self) -> None:
        _nonempty_string("case name", self.name)
        if not isinstance(self.device, torch.device):
            raise BenchmarkProtocolError("device must be a torch.device")
        if not callable(self.candidate_step) or not callable(self.baseline_step):
            raise BenchmarkProtocolError("candidate_step and baseline_step must be callable")
        if not isinstance(self.workload, Mapping):
            raise BenchmarkProtocolError("workload must be a mapping")
        for name, hook in (
            ("candidate_metadata", self.candidate_metadata),
            ("baseline_metadata", self.baseline_metadata),
            ("synchronize", self.synchronize),
            ("reset_peak_memory", self.reset_peak_memory),
            ("peak_memory", self.peak_memory),
        ):
            if hook is not None and not callable(hook):
                raise BenchmarkProtocolError(f"{name} must be callable or None")


@dataclass(frozen=True, slots=True)
class Entrypoint:
    """Importable benchmark entrypoint registered by a CLI module."""

    build_case: Callable[[argparse.Namespace, torch.device], BenchmarkCase]
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None

    def __post_init__(self) -> None:
        if not callable(self.build_case):
            raise BenchmarkProtocolError("entrypoint build_case must be callable")
        if self.add_arguments is not None and not callable(self.add_arguments):
            raise BenchmarkProtocolError("entrypoint add_arguments must be callable or None")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkProtocolError(f"value is not canonical JSON: {error}") from error


def _hash_update(hasher: Any, value: object, active: set[int]) -> None:
    if isinstance(value, torch.Tensor):
        source = value.detach().cpu()
        metadata = {
            "dtype": str(source.dtype),
            "shape": list(source.shape),
            "stride": list(source.stride()),
        }
        tensor = source.contiguous()
        hasher.update(b"tensor\0")
        hasher.update(_canonical_json(metadata).encode())
        hasher.update(b"\0")
        hasher.update(bytes(tensor.untyped_storage()))
        return
    if value is None or isinstance(value, bool | int | float | str):
        hasher.update(b"scalar\0")
        hasher.update(_canonical_json(value).encode())
        return
    if isinstance(value, torch.device | torch.dtype):
        hasher.update(b"torch-value\0")
        hasher.update(str(value).encode())
        return
    if is_dataclass(value) and not isinstance(value, type):
        _hash_update(hasher, asdict(value), active)
        return

    identity = id(value)
    if identity in active:
        raise BenchmarkProtocolError("hash payload must not contain recursive containers")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            hasher.update(b"mapping\0")
            normalized = []
            for key, item in value.items():
                key_json = _canonical_json(key)
                normalized.append((key_json, item))
            for key_json, item in sorted(normalized, key=lambda pair: pair[0]):
                hasher.update(key_json.encode())
                hasher.update(b"\0")
                _hash_update(hasher, item, active)
            return
        if isinstance(value, tuple):
            hasher.update(b"tuple\0")
            for item in value:
                _hash_update(hasher, item, active)
            return
        if isinstance(value, list):
            hasher.update(b"list\0")
            for item in value:
                _hash_update(hasher, item, active)
            return
    finally:
        active.remove(identity)
    raise BenchmarkProtocolError(
        f"unsupported hash payload type {type(value).__name__}; use tensors and JSON-like data"
    )


def stable_hash(value: object) -> str:
    """Return a deterministic SHA-256 for tensors and nested JSON-like structures."""

    hasher = hashlib.sha256()
    _hash_update(hasher, value, set())
    return hasher.hexdigest()


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample sequence."""

    if not samples:
        raise BenchmarkProtocolError("percentile requires at least one sample")
    if isinstance(quantile, bool) or not isinstance(quantile, int | float):
        raise BenchmarkProtocolError("quantile must be a finite number in [0, 1]")
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise BenchmarkProtocolError("quantile must be a finite number in [0, 1]")
    ordered = sorted(float(sample) for sample in samples)
    if not all(math.isfinite(sample) and sample >= 0.0 for sample in ordered):
        raise BenchmarkProtocolError("percentile samples must be finite and non-negative")
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def balanced_interleaved_orders(processes: int) -> list[tuple[str, str]]:
    """Alternate first position and keep its counts balanced within one process."""

    _positive_int("processes", processes)
    return [
        ("candidate", "baseline") if process_index % 2 == 0 else ("baseline", "candidate")
        for process_index in range(processes)
    ]


def _resolve_entrypoint(module_name: str) -> Entrypoint:
    module_name = _nonempty_string("entrypoint module", module_name)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, AttributeError) as error:
        raise BenchmarkProtocolError(
            f"cannot import benchmark entrypoint module {module_name!r}: {error}"
        ) from error
    entrypoint = getattr(module, "ENTRYPOINT", None)
    if not isinstance(entrypoint, Entrypoint):
        raise BenchmarkProtocolError(
            f"{module_name!r} must expose ENTRYPOINT as benchmarks._protocol.Entrypoint"
        )
    return entrypoint


def _add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--fresh-processes",
        type=int,
        default=DEFAULT_FRESH_PROCESSES,
        help="five is required by the frozen acceptance policy",
    )
    parser.add_argument("--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument(
        "--minimum-measured-work-seconds",
        type=float,
        default=DEFAULT_MINIMUM_MEASURED_WORK_SECONDS,
    )
    parser.add_argument(
        "--minimum-measured-iterations",
        type=int,
        default=DEFAULT_MINIMUM_MEASURED_ITERATIONS,
    )
    parser.add_argument(
        "--maximum-measured-iterations",
        type=int,
        default=DEFAULT_MAXIMUM_MEASURED_ITERATIONS,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="allow shorter/non-five-process CPU checks; output is non-evidence",
    )


def _parser(module_name: str, entrypoint: Entrypoint) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fresh-process benchmark protocol for {module_name}"
    )
    _add_protocol_arguments(parser)
    if entrypoint.add_arguments is not None:
        entrypoint.add_arguments(parser)
    return parser


def _settings_from_args(args: argparse.Namespace) -> MeasurementSettings:
    return MeasurementSettings(
        warmup_iterations=args.warmup_iterations,
        minimum_measured_work_seconds=args.minimum_measured_work_seconds,
        minimum_measured_iterations=args.minimum_measured_iterations,
        maximum_measured_iterations=args.maximum_measured_iterations,
    )


def _validate_common_args(args: argparse.Namespace) -> None:
    _positive_int("threads", args.threads)
    _positive_int("fresh_processes", args.fresh_processes)
    _settings_from_args(args)
    if not args.smoke:
        if args.fresh_processes != DEFAULT_FRESH_PROCESSES:
            raise BenchmarkProtocolError(
                "formal protocol requires exactly five fresh processes; pass --smoke only "
                "for non-evidence development checks"
            )
        if args.warmup_iterations <= 0:
            raise BenchmarkProtocolError("formal protocol requires at least one warmup iteration")
        if args.minimum_measured_work_seconds < DEFAULT_MINIMUM_MEASURED_WORK_SECONDS:
            raise BenchmarkProtocolError(
                "formal protocol requires at least five measured-work seconds per "
                "implementation/process; pass --smoke only for shorter checks"
            )
    if args.output is not None:
        output = args.output.expanduser().resolve()
        repository = Path(__file__).resolve().parents[1]
        try:
            output.relative_to(repository)
        except ValueError:
            pass
        else:
            raise BenchmarkProtocolError(
                "generated benchmark JSON must be written outside the source repository"
            )


def resolve_device(device_spec: str) -> torch.device:
    """Resolve CPU or lazily configure an explicitly requested Ascend device."""

    device_spec = _nonempty_string("device", device_spec)
    if device_spec == "auto":
        return torch.device("cpu")
    if device_spec.startswith("npu"):
        from spikingjelly_npu.npu import configure_npu

        return configure_npu(device_spec)
    device = torch.device(device_spec)
    if device.type not in {"cpu", "npu"}:
        raise BenchmarkProtocolError(
            f"unsupported benchmark device type {device.type!r}; expected CPU or NPU"
        )
    return device


def synchronize_device(device: torch.device) -> None:
    """Synchronize the selected backend without importing torch-npu on CPU."""

    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if callable(synchronize) and device.type != "cpu":
        synchronize(device)


def reset_peak_memory_stats(device: torch.device) -> None:
    """Reset peak device-memory counters when the backend exposes them."""

    backend = getattr(torch, device.type, None)
    reset = getattr(backend, "reset_peak_memory_stats", None)
    if callable(reset):
        reset(device)


def peak_memory_stats(device: torch.device) -> tuple[int | None, int | None]:
    """Return peak allocated/reserved bytes, or ``None`` when unsupported."""

    backend = getattr(torch, device.type, None)
    allocated = getattr(backend, "max_memory_allocated", None)
    reserved = getattr(backend, "max_memory_reserved", None)
    peak_allocated = int(allocated(device)) if callable(allocated) else None
    peak_reserved = int(reserved(device)) if callable(reserved) else None
    return peak_allocated, peak_reserved


def runtime_metadata(device: torch.device) -> dict[str, JsonValue]:
    """Collect runtime identity while keeping CPU execution import-safe."""

    import spikingjelly_npu

    torch_npu_version = None
    device_name = platform.processor() or platform.machine()
    if device.type == "npu":
        torch_npu = importlib.import_module("torch_npu")
        torch_npu_version = getattr(torch_npu, "__version__", None)
        getter = getattr(torch.npu, "get_device_name", None)
        if callable(getter):
            device_name = str(getter(device))
    return {
        "device": str(device),
        "device_name": device_name,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu_version,
        "spikingjelly_npu": spikingjelly_npu.__version__,
        "cann": os.environ.get("ASCEND_TOOLKIT_VERSION"),
    }


def source_and_build_identity() -> dict[str, JsonValue]:
    """Collect source/build identity from Git and optional external build metadata."""

    root = Path(__file__).resolve().parents[1]
    identity: dict[str, JsonValue] = {
        "repository": str(root),
        "git_commit": None,
        "git_tree": None,
        "git_dirty": None,
        "aspy_build_manifest": os.environ.get("SPIKINGJELLY_NPU_ASPY_BUILD_MANIFEST"),
        "aspy_extension": os.environ.get("SPIKINGJELLY_NPU_ASPY_EXTENSION"),
    }
    try:
        identity["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        identity["git_tree"] = subprocess.run(
            ["git", "write-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        identity["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return identity


def _json_compatible(value: object, field: str) -> JsonValue:
    try:
        return json.loads(_canonical_json(value))
    except BenchmarkProtocolError as error:
        raise BenchmarkProtocolError(f"{field} must contain JSON-compatible values") from error


def _route_to_dict(route: object) -> dict[str, JsonValue]:
    if hasattr(route, "to_dict") and callable(route.to_dict):
        value = route.to_dict()
    elif is_dataclass(route) and not isinstance(route, type):
        value = asdict(route)
    elif isinstance(route, Mapping):
        value = dict(route)
    else:
        value = {
            name: getattr(route, name)
            for name in (
                "requested_provider",
                "actual_provider",
                "logical_operation",
                "reason_code",
                "reason",
                "accelerated",
                "strict",
                "mode",
                "native_launch_attempted",
                "abi_version",
                "schema_version",
                "bucket",
                "native_region",
                "format_conversion",
            )
            if hasattr(route, name)
        }
        if not value:
            raise BenchmarkProtocolError(
                f"unsupported provider route value {type(route).__name__}"
            )
    return _json_compatible(value, "provider route")


def collect_module_route_metadata(module: torch.nn.Module) -> dict[str, JsonValue]:
    """Collect leaf route records and summarize native regions/conversions."""

    routes = []
    for name, child in module.named_modules():
        route = getattr(child, "last_backend_route", None)
        if route is None:
            continue
        routes.append({"module": name or "<root>", **_route_to_dict(route)})
    native_regions = [route.get("native_region") for route in routes]
    native_region_counts = Counter(
        str(region) for region in native_regions if isinstance(region, str) and region
    )
    format_conversions = Counter(
        str(route["format_conversion"])
        for route in routes
        if isinstance(route.get("format_conversion"), str) and route["format_conversion"]
    )
    native_launch_count = sum(bool(route.get("native_launch_attempted")) for route in routes)
    return {
        "provider_routes": routes,
        "native_region_count": native_launch_count,
        "native_region_counts": dict(sorted(native_region_counts.items())),
        "format_conversion_count": sum(format_conversions.values()),
        "format_conversion_counts": dict(sorted(format_conversions.items())),
        "format_conversion_bytes": None,
    }


def _metadata(hook: MetadataHook | None) -> dict[str, JsonValue]:
    defaults = {
        "provider_routes": [],
        "native_region_count": 0,
        "native_region_counts": {},
        "format_conversion_count": 0,
        "format_conversion_counts": {},
        "format_conversion_bytes": None,
    }
    if hook is None:
        return defaults
    value = hook()
    if value is None:
        return defaults
    if not isinstance(value, Mapping):
        raise BenchmarkProtocolError("metadata hook must return a mapping or None")
    result = {**defaults, **dict(value)}
    return _json_compatible(result, "metadata hook result")


def _synchronization_description(device: torch.device) -> str:
    if device.type == "cpu":
        return "synchronous CPU execution; protocol boundaries are still recorded"
    return "device synchronization immediately before and after every timed region"


def _measure_implementation(
    implementation: str,
    step: Step,
    settings: MeasurementSettings,
    *,
    synchronize: Callable[[], None],
    reset_peak_memory: Callable[[], None],
    peak_memory: Callable[[], tuple[int | None, int | None]],
    metadata_hook: MetadataHook | None,
) -> dict[str, JsonValue]:
    if implementation not in IMPLEMENTATIONS:
        raise BenchmarkProtocolError(f"unsupported implementation {implementation!r}")

    synchronize()
    cold_start = time.perf_counter()
    step()
    synchronize()
    cold_latency_ms = (time.perf_counter() - cold_start) * 1000.0

    for _ in range(settings.warmup_iterations):
        step()
    synchronize()

    synchronize()
    reset_peak_memory()
    synchronize()
    samples_ms = []
    measured_work_seconds = 0.0
    while (
        len(samples_ms) < settings.minimum_measured_iterations
        or measured_work_seconds < settings.minimum_measured_work_seconds
    ):
        if len(samples_ms) >= settings.maximum_measured_iterations:
            raise BenchmarkProtocolError(
                "maximum_measured_iterations reached before minimum measured work; "
                "increase the cap or lower the smoke duration"
            )
        synchronize()
        start = time.perf_counter()
        step()
        synchronize()
        elapsed = time.perf_counter() - start
        measured_work_seconds += elapsed
        samples_ms.append(elapsed * 1000.0)

    peak_allocated, peak_reserved = peak_memory()
    return {
        "implementation": implementation,
        "cold_latency_ms": cold_latency_ms,
        "warmup_iterations": settings.warmup_iterations,
        "measured_iterations": len(samples_ms),
        "measured_work_seconds": measured_work_seconds,
        "minimum_measured_work_seconds": settings.minimum_measured_work_seconds,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": percentile(samples_ms, 0.9),
        "peak_allocated_device_memory_bytes": peak_allocated,
        "peak_reserved_device_memory_bytes": peak_reserved,
        "metadata": _metadata(metadata_hook),
    }


def run_worker(
    entrypoint_module: str,
    args: argparse.Namespace,
    *,
    process_index: int,
    order: Sequence[str],
) -> dict[str, JsonValue]:
    """Run one fresh-process candidate/baseline comparison and return one JSON record."""

    if sorted(order) != sorted(IMPLEMENTATIONS) or len(order) != len(IMPLEMENTATIONS):
        raise BenchmarkProtocolError(
            "worker order must contain candidate and baseline exactly once"
        )
    _non_negative_int("process_index", process_index)
    _validate_common_args(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    entrypoint = _resolve_entrypoint(entrypoint_module)
    case = entrypoint.build_case(args, device)
    if case.device != device:
        raise BenchmarkProtocolError(
            f"case device {case.device} does not match requested device {device}"
        )

    synchronize = case.synchronize or (lambda: synchronize_device(device))
    reset_memory = case.reset_peak_memory or (lambda: reset_peak_memory_stats(device))
    peak_memory = case.peak_memory or (lambda: peak_memory_stats(device))
    settings = _settings_from_args(args)
    steps = {
        "candidate": case.candidate_step,
        "baseline": case.baseline_step,
    }
    metadata_hooks = {
        "candidate": case.candidate_metadata,
        "baseline": case.baseline_metadata,
    }
    input_hash = stable_hash(case.input_hash_payload)
    state_hash = stable_hash(case.state_hash_payload)
    measurements = {}
    for implementation in order:
        if stable_hash(case.input_hash_payload) != input_hash:
            raise BenchmarkProtocolError(
                "benchmark input payload mutated before the next implementation"
            )
        if stable_hash(case.state_hash_payload) != state_hash:
            raise BenchmarkProtocolError(
                "benchmark state payload mutated before the next implementation"
            )
        measurements[implementation] = _measure_implementation(
            implementation,
            steps[implementation],
            settings,
            synchronize=synchronize,
            reset_peak_memory=reset_memory,
            peak_memory=peak_memory,
            metadata_hook=metadata_hooks[implementation],
        )
    if stable_hash(case.input_hash_payload) != input_hash:
        raise BenchmarkProtocolError("benchmark input payload mutated during measurement")
    if stable_hash(case.state_hash_payload) != state_hash:
        raise BenchmarkProtocolError("benchmark state payload mutated during measurement")
    source_identity = source_and_build_identity()

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_process_result",
        "entrypoint": entrypoint_module,
        "case": case.name,
        "process_index": process_index,
        "pid": os.getpid(),
        "order": list(order),
        "seed": args.seed,
        "mode": args.mode,
        "dtype": args.dtype,
        "smoke": bool(args.smoke),
        "evidence_eligible": not args.smoke and not bool(source_identity.get("git_dirty")),
        "input_hash_sha256": input_hash,
        "state_hash_sha256": state_hash,
        "workload": _json_compatible(dict(case.workload), "workload"),
        "runtime": runtime_metadata(device),
        "source_and_build_identity": source_identity,
        "synchronization": _synchronization_description(device),
        "measurements": measurements,
    }


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkProtocolError(f"{field} must be a mapping")
    return value


def validate_worker_result(
    result: Mapping[str, object],
    *,
    expected_entrypoint: str | None = None,
    expected_process_index: int | None = None,
    expected_order: Sequence[str] | None = None,
) -> None:
    """Validate one worker JSON before it enters aggregate evidence."""

    if result.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkProtocolError("worker result has an unsupported schema_version")
    if result.get("kind") != "fresh_process_result":
        raise BenchmarkProtocolError("worker result kind must be 'fresh_process_result'")
    if expected_entrypoint is not None and result.get("entrypoint") != expected_entrypoint:
        raise BenchmarkProtocolError("worker result entrypoint does not match the request")
    if expected_process_index is not None and result.get("process_index") != expected_process_index:
        raise BenchmarkProtocolError("worker result process_index does not match the request")
    process_index = result.get("process_index")
    if (
        isinstance(process_index, bool)
        or not isinstance(process_index, int)
        or process_index < 0
    ):
        raise BenchmarkProtocolError(
            "worker result process_index must be a non-negative integer"
        )
    pid = result.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise BenchmarkProtocolError("worker result pid must be a positive integer")
    order = result.get("order")
    if not isinstance(order, list) or sorted(order) != sorted(IMPLEMENTATIONS):
        raise BenchmarkProtocolError("worker result order must contain candidate and baseline")
    if expected_order is not None and order != list(expected_order):
        raise BenchmarkProtocolError("worker result order does not match the scheduled order")
    for field in ("input_hash_sha256", "state_hash_sha256"):
        value = result.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise BenchmarkProtocolError(f"worker result {field} must be a SHA-256 string")
    smoke = result.get("smoke")
    evidence_eligible = result.get("evidence_eligible")
    if type(smoke) is not bool or type(evidence_eligible) is not bool:
        raise BenchmarkProtocolError("worker smoke and evidence_eligible fields must be bools")
    if smoke and evidence_eligible:
        raise BenchmarkProtocolError("a smoke worker result cannot be evidence eligible")
    _required_mapping(result.get("runtime"), "runtime")
    source_identity = _required_mapping(
        result.get("source_and_build_identity"), "source_and_build_identity"
    )
    if source_identity.get("git_commit") is not None and not isinstance(
        source_identity.get("git_commit"), str
    ):
        raise BenchmarkProtocolError(
            "source_and_build_identity.git_commit must be a string or None"
        )
    if source_identity.get("git_tree") is not None and not isinstance(
        source_identity.get("git_tree"), str
    ):
        raise BenchmarkProtocolError("source_and_build_identity.git_tree must be a string or None")
    if source_identity.get("git_dirty") is not None and type(
        source_identity.get("git_dirty")
    ) is not bool:
        raise BenchmarkProtocolError("source_and_build_identity.git_dirty must be a bool or None")
    if source_identity.get("git_dirty") and evidence_eligible:
        raise BenchmarkProtocolError("a dirty-source worker result cannot be evidence eligible")
    _required_mapping(result.get("workload"), "workload")
    measurements = _required_mapping(result.get("measurements"), "measurements")
    if set(measurements) != set(IMPLEMENTATIONS):
        raise BenchmarkProtocolError("worker measurements must contain candidate and baseline")
    for implementation in IMPLEMENTATIONS:
        measurement = _required_mapping(
            measurements[implementation], f"measurements.{implementation}"
        )
        for metric in (
            "cold_latency_ms",
            "measured_work_seconds",
            "median_ms",
            "mean_ms",
            "p90_ms",
        ):
            value = measurement.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise BenchmarkProtocolError(
                    f"measurements.{implementation}.{metric} must be finite and non-negative"
                )
        measured_iterations = measurement.get("measured_iterations")
        if (
            isinstance(measured_iterations, bool)
            or not isinstance(measured_iterations, int)
            or measured_iterations <= 0
        ):
            raise BenchmarkProtocolError(
                f"measurements.{implementation}.measured_iterations must be positive"
            )
        metadata = _required_mapping(
            measurement.get("metadata"), f"measurements.{implementation}.metadata"
        )
        routes = metadata.get("provider_routes")
        if not isinstance(routes, list):
            raise BenchmarkProtocolError(
                f"measurements.{implementation}.metadata.provider_routes must be a list"
            )
        native_region_count = metadata.get("native_region_count")
        if (
            isinstance(native_region_count, bool)
            or not isinstance(native_region_count, int)
            or native_region_count < 0
        ):
            raise BenchmarkProtocolError(
                f"measurements.{implementation}.metadata.native_region_count "
                "must be a non-negative integer"
            )
        conversion_bytes = metadata.get("format_conversion_bytes")
        if conversion_bytes is not None and (
            isinstance(conversion_bytes, bool)
            or not isinstance(conversion_bytes, int)
            or conversion_bytes < 0
        ):
            raise BenchmarkProtocolError(
                f"measurements.{implementation}.metadata.format_conversion_bytes "
                "must be a non-negative integer or None"
            )


def _load_single_json(stdout: str, process_index: int) -> dict[str, object]:
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkProtocolError(
            f"worker {process_index} did not emit one valid JSON document: {error}"
        ) from error
    if not isinstance(result, dict):
        raise BenchmarkProtocolError(f"worker {process_index} JSON root must be an object")
    return result


def _worker_command(
    entrypoint_module: str,
    args: argparse.Namespace,
    process_index: int,
    order: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks._protocol",
        "--worker",
        "--entrypoint",
        entrypoint_module,
        "--process-index",
        str(process_index),
        "--order",
        *order,
        "--device",
        args.device,
        "--mode",
        args.mode,
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--threads",
        str(args.threads),
        "--fresh-processes",
        str(args.fresh_processes),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--minimum-measured-work-seconds",
        str(args.minimum_measured_work_seconds),
        "--minimum-measured-iterations",
        str(args.minimum_measured_iterations),
        "--maximum-measured-iterations",
        str(args.maximum_measured_iterations),
    ]
    if args.smoke:
        command.append("--smoke")
    entrypoint = _resolve_entrypoint(entrypoint_module)
    parser = _parser(entrypoint_module, entrypoint)
    protocol_names = {
        "device",
        "mode",
        "seed",
        "dtype",
        "threads",
        "fresh_processes",
        "warmup_iterations",
        "minimum_measured_work_seconds",
        "minimum_measured_iterations",
        "maximum_measured_iterations",
        "output",
        "smoke",
    }
    for action in parser._actions:
        if (
            not action.option_strings
            or action.dest in protocol_names
            or action.dest == argparse.SUPPRESS
            or not hasattr(args, action.dest)
        ):
            continue
        value = getattr(args, action.dest)
        default = action.default
        if value == default:
            continue
        option = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                command.append(option)
        else:
            command.extend((option, str(value)))
    return command


def aggregate_worker_results(
    entrypoint_module: str,
    results: Sequence[Mapping[str, object]],
    *,
    smoke: bool,
) -> dict[str, JsonValue]:
    """Validate raw workers and compute median-of-process-medians aggregates."""

    if not results:
        raise BenchmarkProtocolError("aggregation requires at least one worker result")
    for result in results:
        validate_worker_result(result, expected_entrypoint=entrypoint_module)
    if not smoke and len(results) != DEFAULT_FRESH_PROCESSES:
        raise BenchmarkProtocolError("formal aggregation requires exactly five worker results")

    process_indices = [result.get("process_index") for result in results]
    if process_indices != list(range(len(results))):
        raise BenchmarkProtocolError(
            "worker results must have unique contiguous process_index values in order"
        )
    pids = [result.get("pid") for result in results]
    if any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids):
        raise BenchmarkProtocolError("worker result pid values must be positive integers")
    if len(set(pids)) != len(pids):
        raise BenchmarkProtocolError("worker results must come from distinct fresh processes")

    input_hashes = {str(result["input_hash_sha256"]) for result in results}
    state_hashes = {str(result["state_hash_sha256"]) for result in results}
    workloads = {_canonical_json(result["workload"]) for result in results}
    if len(input_hashes) != 1:
        raise BenchmarkProtocolError("fresh processes did not use identical deterministic inputs")
    if len(state_hashes) != 1:
        raise BenchmarkProtocolError("fresh processes did not use identical initial state")
    if len(workloads) != 1:
        raise BenchmarkProtocolError("fresh processes did not use an identical workload")

    observed_orders = [tuple(result["order"]) for result in results]
    expected_orders = balanced_interleaved_orders(len(results))
    if observed_orders != expected_orders:
        raise BenchmarkProtocolError(
            "worker results are not in the required balanced/interleaved order"
        )

    aggregate = {}
    for implementation in IMPLEMENTATIONS:
        medians = [
            float(result["measurements"][implementation]["median_ms"])
            for result in results
        ]
        means = [
            float(result["measurements"][implementation]["mean_ms"])
            for result in results
        ]
        p90s = [
            float(result["measurements"][implementation]["p90_ms"])
            for result in results
        ]
        cold = [
            float(result["measurements"][implementation]["cold_latency_ms"])
            for result in results
        ]
        allocated = [
            result["measurements"][implementation][
                "peak_allocated_device_memory_bytes"
            ]
            for result in results
        ]
        reserved = [
            result["measurements"][implementation][
                "peak_reserved_device_memory_bytes"
            ]
            for result in results
        ]
        aggregate[implementation] = {
            "raw_process_medians_ms": medians,
            "median_of_process_medians_ms": statistics.median(medians),
            "mean_of_process_means_ms": statistics.fmean(means),
            "median_of_process_p90_ms": statistics.median(p90s),
            "median_cold_latency_ms": statistics.median(cold),
            "raw_peak_allocated_device_memory_bytes": allocated,
            "median_peak_allocated_device_memory_bytes": (
                statistics.median(float(value) for value in allocated)
                if all(value is not None for value in allocated)
                else None
            ),
            "raw_peak_reserved_device_memory_bytes": reserved,
            "median_peak_reserved_device_memory_bytes": (
                statistics.median(float(value) for value in reserved)
                if all(value is not None for value in reserved)
                else None
            ),
        }

    worker_evidence_flags = [bool(result.get("evidence_eligible")) for result in results]
    evidence_eligible = (
        not smoke
        and all(worker_evidence_flags)
        and all(
            not bool(result["source_and_build_identity"].get("git_dirty"))
            for result in results
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fresh_process_aggregate",
        "entrypoint": entrypoint_module,
        "case": results[0]["case"],
        "smoke": smoke,
        "evidence_eligible": evidence_eligible,
        "evidence_ineligibility_reasons": (
            []
            if evidence_eligible
            else [
                reason
                for reason, present in (
                    ("smoke_mode", smoke),
                    (
                        "worker_marked_non_evidence",
                        not smoke and not all(worker_evidence_flags),
                    ),
                    (
                        "dirty_source_tree",
                        any(
                            bool(result["source_and_build_identity"].get("git_dirty"))
                            for result in results
                        ),
                    ),
                )
                if present
            ]
        ),
        "fresh_processes": len(results),
        "order_policy": "alternating candidate/baseline first position",
        "orders": [list(order) for order in observed_orders],
        "input_hash_sha256": next(iter(input_hashes)),
        "state_hash_sha256": next(iter(state_hashes)),
        "workload": results[0]["workload"],
        "raw_process_results": list(results),
        "aggregate": aggregate,
    }


def run_orchestrator(
    entrypoint_module: str,
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, JsonValue]:
    """Launch balanced fresh workers and aggregate their JSON records."""

    _validate_common_args(args)
    results = []
    orders = balanced_interleaved_orders(args.fresh_processes)
    for process_index, order in enumerate(orders):
        command = _worker_command(entrypoint_module, args, process_index, order)
        environment = os.environ.copy()
        environment.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
        try:
            completed = runner(
                command,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"worker {process_index} failed with exit code {error.returncode}:\n"
                f"{error.stderr.strip()}"
            ) from error
        result = _load_single_json(completed.stdout, process_index)
        validate_worker_result(
            result,
            expected_entrypoint=entrypoint_module,
            expected_process_index=process_index,
            expected_order=order,
        )
        results.append(result)
    return aggregate_worker_results(entrypoint_module, results, smoke=bool(args.smoke))


def _emit(result: Mapping[str, object], output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output is not None:
        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
    print(payload)


def benchmark_main(entrypoint_module: str, argv: Sequence[str] | None = None) -> None:
    """CLI used by each representative benchmark entrypoint module."""

    entrypoint = _resolve_entrypoint(entrypoint_module)
    parser = _parser(entrypoint_module, entrypoint)
    args = parser.parse_args(argv)
    result = run_orchestrator(entrypoint_module, args)
    _emit(result, args.output)


def _worker_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Internal fresh-process benchmark worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--process-index", type=int, required=True)
    parser.add_argument("--order", nargs=2, choices=IMPLEMENTATIONS, required=True)
    known, remaining = parser.parse_known_args(argv)
    entrypoint = _resolve_entrypoint(known.entrypoint)
    worker_parser = _parser(known.entrypoint, entrypoint)
    args = worker_parser.parse_args(remaining)
    result = run_worker(
        known.entrypoint,
        args,
        process_index=known.process_index,
        order=known.order,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _worker_main()
