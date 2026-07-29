"""Compare packed-time, stepwise, and NPUGraph paths on a compact image SNN."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import time
from collections.abc import Callable

import torch

import spikingjelly_npu
from spikingjelly_npu.fedsnn import PackedBNTTConvNet
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu, is_npu_available


class BatchFirstCurrentSequence(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self, model: PackedBNTTConvNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch_first_sequence: torch.Tensor) -> torch.Tensor:
        return self.model.forward_current_seq(
            batch_first_sequence.transpose(0, 1).contiguous()
        )


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def inferred_cann_version() -> str | None:
    """Best-effort CANN version from the qualified environment path."""
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_AICPU_PATH"):
        path = os.environ.get(variable, "")
        match = re.search(r"cann[-_/]([0-9]+(?:\.[0-9]+)+)", path, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def runtime_metadata(device: torch.device) -> dict[str, object]:
    torch_npu_version = None
    device_name = platform.processor() or platform.machine()
    soc_version = None
    if device.type == "npu":
        import torch_npu

        torch_npu_version = torch_npu.__version__
        device_name = torch.npu.get_device_name(device)
        get_soc_version = getattr(torch.npu, "get_soc_version", None)
        if get_soc_version is not None:
            soc_version = get_soc_version()
    return {
        "device": str(device),
        "device_name": device_name,
        "soc_version": soc_version,
        "cann": inferred_cann_version(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu_version,
        "spikingjelly_npu": spikingjelly_npu.__version__,
    }


def measure(
    step: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    synchronize(device)
    cold_start = time.perf_counter()
    step()
    synchronize(device)
    cold_first_iteration_ms = (time.perf_counter() - cold_start) * 1000.0

    for _ in range(warmup):
        step()
    synchronize(device)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        step()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "cold_first_iteration_ms": cold_first_iteration_ms,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
    }


def make_step(
    function: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    model: torch.nn.Module,
    mode: str,
    learning_rate: float,
) -> Callable[[], torch.Tensor]:
    if mode == "forward":

        def forward_step() -> torch.Tensor:
            with torch.no_grad():
                return function(inputs)

        return forward_step

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def training_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = function(inputs)
        loss = logits.square().mean()
        loss.backward()
        optimizer.step()
        return loss

    return training_step


def build_model(
    time_steps: int,
    device: torch.device,
    training: bool,
) -> PackedBNTTConvNet:
    model = PackedBNTTConvNet(
        3,
        10,
        time_steps,
        channels=(32, 64),
        hidden_features=256,
        pooled_size=4,
    ).to(device)
    return model.train(training)


def snapshot_named_tensors(
    tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in tensors}


def run_training_parity_step(
    function: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    model: PackedBNTTConvNet,
    learning_rate: float,
) -> dict[str, object]:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits = function(inputs)
    loss = logits.square().mean()
    loss.backward()
    missing_gradients = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    gradients = snapshot_named_tensors(
        [
            (name, parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        ]
    )
    buffers = snapshot_named_tensors(list(model.named_buffers()))
    optimizer.step()
    parameters = snapshot_named_tensors(list(model.named_parameters()))
    return {
        "logits": logits.detach().cpu().clone(),
        "loss": float(loss.detach().cpu()),
        "gradients": gradients,
        "buffers": buffers,
        "parameters": parameters,
        "missing_gradients": missing_gradients,
    }


def compare_named_tensors(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    if reference.keys() != candidate.keys():
        return {
            "passed": False,
            "reference_names": sorted(reference),
            "candidate_names": sorted(candidate),
        }
    maximum_absolute_error = 0.0
    maximum_tolerance_ratio = 0.0
    mismatched_elements = 0
    total_elements = 0
    failed_tensors = []
    for name, expected in reference.items():
        actual = candidate[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            failed_tensors.append(name)
            continue
        total_elements += expected.numel()
        if expected.is_floating_point() or expected.is_complex():
            close = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
            tensor_mismatches = int((~close).sum())
            if tensor_mismatches:
                failed_tensors.append(name)
                mismatched_elements += tensor_mismatches
            absolute_error = (actual - expected).abs()
            tolerance = atol + rtol * expected.abs()
            finite = torch.isfinite(absolute_error)
            if finite.any():
                maximum_absolute_error = max(
                    maximum_absolute_error,
                    float(absolute_error[finite].max()),
                )
                maximum_tolerance_ratio = max(
                    maximum_tolerance_ratio,
                    float((absolute_error[finite] / tolerance[finite]).max()),
                )
        elif not torch.equal(actual, expected):
            failed_tensors.append(name)
            mismatched_elements += int((actual != expected).sum())
    return {
        "passed": not failed_tensors,
        "tensor_count": len(reference),
        "failed_tensor_count": len(failed_tensors),
        "failed_tensors": failed_tensors,
        "mismatched_elements": mismatched_elements,
        "total_elements": total_elements,
        "max_absolute_error": maximum_absolute_error,
        "max_tolerance_ratio": maximum_tolerance_ratio,
    }


def compare_training_results(
    reference: dict[str, object],
    candidate: dict[str, object],
    *,
    rtol: float,
    atol: float,
    require_close: bool,
) -> dict[str, object]:
    comparisons = {
        "logits": compare_named_tensors(
            {"logits": reference["logits"]},
            {"logits": candidate["logits"]},
            rtol=rtol,
            atol=atol,
        ),
        "gradients": compare_named_tensors(
            reference["gradients"], candidate["gradients"], rtol=rtol, atol=atol
        ),
        "bntt_and_other_buffers": compare_named_tensors(
            reference["buffers"], candidate["buffers"], rtol=rtol, atol=atol
        ),
        "post_sgd_parameters": compare_named_tensors(
            reference["parameters"], candidate["parameters"], rtol=rtol, atol=atol
        ),
    }
    missing_gradients = {
        "reference": reference["missing_gradients"],
        "candidate": candidate["missing_gradients"],
    }
    loss_absolute_error = abs(float(candidate["loss"]) - float(reference["loss"]))
    loss_tolerance = atol + rtol * abs(float(reference["loss"]))
    passed = (
        not missing_gradients["reference"]
        and not missing_gradients["candidate"]
        and loss_absolute_error <= loss_tolerance
        and all(comparison["passed"] for comparison in comparisons.values())
    )
    result = {
        "passed": passed,
        **comparisons,
        "missing_gradients": missing_gradients,
        "loss_absolute_error": loss_absolute_error,
        "loss_tolerance": loss_tolerance,
    }
    if require_close and not passed:
        raise AssertionError(f"required training parity failed: {result}")
    return result


def training_parity(
    initial_state: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    device: torch.device,
    time_steps: int,
    batch_size: int,
    learning_rate: float,
    rtol: float,
    atol: float,
    require_close: bool,
) -> dict[str, object]:
    packed_model = build_model(time_steps, device, True)
    packed_model.load_state_dict(initial_state)
    stepwise_model = build_model(time_steps, device, True)
    stepwise_model.load_state_dict(initial_state)
    graph_inner = build_model(time_steps, device, True)
    graph_inner.load_state_dict(initial_state)
    graph_model = BatchFirstCurrentSequence(graph_inner).train(True)
    runner = StaticGraphRunner(
        graph_model,
        batch_size,
        strict=True,
        allow_training=True,
    )

    packed = run_training_parity_step(
        packed_model.forward_current_seq, inputs, packed_model, learning_rate
    )
    stepwise = run_training_parity_step(
        stepwise_model.forward_current_seq_stepwise,
        inputs,
        stepwise_model,
        learning_rate,
    )
    graph = run_training_parity_step(
        runner,
        inputs.transpose(0, 1).contiguous(),
        graph_inner,
        learning_rate,
    )
    if device.type == "npu" and runner.last_route.backend != "npugraph":
        raise AssertionError(f"training parity did not capture NPUGraph: {runner.last_route}")
    stepwise_comparison = compare_training_results(
        packed,
        stepwise,
        rtol=rtol,
        atol=atol,
        require_close=False,
    )
    graph_comparison = compare_training_results(
        packed,
        graph,
        rtol=rtol,
        atol=atol,
        require_close=require_close,
    )
    return {
        "passed": graph_comparison["passed"],
        "required_gate": "npugraph_vs_packed" if require_close else None,
        "rtol": rtol,
        "atol": atol,
        "stepwise_vs_packed_diagnostic": stepwise_comparison,
        "npugraph_vs_packed": graph_comparison,
        "graph_route": runner.last_route.__dict__,
        "graph_capture_error": runner.capture_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--mode", choices=("forward", "train"), default="forward")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--parity-rtol", type=float, default=5e-4)
    parser.add_argument("--parity-atol", type=float, default=5e-5)
    parser.add_argument(
        "--require-training-parity",
        action="store_true",
        help="exit nonzero if experimental training NPUGraph differs from packed eager",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.time_steps <= 0:
        raise ValueError("batch-size and time-steps must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.parity_rtol < 0 or args.parity_atol < 0:
        raise ValueError("parity tolerances must be non-negative")

    if args.device == "auto":
        device = configure_npu("npu:0") if is_npu_available() else torch.device("cpu")
    elif args.device.startswith("npu"):
        device = configure_npu(args.device)
    else:
        device = torch.device(args.device)

    training = args.mode == "train"
    if training:
        torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(args.seed)
    reference = build_model(args.time_steps, device, training)
    initial_state = reference.state_dict()
    packed_model = build_model(args.time_steps, device, training)
    packed_model.load_state_dict(initial_state)
    stepwise_model = build_model(args.time_steps, device, training)
    stepwise_model.load_state_dict(initial_state)
    graph_inner = build_model(args.time_steps, device, training)
    graph_inner.load_state_dict(initial_state)
    graph_model = BatchFirstCurrentSequence(graph_inner).train(training)

    inputs = torch.rand(
        args.time_steps,
        args.batch_size,
        3,
        args.height,
        args.width,
        device=device,
    )
    batch_first_inputs = inputs.transpose(0, 1).contiguous()
    runner = StaticGraphRunner(
        graph_model,
        args.batch_size,
        strict=False,
        allow_training=training,
    )

    packed_step = make_step(
        packed_model.forward_current_seq,
        inputs,
        packed_model,
        args.mode,
        args.learning_rate,
    )
    stepwise_step = make_step(
        stepwise_model.forward_current_seq_stepwise,
        inputs,
        stepwise_model,
        args.mode,
        args.learning_rate,
    )
    graph_step = make_step(
        runner,
        batch_first_inputs,
        graph_model,
        args.mode,
        args.learning_rate,
    )

    packed = measure(packed_step, device, args.warmup, args.iterations)
    stepwise = measure(stepwise_step, device, args.warmup, args.iterations)
    graph = measure(graph_step, device, args.warmup, args.iterations)
    parity = None
    if training:
        parity = training_parity(
            initial_state,
            inputs,
            device,
            args.time_steps,
            args.batch_size,
            args.learning_rate,
            args.parity_rtol,
            args.parity_atol,
            args.require_training_parity,
        )
    result = {
        **runtime_metadata(device),
        "mode": args.mode,
        "model_training": training,
        "dtype": str(inputs.dtype),
        "shape_time_first": list(inputs.shape),
        "shape_graph_input": list(batch_first_inputs.shape),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "learning_rate": args.learning_rate if training else None,
        "training_graph_policy": (
            "explicit opt-in with deterministic algorithms in this benchmark; "
            "disabled by default in StaticGraphRunner"
            if training
            else "inference graph enabled for qualified fixed batches"
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "synchronization": "torch.npu.synchronize after every measured iteration"
        if device.type == "npu"
        else "synchronous CPU execution",
        "packed": packed,
        "stepwise": stepwise,
        "static_graph": graph,
        "packed_speedup_vs_stepwise": stepwise["median_ms"] / packed["median_ms"],
        "graph_speedup_vs_stepwise": stepwise["median_ms"] / graph["median_ms"],
        "graph_speedup_vs_packed": packed["median_ms"] / graph["median_ms"],
        "graph_route": runner.last_route.__dict__,
        "graph_capture_error": runner.capture_error,
        "training_parity": parity,
        "capture_state_restoration": "covered by tests; any rollback failure is fatal",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
