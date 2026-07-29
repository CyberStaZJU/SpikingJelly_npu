#!/usr/bin/env python3
"""Qualify native AsPy IF determinism, SGD parity, and NPUGraph routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import time
from typing import Any

import torch
from torch import nn

import spikingjelly_npu
from spikingjelly_npu.activation_based import functional, neuron, surrogate
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def inferred_cann_version() -> str | None:
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_AICPU_PATH"):
        match = re.search(
            r"cann[-_/]([0-9]+(?:\.[0-9]+)+)",
            os.environ.get(variable, ""),
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return None


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    byte_view = value.view(torch.uint8).reshape(-1)
    return hashlib.sha256(bytes(byte_view.tolist())).hexdigest()


def tensor_error(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    difference = (actual.detach() - expected.detach()).abs()
    return {
        "max_absolute_error": float(difference.max().cpu()) if difference.numel() else 0.0,
        "mean_absolute_error": float(difference.mean().cpu()) if difference.numel() else 0.0,
    }


class IFTrainingModel(nn.Module):
    def __init__(self, backend: str, *, features: int = 16, classes: int = 5) -> None:
        super().__init__()
        self.input = nn.Linear(features, features, bias=True)
        self.if_node = neuron.IFNode(
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=2.5),
            detach_reset=False,
            step_mode="m",
            backend=backend,
            backend_strict=backend == "aspy",
            store_v_seq=True,
        )
        self.readout = nn.Linear(features, classes, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        currents = self.input(inputs)
        spikes = self.if_node(currents)
        self.last_spikes = spikes.detach()
        self.last_voltage = self.if_node.v.detach()
        return self.readout(spikes).mean(0)


class BatchFirstIF(nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.if_node = neuron.IFNode(
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=True,
            step_mode="m",
            backend=backend,
            backend_strict=backend == "aspy",
        )

    def forward(self, batch_first: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.if_node(batch_first.transpose(0, 1).contiguous()).mean(0)


class BatchFirstIFTrainingModel(nn.Module):
    """Fixed-shape trainable proxy shared by all three performance paths."""

    _spikingjelly_npu_graph_safe = True

    def __init__(self, backend: str, *, features: int, classes: int = 16) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.if_node = neuron.IFNode(
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=True,
            step_mode="m",
            backend=backend,
            backend_strict=backend == "aspy",
        )
        self.readout = nn.Linear(features, classes)

    def forward(self, batch_first: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        currents = batch_first * self.gain
        spikes = self.if_node(currents.transpose(0, 1).contiguous())
        return self.readout(spikes.mean(0))


def build_training_pair(
    device: torch.device, seed: int
) -> tuple[IFTrainingModel, IFTrainingModel]:
    torch.manual_seed(seed)
    reference = IFTrainingModel("torch").to(device).train()
    accelerated = IFTrainingModel("aspy").to(device).train()
    accelerated.load_state_dict(reference.state_dict())
    return reference, accelerated


def make_training_batches(
    device: torch.device, seed: int, steps: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    return [
        (
            torch.rand(6, 8, 16, dtype=torch.float32, generator=generator).to(device),
            torch.rand(8, 5, dtype=torch.float32, generator=generator).to(device),
        )
        for _ in range(steps)
    ]


def run_sgd_trajectory(
    device: torch.device, seed: int, steps: int, learning_rate: float
) -> dict[str, Any]:
    reference, accelerated = build_training_pair(device, seed)
    batches = make_training_batches(device, seed, steps)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
    accelerated_optimizer = torch.optim.SGD(accelerated.parameters(), lr=learning_rate)
    first_failed_step = None
    maxima = {
        "logit": 0.0,
        "loss": 0.0,
        "spike": 0.0,
        "voltage": 0.0,
        "gradient": 0.0,
        "parameter": 0.0,
    }

    for step, (inputs, target) in enumerate(batches):
        reference_optimizer.zero_grad(set_to_none=True)
        accelerated_optimizer.zero_grad(set_to_none=True)
        expected = reference(inputs)
        actual = accelerated(inputs)
        expected_loss = (expected - target).square().mean()
        actual_loss = (actual - target).square().mean()
        expected_loss.backward()
        actual_loss.backward()
        synchronize(device)

        maxima["logit"] = max(
            maxima["logit"], tensor_error(expected, actual)["max_absolute_error"]
        )
        maxima["loss"] = max(
            maxima["loss"], abs(float(expected_loss.detach()) - float(actual_loss.detach()))
        )
        maxima["spike"] = max(
            maxima["spike"],
            tensor_error(reference.last_spikes, accelerated.last_spikes)[
                "max_absolute_error"
            ],
        )
        maxima["voltage"] = max(
            maxima["voltage"],
            tensor_error(reference.last_voltage, accelerated.last_voltage)[
                "max_absolute_error"
            ],
        )
        step_failed = (
            not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
            or not torch.allclose(
                actual_loss.detach(), expected_loss.detach(), rtol=2e-5, atol=2e-6
            )
            or not torch.equal(accelerated.last_spikes, reference.last_spikes)
            or not torch.allclose(
                accelerated.last_voltage,
                reference.last_voltage,
                rtol=2e-5,
                atol=2e-6,
            )
        )
        for (reference_name, reference_parameter), (
            accelerated_name,
            accelerated_parameter,
        ) in zip(reference.named_parameters(), accelerated.named_parameters(), strict=True):
            if reference_name != accelerated_name:
                raise AssertionError("training model parameter names differ")
            if reference_parameter.grad is None or accelerated_parameter.grad is None:
                raise AssertionError(f"missing gradient for {reference_name}")
            gradient_error = tensor_error(
                reference_parameter.grad, accelerated_parameter.grad
            )["max_absolute_error"]
            maxima["gradient"] = max(maxima["gradient"], gradient_error)
            step_failed = step_failed or not torch.allclose(
                accelerated_parameter.grad,
                reference_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
            )

        reference_optimizer.step()
        accelerated_optimizer.step()
        for reference_parameter, accelerated_parameter in zip(
            reference.parameters(), accelerated.parameters(), strict=True
        ):
            parameter_error = tensor_error(
                reference_parameter, accelerated_parameter
            )["max_absolute_error"]
            maxima["parameter"] = max(maxima["parameter"], parameter_error)
            step_failed = step_failed or not torch.allclose(
                accelerated_parameter,
                reference_parameter,
                rtol=2e-5,
                atol=2e-6,
            )
        if step_failed and first_failed_step is None:
            first_failed_step = step

    return {
        "steps": steps,
        "learning_rate": learning_rate,
        "first_failed_step": first_failed_step,
        "passed": first_failed_step is None,
        "maximum_errors": maxima,
        "final_parameter_hashes": {
            name: tensor_hash(parameter)
            for name, parameter in accelerated.named_parameters()
        },
        "last_backend_route": accelerated.if_node.last_backend_route.__dict__,
    }


def run_determinism_probe(device: torch.device, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.rand(
        7, 4, 17, dtype=torch.float32, generator=generator
    ).to(device).requires_grad_(True)
    node = neuron.IFNode(
        v_reset=None,
        surrogate_function=surrogate.ATan(alpha=2.5),
        detach_reset=False,
        step_mode="m",
        backend="aspy",
        backend_strict=True,
        store_v_seq=True,
    ).to(device)
    output = node(inputs)
    output_weight = torch.linspace(
        0.25, 1.25, output.numel(), dtype=output.dtype, device=device
    ).reshape_as(output)
    voltage_weight = torch.linspace(
        0.1, 0.9, node.v_seq.numel(), dtype=node.v_seq.dtype, device=device
    ).reshape_as(node.v_seq)
    loss = (
        (output * output_weight).sum()
        + (node.v_seq * voltage_weight).sum()
        + node.v.square().sum()
    )
    loss.backward()
    synchronize(device)
    return {
        "output": tensor_hash(output),
        "v_seq": tensor_hash(node.v_seq),
        "v_final": tensor_hash(node.v),
        "input_gradient": tensor_hash(inputs.grad),
        "loss": tensor_hash(loss.reshape(1)),
        "last_backend_route": node.last_backend_route.__dict__,
    }


def run_graph_probe(device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = BatchFirstIF("aspy").to(device).eval()
    runner = StaticGraphRunner(
        model,
        batch_size=8,
        strict=True,
        assume_graph_safe=True,
    )
    inputs = torch.rand(8, 6, 16, dtype=torch.float32, device=device)
    eager_inputs = inputs.detach().clone().requires_grad_(True)
    graph_inputs = inputs.detach().clone().requires_grad_(True)

    functional.reset_net(model)
    eager_output = model(eager_inputs)
    eager_loss = eager_output.square().sum()
    eager_loss.backward()
    eager_gradient = eager_inputs.grad.detach().clone()
    eager_spikes = model.if_node.last_backend_route.__dict__.copy()

    graph_output = runner(graph_inputs)
    graph_loss = graph_output.square().sum()
    graph_loss.backward()
    synchronize(device)
    if runner.last_route.backend != "npugraph" or not runner.last_route.captured:
        raise AssertionError(f"AsPy did not enter NPUGraph: {runner.last_route}")
    if runner.capture_error is not None:
        raise AssertionError(f"AsPy NPUGraph capture recorded an error: {runner.capture_error}")
    torch.testing.assert_close(graph_output, eager_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(graph_inputs.grad, eager_gradient, rtol=2e-5, atol=2e-6)
    if model.if_node.last_backend_route.backend != "aspy":
        raise AssertionError(
            f"inner IF did not use AsPy during capture: {model.if_node.last_backend_route}"
        )
    return {
        "eager_output_hash": tensor_hash(eager_output),
        "graph_output_hash": tensor_hash(graph_output),
        "eager_input_gradient_hash": tensor_hash(eager_gradient),
        "graph_input_gradient_hash": tensor_hash(graph_inputs.grad),
        "output_error": tensor_error(eager_output, graph_output),
        "input_gradient_error": tensor_error(eager_gradient, graph_inputs.grad),
        "route": runner.last_route.__dict__,
        "capture_error": runner.capture_error,
        "eager_aspy_route": eager_spikes,
        "capture_aspy_route": model.if_node.last_backend_route.__dict__,
    }


def build_performance_model(
    device: torch.device,
    backend: str,
    *,
    features: int,
    state_dict: dict[str, torch.Tensor],
) -> BatchFirstIFTrainingModel:
    model = BatchFirstIFTrainingModel(backend, features=features).to(device).train()
    model.load_state_dict(state_dict)
    return model


def run_performance_step(
    callable_model: Any,
    model: BatchFirstIFTrainingModel,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for parameter in model.parameters():
        parameter.grad = None
    inputs.grad = None
    output = callable_model(inputs)
    loss = (output - target).square().mean()
    loss.backward()
    return output, loss


def measure_training_path(
    device: torch.device,
    callable_model: Any,
    model: BatchFirstIFTrainingModel,
    inputs: torch.Tensor,
    target: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    synchronize(device)
    cold_start = time.perf_counter()
    run_performance_step(callable_model, model, inputs, target)
    synchronize(device)
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    for _ in range(warmup):
        run_performance_step(callable_model, model, inputs, target)
    synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_performance_step(callable_model, model, inputs, target)
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    route = None
    if isinstance(callable_model, StaticGraphRunner):
        route = callable_model.last_route.__dict__
    return {
        "cold_ms": cold_ms,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
        "route": route,
    }


def run_performance_probe(
    device: torch.device, seed: int, warmup: int, iterations: int
) -> dict[str, Any]:
    batch_size, time_steps, features = 64, 8, 4096
    shape = (batch_size, time_steps, features)
    torch.manual_seed(seed)
    initial = BatchFirstIFTrainingModel("torch", features=features)
    state_dict = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    base_inputs = torch.rand(shape, dtype=torch.float32, generator=generator).to(device)
    target = torch.rand(batch_size, 16, dtype=torch.float32, generator=generator).to(device)

    torch_model = build_performance_model(
        device, "torch", features=features, state_dict=state_dict
    )
    aspy_model = build_performance_model(
        device, "aspy", features=features, state_dict=state_dict
    )
    graph_model = build_performance_model(
        device, "aspy", features=features, state_dict=state_dict
    )
    graph_runner = StaticGraphRunner(
        graph_model,
        batch_size=batch_size,
        strict=True,
        allow_training=True,
        assume_graph_safe=True,
    )
    inputs_by_path = {
        "torch_stepwise": base_inputs.detach().clone().requires_grad_(True),
        "aspy_eager": base_inputs.detach().clone().requires_grad_(True),
        "aspy_npugraph": base_inputs.detach().clone().requires_grad_(True),
    }

    references: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
    for name, callable_model, model in (
        ("torch_stepwise", torch_model, torch_model),
        ("aspy_eager", aspy_model, aspy_model),
        ("aspy_npugraph", graph_runner, graph_model),
    ):
        output, loss = run_performance_step(
            callable_model, model, inputs_by_path[name], target
        )
        synchronize(device)
        references[name] = (
            output.detach().clone(),
            loss.detach().clone(),
            {
                parameter_name: parameter.grad.detach().clone()
                for parameter_name, parameter in model.named_parameters()
            },
        )
    expected_output, expected_loss, expected_gradients = references["torch_stepwise"]
    parity: dict[str, Any] = {}
    for name in ("aspy_eager", "aspy_npugraph"):
        output, loss, gradients = references[name]
        torch.testing.assert_close(output, expected_output, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-6)
        gradient_errors = {
            parameter_name: tensor_error(expected_gradients[parameter_name], gradient)
            for parameter_name, gradient in gradients.items()
        }
        for parameter_name, gradient in gradients.items():
            torch.testing.assert_close(
                gradient,
                expected_gradients[parameter_name],
                rtol=2e-5,
                atol=2e-6,
            )
        parity[name] = {
            "output_error": tensor_error(expected_output, output),
            "loss_absolute_error": abs(float(loss) - float(expected_loss)),
            "parameter_gradient_errors": gradient_errors,
        }
    if graph_runner.last_route.backend != "npugraph" or not graph_runner.last_route.captured:
        raise AssertionError(f"performance graph path did not capture: {graph_runner.last_route}")

    measurements = {
        "torch_stepwise": measure_training_path(
            device,
            torch_model,
            torch_model,
            inputs_by_path["torch_stepwise"],
            target,
            warmup,
            iterations,
        ),
        "aspy_eager": measure_training_path(
            device,
            aspy_model,
            aspy_model,
            inputs_by_path["aspy_eager"],
            target,
            warmup,
            iterations,
        ),
        "aspy_npugraph": measure_training_path(
            device,
            graph_runner,
            graph_model,
            inputs_by_path["aspy_npugraph"],
            target,
            warmup,
            iterations,
        ),
    }
    return {
        "shape_batch_time_features": list(shape),
        "dtype": "torch.float32",
        "warmup": warmup,
        "iterations": iterations,
        "scope": "full fixed-shape training step: reset, gain, IF, linear readout, MSE, backward",
        "paths": measurements,
        "parity": parity,
        "speedups_vs_torch_stepwise": {
            name: measurements["torch_stepwise"]["median_ms"]
            / measurements[name]["median_ms"]
            for name in ("aspy_eager", "aspy_npugraph")
        },
    }


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    import torch_npu

    return {
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
        "cann": inferred_cann_version(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "spikingjelly_npu": spikingjelly_npu.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "synchronization": "torch.npu.synchronize after each measured iteration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--trajectory-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=("determinism", "trajectory", "graph", "performance", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.trajectory_steps <= 0 or args.learning_rate <= 0:
        raise ValueError("trajectory steps and learning rate must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    device = configure_npu(args.device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    result: dict[str, Any] = {**runtime_metadata(device), "seed": args.seed}
    if args.mode in {"determinism", "all"}:
        result["determinism"] = run_determinism_probe(device, args.seed)
    if args.mode in {"trajectory", "all"}:
        result["trajectory"] = run_sgd_trajectory(
            device, args.seed, args.trajectory_steps, args.learning_rate
        )
    if args.mode in {"graph", "all"}:
        result["npugraph"] = run_graph_probe(device, args.seed)
    if args.mode in {"performance", "all"}:
        result["performance"] = run_performance_probe(
            device, args.seed, args.warmup, args.iterations
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
