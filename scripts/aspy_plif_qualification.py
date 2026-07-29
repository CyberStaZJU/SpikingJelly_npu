#!/usr/bin/env python3
"""Qualify native AsPy PLIF determinism, optimizer parity, graph replay, and speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from typing import Any

import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, neuron, surrogate
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu


def synchronize(device: torch.device) -> None:
    torch.npu.synchronize(device)


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    return hashlib.sha256(bytes(value.tolist())).hexdigest()


def maximum_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float((actual.detach() - expected.detach()).abs().max().cpu())


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    import torch_npu

    return {
        "host": platform.node(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def make_node(backend: str, *, store_v_seq: bool = True) -> neuron.ParametricLIFNode:
    return neuron.ParametricLIFNode(
        init_tau=2.5,
        decay_input=True,
        v_threshold=0.7,
        v_reset=None,
        surrogate_function=surrogate.ATan(alpha=2.5),
        detach_reset=False,
        step_mode="m",
        backend=backend,
        backend_strict=backend == "aspy",
        store_v_seq=store_v_seq,
    )


def run_determinism(device: torch.device, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = torch.rand(7, 4, 17, generator=generator).to(device).requires_grad_(True)
    node = make_node("aspy")
    node = node.to(device)
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
        "hashes": {
            "output": tensor_hash(output),
            "v_seq": tensor_hash(node.v_seq),
            "v_final": tensor_hash(node.v),
            "input_gradient": tensor_hash(inputs.grad),
            "w_gradient": tensor_hash(node.w.grad),
        },
        "w_gradient_value": float(node.w.grad.cpu()),
        "route": node.last_backend_route.__dict__,
    }


class TrainingModel(nn.Module):
    def __init__(self, backend: str) -> None:
        super().__init__()
        self.input = nn.Linear(16, 16)
        self.node = make_node(backend, store_v_seq=False)
        self.readout = nn.Linear(16, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.readout(self.node(self.input(inputs))).mean(0)


def run_trajectory(device: torch.device, seed: int, steps: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    reference = TrainingModel("torch").to(device).train()
    accelerated = TrainingModel("aspy").to(device).train()
    accelerated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    accelerated_optimizer = torch.optim.SGD(accelerated.parameters(), lr=0.01)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    maximum_gradient_error = 0.0
    maximum_parameter_error = 0.0
    first_failed_step = None

    for step in range(steps):
        inputs = torch.rand(6, 8, 16, generator=generator).to(device)
        target = torch.rand(8, 5, generator=generator).to(device)
        reference_optimizer.zero_grad(set_to_none=True)
        accelerated_optimizer.zero_grad(set_to_none=True)
        expected = reference(inputs)
        actual = accelerated(inputs)
        expected_loss = (expected - target).square().mean()
        actual_loss = (actual - target).square().mean()
        expected_loss.backward()
        actual_loss.backward()
        synchronize(device)
        failed = not torch.allclose(actual, expected, rtol=3e-5, atol=3e-6)
        for (name_expected, parameter_expected), (name_actual, parameter_actual) in zip(
            reference.named_parameters(), accelerated.named_parameters(), strict=True
        ):
            if name_expected != name_actual:
                raise AssertionError("parameter name mismatch")
            error = maximum_error(parameter_expected.grad, parameter_actual.grad)
            maximum_gradient_error = max(maximum_gradient_error, error)
            failed = failed or not torch.allclose(
                parameter_actual.grad, parameter_expected.grad, rtol=3e-5, atol=3e-6
            )
        reference_optimizer.step()
        accelerated_optimizer.step()
        for parameter_expected, parameter_actual in zip(
            reference.parameters(), accelerated.parameters(), strict=True
        ):
            error = maximum_error(parameter_expected, parameter_actual)
            maximum_parameter_error = max(maximum_parameter_error, error)
            failed = failed or not torch.allclose(
                parameter_actual, parameter_expected, rtol=3e-5, atol=3e-6
            )
        if failed and first_failed_step is None:
            first_failed_step = step

    return {
        "steps": steps,
        "passed": first_failed_step is None,
        "first_failed_step": first_failed_step,
        "maximum_gradient_error": maximum_gradient_error,
        "maximum_parameter_error": maximum_parameter_error,
        "final_parameter_hashes": {
            name: tensor_hash(parameter)
            for name, parameter in accelerated.named_parameters()
        },
        "route": accelerated.node.last_backend_route.__dict__,
    }


class GraphModel(nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self, backend: str) -> None:
        super().__init__()
        self.node = neuron.ParametricLIFNode(
            init_tau=2.5,
            decay_input=True,
            v_threshold=0.35,
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=False,
            step_mode="m",
            backend=backend,
            backend_strict=backend == "aspy",
        )

    def forward(self, batch_first: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def run_graph(device: torch.device, seed: int) -> dict[str, Any]:
    eager = GraphModel("aspy").to(device).train()
    graph = GraphModel("aspy").to(device).train()
    graph.load_state_dict(eager.state_dict())
    runner = StaticGraphRunner(
        graph,
        batch_size=4,
        strict=True,
        allow_training=True,
        assume_graph_safe=True,
    )
    replays = []
    previous_hash = None
    for replay in range(5):
        with torch.no_grad():
            new_w = torch.tensor(-0.8 + replay * 0.3, dtype=torch.float32, device=device)
            eager.node.w.copy_(new_w)
            graph.node.w.copy_(new_w)
        torch.manual_seed(seed + replay)
        inputs = torch.rand(4, 3, 17, dtype=torch.float32, device=device)
        eager_inputs = inputs.detach().clone().requires_grad_(True)
        graph_inputs = inputs.detach().clone().requires_grad_(True)
        eager.node.w.grad = None
        graph.node.w.grad = None
        expected = eager(eager_inputs)
        expected.square().sum().backward()
        actual = runner(graph_inputs)
        actual.square().sum().backward()
        synchronize(device)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(graph_inputs.grad, eager_inputs.grad, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(graph.node.w.grad, eager.node.w.grad, rtol=3e-5, atol=3e-6)
        output_hash = tensor_hash(actual)
        if previous_hash is not None and output_hash == previous_hash:
            raise AssertionError("dynamic graph replay did not change output hash")
        previous_hash = output_hash
        replays.append(
            {
                "w": float(new_w.cpu()),
                "output_hash": output_hash,
                "input_gradient_hash": tensor_hash(graph_inputs.grad),
                "w_gradient_hash": tensor_hash(graph.node.w.grad),
            }
        )
    return {
        "replays": replays,
        "route": runner.last_route.__dict__,
        "inner_route": graph.node.last_backend_route.__dict__,
    }


def run_step(
    node: neuron.ParametricLIFNode, inputs: torch.Tensor, weight: torch.Tensor
) -> None:
    functional.reset_net(node)
    node.w.grad = None
    inputs.grad = None
    output = node(inputs)
    loss = (output * weight).sum() + node.v.square().mean()
    loss.backward()


def measure(
    device: torch.device,
    node: neuron.ParametricLIFNode,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        run_step(node, inputs, weight)
    synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_step(node, inputs, weight)
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_performance(
    device: torch.device, seed: int, warmup: int, iterations: int
) -> dict[str, Any]:
    shape = (8, 64, 4096)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.rand(shape, generator=generator).to(device)
    weight = torch.rand(shape, generator=generator).to(device)
    reference = make_node("torch", store_v_seq=False).to(device)
    accelerated = make_node("aspy", store_v_seq=False).to(device)
    accelerated.load_state_dict(reference.state_dict())
    expected_inputs = base.detach().clone().requires_grad_(True)
    actual_inputs = base.detach().clone().requires_grad_(True)
    run_step(reference, expected_inputs, weight)
    run_step(accelerated, actual_inputs, weight)
    synchronize(device)
    torch.testing.assert_close(actual_inputs.grad, expected_inputs.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=3e-5, atol=3e-6)
    torch_path = measure(
        device,
        reference,
        base.detach().clone().requires_grad_(True),
        weight,
        warmup,
        iterations,
    )
    aspy_path = measure(
        device,
        accelerated,
        base.detach().clone().requires_grad_(True),
        weight,
        warmup,
        iterations,
    )
    return {
        "shape_time_batch_features": list(shape),
        "warmup": warmup,
        "iterations": iterations,
        "synchronization": "torch.npu.synchronize after warmup and every measured iteration",
        "paths": {"pytorch_plif": torch_path, "aspy_plif": aspy_path},
        "speedup": torch_path["median_ms"] / aspy_path["median_ms"],
        "route": accelerated.last_backend_route.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--mode", choices=("determinism", "trajectory", "graph", "performance"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    device = configure_npu(args.device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    result: dict[str, Any] = {**runtime_metadata(device), "seed": args.seed}
    if args.mode == "determinism":
        result["determinism"] = run_determinism(device, args.seed)
    elif args.mode == "trajectory":
        result["trajectory"] = run_trajectory(device, args.seed, args.steps)
    elif args.mode == "graph":
        result["npugraph"] = run_graph(device, args.seed)
    else:
        result["performance"] = run_performance(
            device, args.seed, args.warmup, args.iterations
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
