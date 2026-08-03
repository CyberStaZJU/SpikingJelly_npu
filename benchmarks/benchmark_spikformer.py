"""Representative SpikingSelfAttention and Spikformer benchmark entrypoint."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from benchmarks import _protocol
from spikingjelly_npu.activation_based import functional
from spikingjelly_npu.activation_based.layer import SpikingSelfAttention
from spikingjelly_npu.activation_based.model.spikformer import Spikformer


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", choices=("attention", "spikformer"), default="attention")
    parser.add_argument("--backend", choices=("torch", "npu", "aspy"), default="torch")
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=196)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _protocol.BenchmarkProtocolError(f"{name} must be a positive integer")
    return value


def _state_payload(module: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _build_module_and_input(args: argparse.Namespace, device: torch.device):
    if args.case == "attention":
        module = SpikingSelfAttention(
            dim=args.d_model,
            num_heads=args.heads,
            backend=args.backend,
        ).to(device)
        inputs = torch.randn(
            args.time_steps,
            args.batch_size,
            args.d_model,
            args.tokens,
            device=device,
        )
        shape = {
            "T": args.time_steps,
            "B": args.batch_size,
            "tokens": args.tokens,
            "D": args.d_model,
        }
        return module, inputs, shape
    module = Spikformer(
        T=args.time_steps,
        in_channels=3,
        img_size_h=args.image_size,
        img_size_w=args.image_size,
        num_classes=args.num_classes,
        embed_dims=args.d_model,
        num_heads=args.heads,
        mlp_ratio=4.0,
        depths=args.blocks,
        backend=args.backend,
    ).to(device)
    inputs = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )
    shape = {
        "T": args.time_steps,
        "B": args.batch_size,
        "image": [3, args.image_size, args.image_size],
        "tokens": (args.image_size // 16) ** 2,
        "D": args.d_model,
    }
    return module, inputs, shape


def _build_case(args: argparse.Namespace, device: torch.device) -> _protocol.BenchmarkCase:
    for name in (
        "time_steps",
        "batch_size",
        "tokens",
        "d_model",
        "heads",
        "blocks",
        "image_size",
        "num_classes",
    ):
        _positive(name, getattr(args, name))
    if args.d_model % args.heads != 0:
        raise _protocol.BenchmarkProtocolError("d_model must be divisible by heads")
    if args.case == "spikformer":
        if args.d_model % 8 != 0:
            raise _protocol.BenchmarkProtocolError(
                "Spikformer d_model must be divisible by eight"
            )
        if args.image_size < 16 or args.image_size % 16 != 0:
            raise _protocol.BenchmarkProtocolError(
                "Spikformer image_size must be a multiple of 16"
            )
    if args.learning_rate <= 0.0:
        raise _protocol.BenchmarkProtocolError("learning_rate must be positive")
    if device.type == "cpu" and args.backend != "torch":
        raise _protocol.BenchmarkProtocolError(
            "CPU smoke runs must use --backend torch; accelerator routing requires NPU"
        )
    if device.type == "npu" and args.backend == "torch":
        raise _protocol.BenchmarkProtocolError(
            "NPU sequence qualification must request --backend npu or --backend aspy "
            "so provider intent is explicit"
        )

    torch.manual_seed(args.seed)
    reference, inputs, shape = _build_module_and_input(args, device)
    initial_state = reference.state_dict()
    torch.manual_seed(args.seed + 1)
    candidate, _, _ = _build_module_and_input(args, device)
    torch.manual_seed(args.seed + 2)
    baseline, _, _ = _build_module_and_input(args, device)
    candidate.load_state_dict(initial_state)
    baseline.load_state_dict(initial_state)
    training = args.mode == "train"
    candidate.train(training)
    baseline.train(training)

    def make_step(module: nn.Module):
        optimizer = (
            torch.optim.SGD(module.parameters(), lr=args.learning_rate)
            if training
            else None
        )

        def step() -> object:
            functional.reset_net(module)
            if optimizer is None:
                with torch.no_grad():
                    return module(inputs)
            optimizer.zero_grad(set_to_none=True)
            output = module(inputs)
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            return loss

        return step

    return _protocol.BenchmarkCase(
        name=("spiking-self-attention" if args.case == "attention" else "spikformer"),
        device=device,
        candidate_step=make_step(candidate),
        baseline_step=make_step(baseline),
        input_hash_payload={"input": inputs},
        state_hash_payload=_state_payload(reference),
        workload={
            "family": "spikformer",
            "case": args.case,
            "shape": shape,
            "heads": args.heads,
            "minimum_blocks": args.blocks if args.case == "spikformer" else None,
            "backend": args.backend,
            "mode": args.mode,
            "complete_measured_workload": (
                "reset and forward"
                if not training
                else "reset, zero_grad, forward, loss, backward, SGD update"
            ),
        },
        candidate_metadata=lambda: _protocol.collect_module_route_metadata(candidate),
        baseline_metadata=lambda: _protocol.collect_module_route_metadata(baseline),
    )


ENTRYPOINT = _protocol.Entrypoint(build_case=_build_case, add_arguments=_add_arguments)


if __name__ == "__main__":
    _protocol.benchmark_main("benchmarks.benchmark_spikformer")
