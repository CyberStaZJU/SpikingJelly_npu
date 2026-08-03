"""Representative standard Transformer encoder/decoder benchmark entrypoint."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from benchmarks import _protocol
from spikingjelly_npu import sequence


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", choices=("encoder", "decoder"), default="encoder")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--target-length", type=int, default=64)
    parser.add_argument("--source-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _protocol.BenchmarkProtocolError(f"{name} must be a positive integer")
    return value


def _state_payload(module: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _build_encoder(args: argparse.Namespace, device: torch.device):
    batch_size = 32 if args.batch_size is None else args.batch_size
    sequence_length = 128 if args.sequence_length is None else args.sequence_length
    layer = sequence.TransformerEncoderLayer(
        d_model=args.d_model,
        nhead=args.heads,
        dim_feedforward=args.ffn,
        dropout=0.0,
        batch_first=True,
    )
    module = sequence.TransformerEncoder(
        layer,
        args.layers,
        enable_nested_tensor=False,
    ).to(device)
    inputs = torch.randn(batch_size, sequence_length, args.d_model, device=device)
    padding_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
    if sequence_length > 1:
        padding_mask[:, -1] = True
    kwargs = {"src_key_padding_mask": padding_mask}
    shape = {"B": batch_size, "L": sequence_length, "D": args.d_model}
    return module, (inputs,), kwargs, shape


def _build_decoder(args: argparse.Namespace, device: torch.device):
    batch_size = 16 if args.batch_size is None else args.batch_size
    target_length = _positive("target_length", args.target_length)
    source_length = _positive("source_length", args.source_length)
    layer = sequence.TransformerDecoderLayer(
        d_model=args.d_model,
        nhead=args.heads,
        dim_feedforward=args.ffn,
        dropout=0.0,
        batch_first=True,
    )
    module = sequence.TransformerDecoder(layer, args.layers).to(device)
    target = torch.randn(batch_size, target_length, args.d_model, device=device)
    memory = torch.randn(batch_size, source_length, args.d_model, device=device)
    target_mask = torch.nn.Transformer.generate_square_subsequent_mask(
        target_length,
        device=device,
    )
    target_padding_mask = torch.zeros(
        batch_size, target_length, dtype=torch.bool, device=device
    )
    memory_padding_mask = torch.zeros(
        batch_size, source_length, dtype=torch.bool, device=device
    )
    if target_length > 1:
        target_padding_mask[:, -1] = True
    if source_length > 1:
        memory_padding_mask[:, -1] = True
    kwargs = {
        "tgt_mask": target_mask,
        "tgt_key_padding_mask": target_padding_mask,
        "memory_key_padding_mask": memory_padding_mask,
        "tgt_is_causal": True,
        "memory_is_causal": False,
    }
    shape = {
        "B": batch_size,
        "target_length": target_length,
        "source_length": source_length,
        "D": args.d_model,
    }
    return module, (target, memory), kwargs, shape


def _build_case(args: argparse.Namespace, device: torch.device) -> _protocol.BenchmarkCase:
    for name in ("d_model", "heads", "ffn", "layers"):
        _positive(name, getattr(args, name))
    if args.d_model % args.heads != 0:
        raise _protocol.BenchmarkProtocolError("d_model must be divisible by heads")
    if args.batch_size is not None:
        _positive("batch_size", args.batch_size)
    if args.sequence_length is not None:
        _positive("sequence_length", args.sequence_length)
    if args.learning_rate <= 0.0:
        raise _protocol.BenchmarkProtocolError("learning_rate must be positive")

    torch.manual_seed(args.seed)
    builder = _build_encoder if args.case == "encoder" else _build_decoder
    reference, inputs, kwargs, shape = builder(args, device)
    initial_state = reference.state_dict()
    torch.manual_seed(args.seed + 1)
    candidate, _, _, _ = builder(args, device)
    torch.manual_seed(args.seed + 2)
    baseline, _, _, _ = builder(args, device)
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
            if optimizer is None:
                with torch.no_grad():
                    return module(*inputs, **kwargs)
            optimizer.zero_grad(set_to_none=True)
            output = module(*inputs, **kwargs)
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            return loss

        return step

    provider_route = {
        "requested_provider": "torch",
        "actual_provider": "torch",
        "logical_operation": f"sequence.transformer.{args.case}",
        "reason_code": "torch.reference",
        "reason": "semantic-alpha Transformer wrappers currently use eager PyTorch",
        "accelerated": False,
        "strict": False,
        "mode": args.mode,
        "native_launch_attempted": False,
        "abi_version": None,
        "schema_version": None,
        "bucket": None,
        "native_region": None,
        "format_conversion": None,
    }

    def metadata() -> dict[str, object]:
        return {
            "provider_routes": [provider_route],
            "native_region_count": 0,
            "native_region_counts": {},
            "format_conversion_count": 0,
            "format_conversion_counts": {},
            "format_conversion_bytes": 0,
        }

    input_payload = {
        "inputs": inputs,
        "kwargs": kwargs,
    }
    return _protocol.BenchmarkCase(
        name=f"transformer-{args.case}",
        device=device,
        candidate_step=make_step(candidate),
        baseline_step=make_step(baseline),
        input_hash_payload=input_payload,
        state_hash_payload=_state_payload(reference),
        workload={
            "family": "standard_transformer",
            "case": args.case,
            "shape": shape,
            "heads": args.heads,
            "ffn": args.ffn,
            "layers": args.layers,
            "layout": "batch_first",
            "mode": args.mode,
            "complete_measured_workload": (
                "teacher-forced forward"
                if not training
                else "zero_grad, teacher-forced forward, loss, backward, SGD update"
            ),
        },
        candidate_metadata=metadata,
        baseline_metadata=metadata,
    )


ENTRYPOINT = _protocol.Entrypoint(build_case=_build_case, add_arguments=_add_arguments)


if __name__ == "__main__":
    _protocol.benchmark_main("benchmarks.benchmark_sequence_transformer")
