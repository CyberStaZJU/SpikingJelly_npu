"""Representative standard and spiking recurrent benchmark entrypoint."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn

from benchmarks import _protocol
from spikingjelly_npu import sequence
from spikingjelly_npu.activation_based import recurrent as spiking_recurrent

_STANDARD_MODULES = {
    "rnn": sequence.RNN,
    "gru": sequence.GRU,
    "lstm": sequence.LSTM,
}
_SPIKING_MODULES = {
    "spiking-rnn": spiking_recurrent.SpikingRNN,
    "spiking-gru": spiking_recurrent.SpikingGRU,
    "spiking-lstm": spiking_recurrent.SpikingLSTM,
}


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case",
        choices=tuple(_STANDARD_MODULES) + tuple(_SPIKING_MODULES),
        default="gru",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        help="defaults to 128 for standard recurrent cases and 64 for spiking cases",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "input_size",
        "hidden_size",
        "num_layers",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _protocol.BenchmarkProtocolError(f"{name} must be a positive integer")
    if args.learning_rate <= 0.0:
        raise _protocol.BenchmarkProtocolError("learning_rate must be positive")


def _clone_state(state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
    if isinstance(state, tuple):
        return tuple(value.detach().clone() for value in state)
    return state.detach().clone()


def _state_payload(module: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _build_case(args: argparse.Namespace, device: torch.device) -> _protocol.BenchmarkCase:
    _validate_args(args)
    is_spiking = args.case in _SPIKING_MODULES
    time_steps = (64 if is_spiking else 128) if args.time_steps is None else args.time_steps
    if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps <= 0:
        raise _protocol.BenchmarkProtocolError("time_steps must be a positive integer")
    module_class = (_SPIKING_MODULES if is_spiking else _STANDARD_MODULES)[args.case]
    module_kwargs = {
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": 0.0,
        "batch_first": False,
    }

    torch.manual_seed(args.seed)
    reference = module_class(**module_kwargs).to(device)
    initial_state = reference.state_dict()
    torch.manual_seed(args.seed + 1)
    candidate = module_class(**module_kwargs).to(device)
    torch.manual_seed(args.seed + 2)
    baseline = module_class(**module_kwargs).to(device)
    candidate.load_state_dict(initial_state)
    baseline.load_state_dict(initial_state)
    training = args.mode == "train"
    candidate.train(training)
    baseline.train(training)

    inputs = torch.randn(
        time_steps,
        args.batch_size,
        args.input_size,
        device=device,
        dtype=torch.float32,
    )
    state_shape = (args.num_layers, args.batch_size, args.hidden_size)
    if args.case.endswith("lstm"):
        initial_carry = (
            torch.randn(*state_shape, device=device),
            torch.randn(*state_shape, device=device),
        )
    else:
        initial_carry = torch.randn(*state_shape, device=device)

    def make_step(module: nn.Module):
        optimizer = (
            torch.optim.SGD(module.parameters(), lr=args.learning_rate)
            if training
            else None
        )

        def step() -> object:
            carry = _clone_state(initial_carry)
            if optimizer is None:
                with torch.no_grad():
                    output, next_state = module(inputs, carry)
                    return output, next_state
            optimizer.zero_grad(set_to_none=True)
            output, next_state = module(inputs, carry)
            state_loss = (
                sum(value.square().mean() for value in next_state)
                if isinstance(next_state, tuple)
                else next_state.square().mean()
            )
            loss = output.square().mean() + state_loss
            loss.backward()
            optimizer.step()
            return loss

        return step

    provider_route = {
        "requested_provider": "torch",
        "actual_provider": "torch",
        "logical_operation": (
            "activation_based.spiking_recurrent" if is_spiking else "sequence.recurrent"
        ),
        "reason_code": "torch.reference",
        "reason": "semantic-alpha recurrent wrappers currently use eager PyTorch",
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

    return _protocol.BenchmarkCase(
        name=args.case,
        device=device,
        candidate_step=make_step(candidate),
        baseline_step=make_step(baseline),
        input_hash_payload={"input": inputs, "initial_state": initial_carry},
        state_hash_payload=_state_payload(reference),
        workload={
            "family": "spiking_recurrent" if is_spiking else "standard_recurrent",
            "case": args.case,
            "shape": {
                "T": time_steps,
                "B": args.batch_size,
                "I": args.input_size,
                "H": args.hidden_size,
            },
            "num_layers": args.num_layers,
            "layout": "TBF",
            "mode": args.mode,
            "complete_measured_workload": (
                "forward, output and final state"
                if not training
                else "zero_grad, forward, output/state loss, backward, SGD update"
            ),
        },
        candidate_metadata=metadata,
        baseline_metadata=metadata,
    )


ENTRYPOINT = _protocol.Entrypoint(build_case=_build_case, add_arguments=_add_arguments)


if __name__ == "__main__":
    _protocol.benchmark_main("benchmarks.benchmark_sequence_recurrent")
