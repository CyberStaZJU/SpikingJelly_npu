#!/usr/bin/env python3
"""Audit repeated public KLIF eager and strict-AsPy consumer executions.

Run CPU mode on any supported PyTorch host. Run NPU mode only after activating
an external qualified AsPy build and selecting an idle Ascend device.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from spikingjelly_npu.activation_based import neuron, surrogate
from spikingjelly_npu.npu import configure_npu

EXPECTED_CPU = {
    "output": [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]],
    "v": [[0.3144000172615051, 0.41599997878074646]],
    "x_grad": [
        [[0.23485450446605682, 0.0]],
        [[0.3554198443889618, 0.4537563920021057]],
        [[0.46165841817855835, 0.5249918699264526]],
    ],
    "v0_grad": [[0.35228174924850464, 0.0]],
    "k_grad": -0.07954845577478409,
}


def tensor_data(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)


def run_once(*, backend: str, device: torch.device) -> dict[str, Any]:
    x = torch.tensor(
        [[[0.2, -0.1]], [[0.6, 0.4]], [[0.3, 0.8]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    v0 = torch.tensor(
        [[0.1, -0.2]], dtype=torch.float32, device=device, requires_grad=True
    )
    node = neuron.KLIFNode(
        scale_reset=True,
        tau=2.5,
        decay_input=True,
        v_threshold=0.7,
        v_reset=None,
        surrogate_function=surrogate.ATan(alpha=2.5),
        detach_reset=False,
        step_mode="m",
        backend=backend,
        backend_strict=backend == "aspy",
        store_v_seq=True,
    ).to(device)
    with torch.no_grad():
        node.k.fill_(1.25)
    node.v = v0

    output = node(x)
    output_weight = torch.tensor(
        [[[0.2, 0.3]], [[0.4, 0.5]], [[0.6, 0.7]]],
        dtype=torch.float32,
        device=device,
    )
    voltage_weight = torch.tensor(
        [[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.6]]],
        dtype=torch.float32,
        device=device,
    )
    final_weight = torch.tensor([[0.7, 0.8]], dtype=torch.float32, device=device)
    loss = (
        (output * output_weight).sum()
        + (node.v_seq * voltage_weight).sum()
        + (node.v * final_weight).sum()
    )
    loss.backward()
    synchronize(device)

    route = node.last_backend_route
    return {
        "output": tensor_data(output),
        "v": tensor_data(node.v),
        "x_grad": tensor_data(x.grad),
        "v0_grad": tensor_data(v0.grad),
        "k_grad": float(node.k.grad.detach().cpu()),
        "route": route.backend,
        "accelerated": route.accelerated,
        "reason": route.reason,
    }


def assert_cpu_result(result: dict[str, Any]) -> None:
    assert result["route"] == "torch"
    assert result["accelerated"] is False
    for key in ("output", "v", "x_grad", "v0_grad"):
        torch.testing.assert_close(
            torch.tensor(result[key]), torch.tensor(EXPECTED_CPU[key]), rtol=0, atol=0
        )
    assert result["k_grad"] == EXPECTED_CPU["k_grad"]


def assert_npu_result(result: dict[str, Any]) -> None:
    assert result["route"] == "aspy"
    assert result["accelerated"] is True
    assert result["reason"] == "Ascend C fused multi-step KLIF kernel"
    for key in ("output", "v", "x_grad", "v0_grad"):
        torch.testing.assert_close(
            torch.tensor(result[key]),
            torch.tensor(EXPECTED_CPU[key]),
            rtol=5e-5,
            atol=3e-5,
        )
    torch.testing.assert_close(
        torch.tensor(result["k_grad"]),
        torch.tensor(EXPECTED_CPU["k_grad"]),
        rtol=5e-5,
        atol=3e-5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "npu"), required=True)
    parser.add_argument("--device", default="npu:7")
    args = parser.parse_args()

    if args.mode == "cpu":
        backend = "torch"
        device = torch.device("cpu")
        assertion = assert_cpu_result
        label = "cpu-eager"
    else:
        backend = "aspy"
        device = configure_npu(args.device, jit_compile=False, allow_internal_format=False)
        assertion = assert_npu_result
        label = "npu-strict-aspy"

    results = []
    for repeat in (1, 2):
        result = run_once(backend=backend, device=device)
        assertion(result)
        results.append(result)
        print(
            json.dumps(
                {"label": f"{label}-{repeat}", **result},
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    assert results[0] == results[1]
    print(json.dumps({"label": label, "status": "PASS", "repeats": 2}, sort_keys=True))


if __name__ == "__main__":
    main()
