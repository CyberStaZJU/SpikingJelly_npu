import os

import pytest
import torch

from spikingjelly_npu.activation_based import functional, layer, neuron, surrogate
from spikingjelly_npu.fedsnn import PackedBNTTMLP
from spikingjelly_npu.npu import (
    GraphPreExecutionError,
    StaticGraphRunner,
    configure_npu,
    is_npu_available,
)

pytestmark = pytest.mark.npu


def require_npu():
    if not is_npu_available():
        pytest.skip("Ascend NPU unavailable")
    index = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    return configure_npu(f"npu:{index}")


def test_activation_subset_forward_backward_on_npu():
    device = require_npu()
    model = torch.nn.Sequential(
        layer.Linear(16, 32, step_mode="m"),
        neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="m",
        ),
        layer.Linear(32, 5, step_mode="m"),
    ).to(device)
    inputs = torch.rand(4, 8, 16, device=device, requires_grad=True)
    outputs = model(inputs).mean(0)
    outputs.square().mean().backward()
    assert outputs.shape == (8, 5)
    assert inputs.grad is not None
    functional.reset_net(model)


class BatchFirst(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, batch):
        return self.inner(batch.transpose(0, 1).contiguous())


def test_fixed_shape_npugraph_forward_backward():
    device = require_npu()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        model = PackedBNTTMLP(16, 32, 5, 4).to(device).train()
        # StaticGraphRunner buckets dimension 0. This canary uses [N,T,F]
        # externally and transposes inside a wrapper so the graph key is the
        # actual batch size.
        wrapped = BatchFirst(model)
        runner = StaticGraphRunner(
            wrapped,
            batch_size=8,
            strict=True,
            allow_training=True,
        )
        inputs = torch.rand(8, 4, 16, device=device)
        outputs = runner(inputs)
        outputs.square().mean().backward()
        assert outputs.shape == (8, 5)
        assert runner.last_route.backend == "npugraph"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def test_training_npugraph_is_disabled_by_default():
    device = require_npu()
    model = PackedBNTTMLP(16, 32, 5, 4).to(device).train()
    runner = StaticGraphRunner(BatchFirst(model), batch_size=8, strict=True)
    with pytest.raises(GraphPreExecutionError, match="allow_training=True"):
        runner(torch.rand(8, 4, 16, device=device))
    assert runner.last_route.backend == "eager"


def test_training_npugraph_requires_deterministic_algorithms():
    device = require_npu()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False, warn_only=False)
    try:
        model = PackedBNTTMLP(16, 32, 5, 4).to(device).train()
        runner = StaticGraphRunner(
            BatchFirst(model),
            batch_size=8,
            strict=True,
            allow_training=True,
        )
        with pytest.raises(
            GraphPreExecutionError,
            match="use_deterministic_algorithms\\(True, warn_only=False\\)",
        ):
            runner(torch.rand(8, 4, 16, device=device))
        assert runner.last_route.backend == "eager"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def test_partial_batch_strict_rejects_before_eager_npu():
    device = require_npu()
    model = torch.nn.Linear(4, 2).to(device)
    runner = StaticGraphRunner(
        model, batch_size=8, strict=True, assume_graph_safe=True
    )
    with pytest.raises(GraphPreExecutionError, match="batch shape"):
        runner(torch.rand(3, 4, device=device))
    assert runner.last_route.backend == "eager"
