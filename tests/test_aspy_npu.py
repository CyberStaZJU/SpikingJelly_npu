import os

import pytest
import torch

from spikingjelly_npu.activation_based import functional, neuron, surrogate
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu, is_npu_available

pytestmark = [pytest.mark.npu, pytest.mark.aspy]


def require_native_aspy():
    if os.environ.get("SPIKINGJELLY_NPU_ASPY_EXPECT_NATIVE") != "1":
        pytest.skip("optional AsPy native extension was not requested")
    if not is_npu_available():
        pytest.skip("Ascend NPU unavailable")
    index = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    return configure_npu(f"npu:{index}")


@pytest.mark.parametrize("shape", [(3, 2, 4), (5, 3, 5), (4, 1, 4097), (4, 2, 4096)])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_if_forward_exact_parity(shape, v_reset):
    device = require_native_aspy()
    torch.manual_seed(20260729)
    x_seq = torch.rand(shape, dtype=torch.float32, device=device)
    kwargs = {
        "v_threshold": 1.0,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_seq)
    actual = accelerated(x_seq)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=0, atol=0)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.accelerated


@pytest.mark.parametrize("shape", [(3, 2, 4), (5, 2, 4097)])
@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("store_v_seq", [False, True])
def test_aspy_if_backward_matches_eager(
    shape,
    v_reset,
    detach_reset,
    store_v_seq,
):
    device = require_native_aspy()
    torch.manual_seed(20260730)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": store_v_seq,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25,
        1.25,
        expected.numel(),
        dtype=expected.dtype,
        device=device,
    ).reshape_as(expected)
    final_weight = torch.linspace(
        0.5,
        1.5,
        reference.v.numel(),
        dtype=reference.v.dtype,
        device=device,
    ).reshape_as(reference.v)
    reference_loss = (expected * spike_weight).sum() + (reference.v * final_weight).sum()
    accelerated_loss = (actual * spike_weight).sum() + (
        accelerated.v * final_weight
    ).sum()
    if store_v_seq:
        voltage_weight = torch.linspace(
            0.1,
            0.9,
            reference.v_seq.numel(),
            dtype=reference.v_seq.dtype,
            device=device,
        ).reshape_as(reference.v_seq)
        reference_loss = reference_loss + (reference.v_seq * voltage_weight).sum()
        accelerated_loss = accelerated_loss + (
            accelerated.v_seq * voltage_weight
        ).sum()

    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=0, atol=0)
    if store_v_seq:
        torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=0, atol=0)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=2e-5, atol=2e-6)
    assert accelerated.last_backend_route.backend == "aspy"


@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_stateful_sequence_gradient_chains_through_initial_voltage(v_reset):
    device = require_native_aspy()
    torch.manual_seed(20260731)
    first_reference = torch.rand(3, 2, 8, device=device, requires_grad=True)
    second_reference = torch.rand(2, 2, 8, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    reference(first_reference)
    accelerated(first_accelerated)
    reference_output = reference(second_reference)
    accelerated_output = accelerated(second_accelerated)
    reference_loss = reference_output.square().sum() + reference.v.square().sum()
    accelerated_loss = accelerated_output.square().sum() + accelerated.v.square().sum()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(
        first_accelerated.grad,
        first_reference.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        second_accelerated.grad,
        second_reference.grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_aspy_if_forward_without_voltage_sequence():
    device = require_native_aspy()
    x_seq = torch.full((3, 2, 4), 0.6, dtype=torch.float32, device=device)
    node = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        store_v_seq=False,
        surrogate_function=surrogate.ATan(),
    ).to(device)

    output = node(x_seq)
    torch.npu.synchronize(device)

    assert output.shape == x_seq.shape
    assert not hasattr(node, "v_seq")
    assert node.last_backend_route.backend == "aspy"


class _BatchFirstAsPy(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.IFNode(
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_npugraph_capture_matches_native_eager_forward_and_gradient():
    device = require_native_aspy()
    eager_model = _BatchFirstAsPy().to(device).eval()
    graph_model = _BatchFirstAsPy().to(device).eval()
    runner = StaticGraphRunner(graph_model, batch_size=4, strict=True)
    inputs = torch.rand(4, 3, 8, dtype=torch.float32, device=device)
    eager_inputs = inputs.detach().clone().requires_grad_(True)
    graph_inputs = inputs.detach().clone().requires_grad_(True)

    eager_output = eager_model(eager_inputs)
    eager_output.square().sum().backward()
    graph_output = runner(graph_inputs)
    graph_output.square().sum().backward()
    torch.npu.synchronize(device)

    assert graph_output.shape == (3, 4, 8)
    assert runner.last_route.backend == "npugraph"
    assert runner.last_route.captured
    assert runner.capture_error is None
    assert graph_model.node.last_backend_route.backend == "aspy"
    torch.testing.assert_close(graph_output, eager_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(graph_inputs.grad, eager_inputs.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(4, 2, 5), (3, 1, 4097)])
@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("store_v_seq", [False, True])
def test_aspy_lif_forward_backward_matches_eager(
    shape,
    v_reset,
    detach_reset,
    decay_input,
    store_v_seq,
):
    device = require_native_aspy()
    torch.manual_seed(20260801)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": store_v_seq,
    }
    reference = neuron.LIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    final_weight = torch.linspace(
        0.5, 1.5, reference.v.numel(), dtype=reference.v.dtype, device=device
    ).reshape_as(reference.v)
    reference_loss = (expected * spike_weight).sum() + (reference.v * final_weight).sum()
    accelerated_loss = (actual * spike_weight).sum() + (
        accelerated.v * final_weight
    ).sum()
    if store_v_seq:
        voltage_weight = torch.linspace(
            0.1,
            0.9,
            reference.v_seq.numel(),
            dtype=reference.v_seq.dtype,
            device=device,
        ).reshape_as(reference.v_seq)
        reference_loss = reference_loss + (reference.v_seq * voltage_weight).sum()
        accelerated_loss = accelerated_loss + (
            accelerated.v_seq * voltage_weight
        ).sum()

    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=1e-6, atol=1e-7)
    if store_v_seq:
        torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=3e-5, atol=3e-6)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.accelerated
    assert "LIF" in accelerated.last_backend_route.reason


@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_lif_stateful_sequence_gradient_chains_through_initial_voltage(
    decay_input,
    v_reset,
):
    device = require_native_aspy()
    torch.manual_seed(20260802)
    first_reference = torch.rand(3, 2, 9, device=device, requires_grad=True)
    second_reference = torch.rand(2, 2, 9, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "tau": 3.0,
        "decay_input": decay_input,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.LIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    reference(first_reference)
    accelerated(first_accelerated)
    reference_output = reference(second_reference)
    accelerated_output = accelerated(second_accelerated)
    reference_loss = reference_output.square().sum() + reference.v.square().sum()
    accelerated_loss = accelerated_output.square().sum() + accelerated.v.square().sum()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(first_accelerated.grad, first_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        second_accelerated.grad,
        second_reference.grad,
        rtol=3e-5,
        atol=3e-6,
    )
    assert accelerated.last_backend_route.backend == "aspy"


class _BatchFirstAsPyLIF(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.LIFNode(
            tau=2.5,
            decay_input=True,
            v_reset=None,
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(alpha=2.0),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_lif_npugraph_capture_and_five_variable_replays():
    device = require_native_aspy()
    eager_model = _BatchFirstAsPyLIF().to(device).eval()
    graph_model = _BatchFirstAsPyLIF().to(device).eval()
    runner = StaticGraphRunner(graph_model, batch_size=4, strict=True)

    for replay in range(5):
        torch.manual_seed(20260810 + replay)
        inputs = torch.rand(4, 3, 17, dtype=torch.float32, device=device)
        eager_inputs = inputs.detach().clone().requires_grad_(True)
        graph_inputs = inputs.detach().clone().requires_grad_(True)

        eager_output = eager_model(eager_inputs)
        eager_output.square().sum().backward()
        graph_output = runner(graph_inputs)
        graph_output.square().sum().backward()
        torch.npu.synchronize(device)

        torch.testing.assert_close(graph_output, eager_output, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            graph_inputs.grad,
            eager_inputs.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        assert runner.last_route.backend == "npugraph"
        assert runner.last_route.captured
        assert runner.capture_error is None
        assert graph_model.node.last_backend_route.backend == "aspy"


@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_plif_eight_combinations_forward_backward_and_w_grad(
    v_reset, detach_reset, decay_input
):
    device = require_native_aspy()
    torch.manual_seed(20260820)
    shape = (5, 3, 7)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "init_tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.ParametricLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.ParametricLIFNode(
        backend="aspy", backend_strict=True, **kwargs
    ).to(device)
    accelerated.load_state_dict(reference.state_dict())

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    voltage_weight = torch.linspace(
        0.1, 0.9, reference.v_seq.numel(), dtype=reference.v_seq.dtype, device=device
    ).reshape_as(reference.v_seq)
    final_weight = torch.linspace(
        0.5, 1.5, reference.v.numel(), dtype=reference.v.dtype, device=device
    ).reshape_as(reference.v)
    reference_loss = (
        (expected * spike_weight).sum()
        + (reference.v_seq * voltage_weight).sum()
        + (reference.v * final_weight).sum()
    )
    accelerated_loss = (
        (actual * spike_weight).sum()
        + (accelerated.v_seq * voltage_weight).sum()
        + (accelerated.v * final_weight).sum()
    )
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=3e-5, atol=3e-6)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.reason == "Ascend C fused multi-step PLIF kernel"


@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_plif_cross_tile_state_chain_and_gradients(decay_input):
    device = require_native_aspy()
    torch.manual_seed(20260821)
    first_reference = torch.rand(3, 1, 4097, device=device, requires_grad=True)
    second_reference = torch.rand(2, 1, 4097, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "init_tau": 3.0,
        "decay_input": decay_input,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.ParametricLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.ParametricLIFNode(
        backend="aspy", backend_strict=True, **kwargs
    ).to(device)
    accelerated.load_state_dict(reference.state_dict())

    reference(first_reference)
    accelerated(first_accelerated)
    expected = reference(second_reference)
    actual = accelerated(second_accelerated)
    reference_loss = expected.square().sum() + reference.v.square().mean()
    accelerated_loss = actual.square().sum() + accelerated.v.square().mean()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(first_accelerated.grad, first_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(second_accelerated.grad, second_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=3e-5, atol=3e-6)


class _BatchFirstAsPyPLIF(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.ParametricLIFNode(
            init_tau=2.5,
            decay_input=True,
            v_threshold=0.35,
            v_reset=None,
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(alpha=2.0),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_plif_npugraph_dynamic_w_five_replays_forward_backward():
    device = require_native_aspy()
    torch.use_deterministic_algorithms(True, warn_only=False)
    eager_model = _BatchFirstAsPyPLIF().to(device).train()
    graph_model = _BatchFirstAsPyPLIF().to(device).train()
    graph_model.load_state_dict(eager_model.state_dict())
    runner = StaticGraphRunner(
        graph_model,
        batch_size=4,
        strict=True,
        allow_training=True,
        assume_graph_safe=True,
    )
    previous_output = None

    for replay in range(5):
        with torch.no_grad():
            new_w = torch.tensor(-0.8 + replay * 0.3, dtype=torch.float32, device=device)
            eager_model.node.w.copy_(new_w)
            graph_model.node.w.copy_(new_w)
        torch.manual_seed(20260830 + replay)
        inputs = torch.rand(4, 3, 17, dtype=torch.float32, device=device)
        eager_inputs = inputs.detach().clone().requires_grad_(True)
        graph_inputs = inputs.detach().clone().requires_grad_(True)
        eager_model.node.w.grad = None
        graph_model.node.w.grad = None

        eager_output = eager_model(eager_inputs)
        eager_loss = eager_output.square().sum()
        eager_loss.backward()
        graph_output = runner(graph_inputs)
        graph_loss = graph_output.square().sum()
        graph_loss.backward()
        torch.npu.synchronize(device)

        torch.testing.assert_close(graph_output, eager_output, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(graph_inputs.grad, eager_inputs.grad, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            graph_model.node.w.grad,
            eager_model.node.w.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        if previous_output is not None:
            assert not torch.equal(graph_output, previous_output)
        previous_output = graph_output.detach().clone()
        assert runner.last_route.backend == "npugraph"
        assert runner.last_route.captured
        assert graph_model.node.last_backend_route.backend == "aspy"
