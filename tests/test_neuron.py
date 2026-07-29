
from types import SimpleNamespace

import pytest
import torch

from spikingjelly_npu.activation_based import _aspy, functional, neuron, surrogate


def test_ifnode_hard_reset_multi_step_and_voltage_sequence():
    node = neuron.IFNode(
        v_threshold=1.0,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        step_mode="m",
        store_v_seq=True,
    )
    x = torch.tensor([[[0.6]], [[0.6]], [[0.2]]])
    spikes = node(x)
    torch.testing.assert_close(spikes, torch.tensor([[[0.0]], [[1.0]], [[0.0]]]))
    torch.testing.assert_close(node.v_seq, torch.tensor([[[0.6]], [[0.0]], [[0.2]]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.2]]))


def test_ifnode_soft_reset_and_reset_net_preserve_no_state_dict_memory():
    node = neuron.IFNode(v_reset=None, step_mode="m")
    spikes = node(torch.tensor([[[0.7]], [[0.7]], [[0.7]]]))
    torch.testing.assert_close(spikes, torch.tensor([[[0.0]], [[1.0]], [[1.0]]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.1]]), rtol=0, atol=1e-6)
    assert node.state_dict() == {}
    functional.reset_net(node)
    torch.testing.assert_close(node.v, torch.zeros_like(node.v))


def test_lifnode_equations_and_persistent_state():
    node = neuron.LIFNode(tau=2.0, decay_input=True, v_threshold=10.0, step_mode="m")
    node(torch.tensor([[[2.0]], [[0.0]]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.5]]))
    node.step_mode = "s"
    node(torch.tensor([[0.0]]))
    torch.testing.assert_close(node.v, torch.tensor([[0.25]]))


def test_parametric_lif_parameter_and_gradient():
    node = neuron.ParametricLIFNode(
        init_tau=4.0,
        decay_input=True,
        v_threshold=10.0,
        step_mode="m",
    )
    assert tuple(node.state_dict()) == ("w",)
    assert torch.isclose(node.w.sigmoid(), torch.tensor(0.25))
    x = torch.ones(3, 2, 4, requires_grad=True)
    node(x).sum().backward()
    assert x.grad is not None
    assert node.w.grad is not None
    assert torch.isfinite(node.w.grad)


def test_detach_reset_changes_gradient_but_not_forward():
    x1 = torch.tensor([[1.2]], requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    attached = neuron.IFNode(detach_reset=False, surrogate_function=surrogate.ATan())
    detached = neuron.IFNode(detach_reset=True, surrogate_function=surrogate.ATan())
    y1 = attached(x1)
    y2 = detached(x2)
    torch.testing.assert_close(y1, y2)
    attached.v.sum().backward()
    detached.v.sum().backward()
    assert not torch.allclose(x1.grad, x2.grad)


def test_shape_change_reinitializes_voltage():
    node = neuron.IFNode(step_mode="s")
    node(torch.ones(2, 3))
    node(torch.zeros(4, 3))
    assert node.v.shape == (4, 3)


def test_cupy_backend_is_an_observable_aspy_preference_alias():
    node = neuron.IFNode(backend="cupy", step_mode="m")
    assert node.requested_backend == "cupy"
    assert node.backend == "aspy"
    output = node(torch.zeros(2, 1, 1))
    assert output.shape == (2, 1, 1)
    assert node.last_backend_route.backend == "torch"
    assert "requires an NPU tensor" in node.last_backend_route.reason


def test_aspy_cpu_fallback_matches_torch_forward_state_and_gradient():
    kwargs = {
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    torch_node = neuron.IFNode(backend="torch", **kwargs)
    aspy_node = neuron.IFNode(backend="aspy", **kwargs)
    x_torch = torch.tensor([[[0.6]], [[0.6]], [[0.2]]], requires_grad=True)
    x_aspy = x_torch.detach().clone().requires_grad_(True)

    y_torch = torch_node(x_torch)
    y_aspy = aspy_node(x_aspy)
    loss_torch = y_torch.square().sum() + torch_node.v.sum() + torch_node.v_seq.sum()
    loss_aspy = y_aspy.square().sum() + aspy_node.v.sum() + aspy_node.v_seq.sum()
    loss_torch.backward()
    loss_aspy.backward()

    torch.testing.assert_close(y_aspy, y_torch)
    torch.testing.assert_close(aspy_node.v, torch_node.v)
    torch.testing.assert_close(aspy_node.v_seq, torch_node.v_seq)
    torch.testing.assert_close(x_aspy.grad, x_torch.grad)
    assert aspy_node.backend == "aspy"
    assert aspy_node.last_backend_route.requested_backend == "aspy"
    assert aspy_node.last_backend_route.backend == "torch"
    assert not aspy_node.last_backend_route.accelerated
    assert "requires an NPU tensor" in aspy_node.last_backend_route.reason


@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_lif_cpu_fallback_matches_torch_forward_state_and_gradient(decay_input):
    kwargs = {
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    torch_node = neuron.LIFNode(backend="torch", **kwargs)
    aspy_node = neuron.LIFNode(backend="aspy", **kwargs)
    x_torch = torch.tensor([[[0.6]], [[0.6]], [[0.2]]], requires_grad=True)
    x_aspy = x_torch.detach().clone().requires_grad_(True)

    y_torch = torch_node(x_torch)
    y_aspy = aspy_node(x_aspy)
    loss_torch = y_torch.square().sum() + torch_node.v.sum() + torch_node.v_seq.sum()
    loss_aspy = y_aspy.square().sum() + aspy_node.v.sum() + aspy_node.v_seq.sum()
    loss_torch.backward()
    loss_aspy.backward()

    torch.testing.assert_close(y_aspy, y_torch)
    torch.testing.assert_close(aspy_node.v, torch_node.v)
    torch.testing.assert_close(aspy_node.v_seq, torch_node.v_seq)
    torch.testing.assert_close(x_aspy.grad, x_torch.grad)
    assert aspy_node.last_backend_route.backend == "torch"
    assert "requires an NPU tensor" in aspy_node.last_backend_route.reason


def test_aspy_lif_strict_mode_rejects_cpu_before_extension_loading(monkeypatch):
    load_calls = []
    monkeypatch.setattr(_aspy, "_load_extension", lambda: load_calls.append(True))
    node = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )

    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor"):
        node(torch.ones(2, 1, 1))

    assert load_calls == []


def test_aspy_strict_mode_rejects_cpu_before_extension_loading(monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: load_calls.append(True),
    )
    node = neuron.IFNode(backend="aspy", backend_strict=True, step_mode="m")

    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor"):
        node(torch.ones(2, 1, 1))

    assert load_calls == []


def test_aspy_single_step_uses_torch_route_without_strict_rejection():
    node = neuron.IFNode(backend="aspy", backend_strict=True, step_mode="s")

    output = node(torch.tensor([[1.1]]))

    torch.testing.assert_close(output, torch.ones_like(output))
    assert node.last_backend_route.backend == "torch"
    assert "single-step" in node.last_backend_route.reason


def test_if_lif_and_parametric_lif_advertise_aspy():
    assert "aspy" in neuron.IFNode().supported_backends
    assert "aspy" in neuron.LIFNode().supported_backends
    assert "aspy" in neuron.ParametricLIFNode().supported_backends


def test_aspy_commits_validated_extension_result(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    spike_seq = torch.ones_like(x)
    v_seq = torch.full_like(x, 0.25)
    v_final = v_seq[-1].clone()
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                if_multi_step=lambda *args: (spike_seq, v_final, v_seq)
            ),
            None,
        ),
    )

    output = node(x)

    assert output is spike_seq
    assert node.v is v_final
    assert node.v_seq is v_seq
    assert node.last_backend_route.backend == "aspy"
    assert node.last_backend_route.accelerated


def test_aspy_lif_commits_validated_extension_result(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
        tau=2.5,
        decay_input=False,
    )
    x = torch.zeros(3, 2, 4)
    spike_seq = torch.ones_like(x)
    v_seq = torch.full_like(x, 0.25)
    v_final = v_seq[-1].clone()
    calls = []
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                lif_multi_step=lambda *args: calls.append(args)
                or (spike_seq, v_final, v_seq)
            ),
            None,
        ),
    )

    output = node(x)

    assert output is spike_seq
    assert node.v is v_final
    assert node.v_seq is v_seq
    assert calls[0][-2:] == (2.5, False)
    assert node.last_backend_route.backend == "aspy"
    assert "LIF" in node.last_backend_route.reason


def test_aspy_malformed_extension_result_does_not_commit_state(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                if_multi_step=lambda *args: (
                    torch.ones(1),
                    torch.ones_like(x[0]),
                    torch.ones_like(x),
                )
            ),
            None,
        ),
    )

    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        node(x)

    torch.testing.assert_close(node.v, torch.zeros_like(x[0]))
    assert node.v_seq is None


def test_aspy_lif_malformed_extension_result_does_not_commit_state(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                lif_multi_step=lambda *args: (
                    torch.ones(1),
                    torch.ones_like(x[0]),
                    torch.ones_like(x),
                )
            ),
            None,
        ),
    )

    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        node(x)

    torch.testing.assert_close(node.v, torch.zeros_like(x[0]))
    assert node.v_seq is None


def test_unsupported_backend_is_rejected():
    with pytest.raises(NotImplementedError):
        neuron.IFNode(backend="triton")


@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_plif_cpu_fallback_matches_torch_and_w_gradient(decay_input):
    kwargs = {
        "init_tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    torch_node = neuron.ParametricLIFNode(backend="torch", **kwargs)
    aspy_node = neuron.ParametricLIFNode(backend="aspy", **kwargs)
    aspy_node.load_state_dict(torch_node.state_dict())
    x_torch = torch.tensor([[[0.6]], [[0.6]], [[0.2]]], requires_grad=True)
    x_aspy = x_torch.detach().clone().requires_grad_(True)

    y_torch = torch_node(x_torch)
    y_aspy = aspy_node(x_aspy)
    loss_torch = y_torch.square().sum() + torch_node.v.sum() + torch_node.v_seq.sum()
    loss_aspy = y_aspy.square().sum() + aspy_node.v.sum() + aspy_node.v_seq.sum()
    loss_torch.backward()
    loss_aspy.backward()

    torch.testing.assert_close(y_aspy, y_torch)
    torch.testing.assert_close(aspy_node.v, torch_node.v)
    torch.testing.assert_close(aspy_node.v_seq, torch_node.v_seq)
    torch.testing.assert_close(x_aspy.grad, x_torch.grad)
    torch.testing.assert_close(aspy_node.w.grad, torch_node.w.grad)
    assert aspy_node.last_backend_route.backend == "torch"
    assert "requires an NPU tensor" in aspy_node.last_backend_route.reason


def test_aspy_plif_strict_cpu_rejection_and_single_step_fallback(monkeypatch):
    load_calls = []
    monkeypatch.setattr(_aspy, "_load_extension", lambda: load_calls.append(True))
    multi = neuron.ParametricLIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor"):
        multi(torch.ones(2, 1, 1))
    assert load_calls == []

    single = neuron.ParametricLIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="s",
        surrogate_function=surrogate.ATan(),
    )
    output = single(torch.ones(1, 1))
    assert output.shape == (1, 1)
    assert single.last_backend_route.backend == "torch"
    assert "single-step" in single.last_backend_route.reason


def test_aspy_plif_commits_only_validated_transactional_state(monkeypatch):
    node = neuron.ParametricLIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    spike_seq = torch.ones_like(x)
    v_seq = torch.full_like(x, 0.25)
    v_final = v_seq[-1].clone()
    calls = []
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                plif_multi_step=lambda *args: calls.append(args)
                or (spike_seq, v_final, v_seq)
            ),
            None,
        ),
    )

    output = node(x)

    assert output is spike_seq
    assert node.v is v_final
    assert node.v_seq is v_seq
    reciprocal_tau = calls[0][2]
    torch.testing.assert_close(reciprocal_tau, node.w.sigmoid())
    assert reciprocal_tau.grad_fn is not None
    assert calls[0][-1] is True
    assert node.last_backend_route.backend == "aspy"
    assert node.last_backend_route.reason == "Ascend C fused multi-step PLIF kernel"

    malformed = neuron.ParametricLIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                plif_multi_step=lambda *args: (
                    torch.ones(1),
                    torch.ones_like(x[0]),
                    torch.ones_like(x),
                )
            ),
            None,
        ),
    )
    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        malformed(x)
    torch.testing.assert_close(malformed.v, torch.zeros_like(x[0]))
    assert malformed.v_seq is None
