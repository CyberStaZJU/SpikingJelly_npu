
import json
from types import SimpleNamespace

import pytest
import torch

from spikingjelly_npu.activation_based import _aspy, functional, neuron, surrogate
from spikingjelly_npu.routing import ProviderRoute


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
    assert isinstance(aspy_node.last_backend_route, ProviderRoute)
    assert isinstance(aspy_node.last_backend_route, _aspy.AsPyRoute)
    assert aspy_node.last_backend_route.backend == "torch"
    assert aspy_node.last_backend_route.logical_operation == (
        "activation_based.neuron.if.multi_step"
    )
    assert aspy_node.last_backend_route.reason_code == "aspy.if.unsupported_request"
    assert not aspy_node.last_backend_route.native_launch_attempted
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

    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor") as captured:
        node(torch.ones(2, 1, 1))

    assert load_calls == []
    assert isinstance(captured.value.route, ProviderRoute)
    assert captured.value.route.actual_provider is None
    assert captured.value.route.strict
    assert not captured.value.route.native_launch_attempted


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


def test_aspy_route_helper_keeps_historical_display_name_api():
    route = _aspy.native_route("IF")

    assert isinstance(route, ProviderRoute)
    assert isinstance(route, _aspy.AsPyRoute)
    assert route.requested_backend == "aspy"
    assert route.backend == "aspy"
    assert route.reason == "Ascend C fused multi-step IF kernel"
    assert route.accelerated


def test_aspy_route_exact_serialization_is_additive_and_script_compatible():
    route = _aspy.native_route("if", strict=True, training=True)

    expected = {
        "requested_backend": "aspy",
        "backend": "aspy",
        "training": True,
        "requested_provider": "aspy",
        "actual_provider": "aspy",
        "logical_operation": "activation_based.neuron.if.multi_step",
        "reason_code": "aspy.if.native",
        "reason": "Ascend C fused multi-step IF kernel",
        "accelerated": True,
        "strict": True,
        "mode": "train",
        "native_launch_attempted": True,
        "abi_version": None,
        "schema_version": None,
        "bucket": None,
        "native_region": "if",
        "format_conversion": None,
        "route_schema_version": _aspy.ASPY_ROUTE_SCHEMA_VERSION,
    }
    serialized = route.__dict__

    assert serialized == expected
    assert json.loads(json.dumps(serialized)) == expected
    assert serialized.copy() == expected
    assert route.to_dict() == {
        key: value
        for key, value in expected.items()
        if key
        not in {"requested_backend", "backend", "training", "route_schema_version"}
    }

    serialized["reason"] = "local mutation"
    assert route.reason == "Ascend C fused multi-step IF kernel"


def test_aspy_route_fallback_serialization_preserves_historical_values():
    route = _aspy.eager_route(
        "cupy",
        "unsupported request",
        logical_operation="activation_based.neuron.if.multi_step",
        reason_code="aspy.if.unsupported_request",
        training=False,
    )

    assert route.__dict__["requested_backend"] == "cupy"
    assert route.__dict__["backend"] == "torch"
    assert route.__dict__["training"] is False
    assert route.__dict__["requested_provider"] == "cupy"
    assert route.__dict__["actual_provider"] == "torch"
    assert route.__dict__["mode"] == "eval"


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
    assert isinstance(node.last_backend_route, ProviderRoute)
    assert isinstance(node.last_backend_route, _aspy.AsPyRoute)
    assert node.last_backend_route.backend == "aspy"
    assert node.last_backend_route.reason_code == "aspy.if.native"
    assert node.last_backend_route.native_launch_attempted
    assert node.last_backend_route.accelerated


def test_cupy_alias_is_preserved_in_native_route_metadata(monkeypatch):
    node = neuron.IFNode(
        backend="cupy",
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(2, 1, 1)
    spike_seq = torch.ones_like(x)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                if_multi_step=lambda *args: (
                    spike_seq,
                    torch.zeros_like(x[0]),
                    None,
                )
            ),
            None,
        ),
    )

    assert node(x) is spike_seq
    assert node.last_backend_route.requested_provider == "cupy"
    assert node.last_backend_route.actual_provider == "aspy"
    assert node.last_backend_route.native_launch_attempted


def test_declared_partial_bundle_is_not_reinferred_from_adapter_methods(monkeypatch):
    module = SimpleNamespace(
        aspy_abi_version=lambda: 1,
        aspy_capabilities=lambda: {
            "schema_version": 1,
            "capabilities": {"if": ["if_backward", "if_forward"]},
            "symbols": ["if_backward", "if_forward"],
        },
        if_multi_step=lambda *args: (_ for _ in ()).throw(
            AssertionError("partial bundle must not launch")
        ),
    )
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(2, 1, 1)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (module, _aspy._adapter_capabilities(module)),
    )

    output = node(x)

    assert output.shape == x.shape
    assert node.last_backend_route.backend == "torch"
    assert node.last_backend_route.reason_code == "aspy.if.unsupported_bundle"
    assert not node.last_backend_route.native_launch_attempted


def test_aspy_unloadable_bundle_route_is_not_reported_absent(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(2, 1, 1)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (None, OSError("libcust_opapi.so could not be loaded")),
    )

    output = node(x)

    assert output.shape == x.shape
    assert node.last_backend_route.reason_code == "aspy.bundle.load_error"
    assert "unloadable" in node.last_backend_route.reason
    assert "absent" not in node.last_backend_route.reason.lower()


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


@pytest.mark.parametrize("scale_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
def test_klif_single_multi_reset_state_and_first_order_gradients(
    scale_reset, decay_input
):
    kwargs = {
        "scale_reset": scale_reset,
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": False,
        "store_v_seq": True,
    }
    multi = neuron.KLIFNode(step_mode="m", **kwargs)
    single = neuron.KLIFNode(step_mode="s", **kwargs)
    single.load_state_dict(multi.state_dict())
    assert tuple(multi.state_dict()) == ("k",)

    x_multi = torch.tensor(
        [[[0.4, -0.2]], [[0.6, 0.5]], [[0.3, 0.8]]], requires_grad=True
    )
    initial_v = torch.tensor([[0.1, -0.1]], requires_grad=True)
    multi.v = initial_v
    single.v = initial_v.detach().clone().requires_grad_(True)

    y_multi = multi(x_multi)
    x_single = x_multi.detach().clone().requires_grad_(True)
    y_single = torch.stack([single(x_single[t]) for t in range(x_single.shape[0])])
    v_single = single.v

    torch.testing.assert_close(y_multi, y_single)
    torch.testing.assert_close(multi.v, v_single)
    assert multi.v_seq.shape == x_multi.shape
    torch.testing.assert_close(multi.v_seq[-1], multi.v)

    loss_multi = y_multi.sum() + multi.v.sum() + multi.v_seq.sum()
    loss_single = y_single.sum() + v_single.sum()
    loss_multi.backward()
    loss_single.backward()
    assert x_multi.grad is not None and torch.isfinite(x_multi.grad).all()
    assert initial_v.grad is not None and torch.isfinite(initial_v.grad).all()
    assert multi.k.grad is not None and torch.isfinite(multi.k.grad).all()

    old_k = multi.k.detach().clone()
    functional.reset_net(multi)
    torch.testing.assert_close(multi.v, torch.zeros_like(multi.v))
    torch.testing.assert_close(multi.k, old_k)
    multi(torch.zeros(2, 3, 4))
    assert multi.v.shape == (3, 4)


def test_klif_matches_frozen_upstream_equations_and_persistent_state():
    node = neuron.KLIFNode(
        scale_reset=True,
        tau=2.0,
        decay_input=False,
        v_threshold=0.8,
        v_reset=0.0,
        surrogate_function=surrogate.ATan(alpha=2.0),
        step_mode="m",
        store_v_seq=True,
    )
    with torch.no_grad():
        node.k.fill_(1.5)
    x = torch.tensor([[[0.4]], [[0.7]], [[0.2]]], requires_grad=True)

    expected_spikes = []
    expected_v = torch.zeros_like(x[0])
    expected_v_seq = []
    k = node.k
    for current in x:
        charged = expected_v - expected_v / node.tau + current
        fired_voltage = torch.relu(k * charged)
        spike = node.surrogate_function(fired_voltage - node.v_threshold)
        reset_voltage = fired_voltage / k
        expected_v = reset_voltage * (1.0 - spike) + node.v_reset * spike
        expected_spikes.append(spike)
        expected_v_seq.append(expected_v)

    actual = node(x)
    torch.testing.assert_close(actual, torch.stack(expected_spikes))
    torch.testing.assert_close(node.v_seq, torch.stack(expected_v_seq))
    torch.testing.assert_close(node.v, expected_v)

    node.step_mode = "s"
    node(torch.tensor([[0.1]]))
    assert isinstance(node.v, torch.Tensor)


def test_aspy_klif_cpu_fallback_strict_and_single_step(monkeypatch):
    kwargs = {
        "scale_reset": True,
        "tau": 2.5,
        "decay_input": True,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    torch_node = neuron.KLIFNode(backend="torch", **kwargs)
    aspy_node = neuron.KLIFNode(backend="aspy", **kwargs)
    aspy_node.load_state_dict(torch_node.state_dict())
    x_torch = torch.tensor([[[0.6]], [[0.6]], [[0.2]]], requires_grad=True)
    x_aspy = x_torch.detach().clone().requires_grad_(True)

    y_torch = torch_node(x_torch)
    y_aspy = aspy_node(x_aspy)
    (y_torch.sum() + torch_node.v.sum() + torch_node.v_seq.sum()).backward()
    (y_aspy.sum() + aspy_node.v.sum() + aspy_node.v_seq.sum()).backward()
    torch.testing.assert_close(y_aspy, y_torch)
    torch.testing.assert_close(aspy_node.v, torch_node.v)
    torch.testing.assert_close(aspy_node.v_seq, torch_node.v_seq)
    torch.testing.assert_close(x_aspy.grad, x_torch.grad)
    torch.testing.assert_close(aspy_node.k.grad, torch_node.k.grad)
    assert aspy_node.last_backend_route.backend == "torch"
    assert "requires an NPU tensor" in aspy_node.last_backend_route.reason

    load_calls = []
    monkeypatch.setattr(_aspy, "_load_extension", lambda: load_calls.append(True))
    strict = neuron.KLIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor"):
        strict(torch.ones(2, 1, 1))
    assert load_calls == []

    single_fallback = neuron.KLIFNode(
        backend="aspy",
        step_mode="s",
        surrogate_function=surrogate.ATan(),
    )
    output = single_fallback(torch.ones(1, 1))
    assert output.shape == (1, 1)
    assert single_fallback.last_backend_route.backend == "torch"
    assert "single-step" in single_fallback.last_backend_route.reason

    single_strict = neuron.KLIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="s",
        surrogate_function=surrogate.ATan(),
    )
    with pytest.raises(_aspy.AsPyBackendError, match="multi-step only"):
        single_strict(torch.ones(1, 1))
    assert load_calls == []
    assert single_strict.v == 0.0


def test_aspy_klif_old_bundle_falls_back_or_strictly_errors(monkeypatch):
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (SimpleNamespace(lif_multi_step=lambda *args: None), None),
    )
    fallback = neuron.KLIFNode(
        backend="aspy", step_mode="m", surrogate_function=surrogate.ATan()
    )
    output = fallback(torch.zeros(2, 1, 1))
    assert output.shape == (2, 1, 1)
    assert fallback.last_backend_route.backend == "torch"
    assert "does not provide callable klif_multi_step" in fallback.last_backend_route.reason

    strict = neuron.KLIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    with pytest.raises(_aspy.AsPyBackendError, match="klif_multi_step"):
        strict(torch.zeros(2, 1, 1))


def test_aspy_klif_transaction_and_native_failure_propagation(monkeypatch):
    node = neuron.KLIFNode(
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
                klif_multi_step=lambda *args: calls.append(args)
                or (spike_seq, v_final, v_seq)
            ),
            None,
        ),
    )
    output = node(x)
    assert output is spike_seq
    assert node.v is v_final
    assert node.v_seq is v_seq
    assert calls[0][2] is node.k
    assert calls[0][-3:] == (node.tau, node.decay_input, node.scale_reset)
    assert node.last_backend_route.backend == "aspy"
    assert "KLIF" in node.last_backend_route.reason

    failing = neuron.KLIFNode(
        backend="aspy", step_mode="m", surrogate_function=surrogate.ATan()
    )
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                klif_multi_step=lambda *args: (_ for _ in ()).throw(
                    RuntimeError("native KLIF launch failed")
                )
            ),
            None,
        ),
    )
    with pytest.raises(RuntimeError, match="native KLIF launch failed"):
        failing(x)
    torch.testing.assert_close(failing.v, torch.zeros_like(x[0]))

    malformed = neuron.KLIFNode(
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
                klif_multi_step=lambda *args: (
                    torch.ones(1), torch.ones_like(x[0]), torch.ones_like(x)
                )
            ),
            None,
        ),
    )
    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        malformed(x)
    torch.testing.assert_close(malformed.v, torch.zeros_like(x[0]))
    assert malformed.v_seq is None


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
