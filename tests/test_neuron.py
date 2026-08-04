import json
from types import SimpleNamespace

import pytest
import torch

from spikingjelly_npu.activation_based import _aspy, functional, neuron, surrogate
from spikingjelly_npu.npu.amp import npu_bf16_autocast
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


def test_torch_lif_keeps_fp32_state_and_public_output_in_bf16_profile(monkeypatch):
    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: FakeAutocast())
    node = neuron.LIFNode(
        tau=2.0,
        decay_input=True,
        v_threshold=10.0,
        step_mode="m",
        store_v_seq=True,
    )
    x = torch.tensor([[[2.0]], [[0.0]]], dtype=torch.bfloat16)

    with npu_bf16_autocast():
        output = node(x)

    assert output.dtype == torch.bfloat16
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32
    torch.testing.assert_close(node.v, torch.tensor([[0.5]]))


def test_lifnode_bf16_profile_resets_with_fp32_internal_spike(monkeypatch):
    original_autocast = torch.autocast

    class CPUAutocast:
        def __init__(self, **kwargs):
            if not kwargs.get("enabled", True):
                self.context = original_autocast(device_type="cpu", enabled=False)
            else:
                self.context = original_autocast(
                    device_type="cpu", dtype=torch.bfloat16, cache_enabled=False
                )

        def __enter__(self):
            return self.context.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: CPUAutocast(**kwargs))
    node = neuron.LIFNode(
        tau=2.0,
        decay_input=False,
        v_threshold=1.0,
        step_mode="s",
        surrogate_function=surrogate.ATan(),
    )
    reset_dtypes = []
    original_reset = node.neuronal_reset

    def inspect_reset(spike):
        reset_dtypes.append(spike.dtype)
        return original_reset(spike)

    monkeypatch.setattr(node, "neuronal_reset", inspect_reset)
    x = torch.full((2, 3), 1.25, dtype=torch.bfloat16, requires_grad=True)

    with npu_bf16_autocast():
        output = node(x)
        loss = output.float().sum() + node.v.sum()
    loss.backward()

    assert output.dtype == torch.bfloat16
    assert reset_dtypes == [torch.float32]
    assert node.v.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16


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


@pytest.mark.parametrize(
    ("threshold", "reset", "alpha", "match"),
    [
        (float("nan"), 0.0, 2.0, "finite v_threshold"),
        (1.0, float("inf"), 2.0, "finite v_reset"),
        (1.0, 0.0, 0.0, "finite positive ATan alpha"),
        (1.0, 0.0, float("nan"), "finite positive ATan alpha"),
    ],
)
def test_aspy_common_scalar_validation_is_explicit(threshold, reset, alpha, match):
    reason = _aspy._unsupported_scalar_reason(
        surrogate.ATan(alpha=alpha),
        v_threshold=threshold,
        v_reset=reset,
    )

    assert reason is not None
    assert match in reason


@pytest.mark.parametrize(
    ("kind", "value", "match"),
    [
        ("lif", float("inf"), "finite fixed float tau greater than 1"),
        ("klif", float("nan"), "finite fixed float tau greater than 1"),
    ],
)
def test_aspy_invalid_tau_rejects_before_extension_loading(monkeypatch, kind, value, match):
    load_calls = []
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_load_extension", lambda: load_calls.append(True))
    common = {
        "x_seq": torch.zeros(2, 1, 1),
        "v_init": torch.zeros(1, 1),
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "detach_reset": False,
        "surrogate_function": surrogate.ATan(),
        "store_v_seq": False,
        "tau": value,
        "decay_input": True,
        "strict": True,
    }

    with pytest.raises(_aspy.AsPyBackendError, match=match):
        if kind == "lif":
            _aspy.try_lif_multi_step(**common)
        else:
            _aspy.try_klif_multi_step(
                **common,
                k=torch.ones(1),
                scale_reset=False,
            )

    assert load_calls == []


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
        "dtype_conversion": None,
        "dtype_conversion_bytes": None,
        "route_schema_version": _aspy.ASPY_ROUTE_SCHEMA_VERSION,
    }
    serialized = route.__dict__

    assert serialized == expected
    assert json.loads(json.dumps(serialized)) == expected
    assert serialized.copy() == expected
    assert route.to_dict() == {
        key: value
        for key, value in expected.items()
        if key not in {"requested_backend", "backend", "training", "route_schema_version"}
    }

    serialized["reason"] = "local mutation"
    assert route.reason == "Ascend C fused multi-step IF kernel"


def test_aspy_bf16_if_uses_fp32_native_island_and_public_dtype(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    calls = []

    def implementation(x_seq, v_init, *args):
        calls.append((x_seq, v_init, args))
        assert x_seq.dtype == torch.float32
        assert v_init.dtype == torch.float32
        return (
            x_seq * 0.0 + 1.0,
            v_init * 0.0 + 0.25,
            x_seq * 0.0 + 0.25,
        )

    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (SimpleNamespace(if_multi_step=implementation), None),
    )

    output = node(x)
    (output.float().sum() + node.v.sum() + node.v_seq.sum()).backward()

    assert len(calls) == 1
    assert output.dtype == torch.bfloat16
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    assert node.last_backend_route.backend == "aspy"
    assert node.last_backend_route.dtype_conversion == "bf16-public-fp32-aspy-island"
    expected_bytes = 2 * x.numel() * (2 + 4)
    assert node.last_backend_route.dtype_conversion_bytes == expected_bytes


def test_aspy_bf16_plif_and_klif_keep_master_scalars_fp32(monkeypatch):
    cases = (
        (
            neuron.ParametricLIFNode(
                backend="aspy",
                backend_strict=True,
                step_mode="m",
                store_v_seq=True,
                surrogate_function=surrogate.ATan(),
            ),
            "plif_multi_step",
            "w",
        ),
        (
            neuron.KLIFNode(
                backend="aspy",
                backend_strict=True,
                step_mode="m",
                store_v_seq=True,
                surrogate_function=surrogate.ATan(),
            ),
            "klif_multi_step",
            "k",
        ),
    )
    for node, symbol, parameter_name in cases:
        x = torch.zeros(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
        seen = []

        def implementation(x_seq, v_init, scalar, *args, seen=seen):
            seen.append((x_seq.dtype, v_init.dtype, scalar.dtype))
            scalar_term = scalar.reshape(()).to(dtype=torch.float32) * 0.0
            return (
                x_seq * 0.0 + scalar_term + 1.0,
                v_init * 0.0 + scalar_term + 0.25,
                x_seq * 0.0 + scalar_term + 0.25,
            )

        monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
        monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
        monkeypatch.setattr(
            _aspy,
            "_load_extension",
            lambda symbol=symbol, implementation=implementation: (
                SimpleNamespace(**{symbol: implementation}),
                None,
            ),
        )

        output = node(x)
        (output.float().sum() + node.v.sum()).backward()

        assert seen == [(torch.float32, torch.float32, torch.float32)]
        assert output.dtype == torch.bfloat16
        assert node.v.dtype == torch.float32
        assert node.v_seq.dtype == torch.float32
        parameter = getattr(node, parameter_name)
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert x.grad is not None and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "node",
    [
        neuron.IFNode(backend="aspy", step_mode="s", surrogate_function=surrogate.ATan()),
        neuron.LIFNode(backend="aspy", step_mode="s", surrogate_function=surrogate.ATan()),
        neuron.ParametricLIFNode(
            backend="aspy", step_mode="s", surrogate_function=surrogate.ATan()
        ),
        neuron.KLIFNode(backend="aspy", step_mode="s", surrogate_function=surrogate.ATan()),
    ],
)
def test_aspy_bf16_single_step_fallback_preserves_public_dtype(node):
    x = torch.full((2, 4), 0.5, dtype=torch.bfloat16, requires_grad=True)

    output = node(x)
    output.float().sum().backward()

    assert output.dtype == torch.bfloat16
    assert node.v.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    parameter = getattr(node, "w", getattr(node, "k", None))
    if parameter is not None:
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32


@pytest.mark.parametrize(
    ("node", "parameter_name"),
    [
        (
            neuron.ParametricLIFNode(
                backend="aspy",
                step_mode="m",
                store_v_seq=True,
                surrogate_function=surrogate.ATan(),
            ),
            "w",
        ),
        (
            neuron.KLIFNode(
                backend="aspy",
                step_mode="m",
                store_v_seq=True,
                surrogate_function=surrogate.ATan(),
            ),
            "k",
        ),
    ],
)
def test_aspy_module_bf16_conversion_preserves_fp32_master_and_state(node, parameter_name):
    node.v = torch.full((2, 4), 0.375, dtype=torch.float32)
    node.v_seq = torch.full((3, 2, 4), 0.25, dtype=torch.float32)

    node.to(dtype=torch.bfloat16)

    parameter = getattr(node, parameter_name)
    assert parameter.dtype == torch.float32
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32


def test_aspy_bf16_strict_rejection_restores_existing_runtime_state(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    node.v = torch.full((2, 4), 0.375, dtype=torch.float32)
    node.v_seq = torch.full((2, 2, 4), 0.25, dtype=torch.float32)
    original_v = node.v
    original_v_seq = node.v_seq
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(_aspy, "_load_extension", lambda: (None, "missing bundle"))

    with pytest.raises(_aspy.AsPyBackendError):
        node(torch.zeros(3, 1, 4, dtype=torch.bfloat16))

    assert node.v is original_v
    assert node.v_seq is original_v_seq
    torch.testing.assert_close(node.v, torch.full((2, 4), 0.375))
    torch.testing.assert_close(node.v_seq, torch.full((2, 2, 4), 0.25))


def test_aspy_bf16_plif_fallback_keeps_fp32_scalar_recurrence(monkeypatch):
    node = neuron.ParametricLIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.full((3, 2, 4), 0.5, dtype=torch.bfloat16, requires_grad=True)
    seen = []
    original_charge = node.neuronal_charge

    def inspect_charge(step):
        seen.append(node.w.sigmoid().to(device=step.device).dtype)
        return original_charge(step)

    monkeypatch.setattr(node, "neuronal_charge", inspect_charge)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(_aspy, "_load_extension", lambda: (None, "missing bundle"))

    output = node(x)
    output.float().sum().backward()

    assert seen == [torch.float32] * x.shape[0]
    assert output.dtype == torch.bfloat16
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32
    assert node.w.dtype == torch.float32
    assert node.w.grad is not None and node.w.grad.dtype == torch.float32


def test_aspy_bf16_strict_missing_bundle_preserves_conversion_route(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(_aspy, "_load_extension", lambda: (None, "missing bundle"))

    with pytest.raises(_aspy.AsPyBackendError) as error_info:
        node(x)

    route = error_info.value.route
    assert route.native_launch_attempted is False
    assert route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert route.dtype_conversion_bytes == 2 * x.numel() * (2 + 4)


def test_aspy_bf16_prelaunch_fallback_preserves_public_and_state_dtypes(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.full((3, 2, 4), 0.5, dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(_aspy, "_load_extension", lambda: (None, "missing bundle"))

    output = node(x)
    output.float().sum().backward()

    assert output.dtype == torch.bfloat16
    assert node.v.dtype == torch.float32
    assert node.v_seq.dtype == torch.float32
    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    assert node.last_backend_route.backend == "torch"
    assert node.last_backend_route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert node.last_backend_route.dtype_conversion_bytes is not None


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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(if_multi_step=lambda *args: (spike_seq, v_final, v_seq)),
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
    assert node.last_backend_route.native_region == "if"
    assert node.last_backend_route.native_launch_attempted
    assert node.last_backend_route.accelerated


def test_declared_compact_capability_is_observable_in_route_metadata(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=False,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(2, 1, 1)
    spike_seq = torch.ones_like(x)
    module = SimpleNamespace(
        aspy_abi_version=lambda: 1,
        aspy_capabilities=lambda: {
            "schema_version": 1,
            "capabilities": {
                "if": ["if_forward", "if_backward"],
                "if_compact": [
                    "if_forward_compact",
                    "if_backward_compact",
                ],
            },
            "symbols": [
                "if_forward",
                "if_backward",
                "if_forward_compact",
                "if_backward_compact",
            ],
        },
        if_forward=lambda *args: None,
        if_backward=lambda *args: None,
        if_forward_compact=lambda *args: None,
        if_backward_compact=lambda *args: None,
        if_multi_step=lambda *args: (spike_seq, torch.zeros_like(x[0]), None),
    )
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (module, _aspy._adapter_capabilities(module)),
    )

    assert node(x) is spike_seq
    assert node.last_backend_route.reason_code == "aspy.if_compact.native"
    assert node.last_backend_route.native_region == "if_compact"
    assert node.last_backend_route.abi_version == 1
    assert node.last_backend_route.schema_version == 1


def test_cupy_alias_is_preserved_in_native_route_metadata(monkeypatch):
    node = neuron.IFNode(
        backend="cupy",
        step_mode="m",
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(2, 1, 1)
    spike_seq = torch.ones_like(x)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                lif_multi_step=lambda *args: calls.append(args) or (spike_seq, v_final, v_seq)
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
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

    assert node.v == 0.0
    assert node.v_seq is None


@pytest.mark.parametrize("malformed_name", ["spike_seq", "v_final", "v_seq"])
def test_aspy_malformed_result_layout_does_not_commit_state(monkeypatch, malformed_name):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    outputs = {
        "spike_seq": torch.ones_like(x),
        "v_final": torch.ones_like(x[0]),
        "v_seq": torch.ones_like(x),
    }
    if malformed_name == "v_final":
        outputs[malformed_name] = torch.ones(2, 8)[:, ::2]
    else:
        outputs[malformed_name] = torch.ones(3, 2, 8)[:, :, ::2]
    assert outputs[malformed_name].shape == (x[0].shape if malformed_name == "v_final" else x.shape)
    assert not outputs[malformed_name].is_contiguous()
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                if_multi_step=lambda *args: (
                    outputs["spike_seq"],
                    outputs["v_final"],
                    outputs["v_seq"],
                )
            ),
            None,
        ),
    )

    with pytest.raises(ValueError, match="contiguous with storage offset zero"):
        node(x)

    assert node.v == 0.0
    assert node.v_seq is None


def test_aspy_malformed_result_physical_format_does_not_commit_state(monkeypatch):
    node = neuron.IFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    spike_seq = torch.ones_like(x)
    v_final = torch.ones_like(x[0])
    v_seq = torch.ones_like(x)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_require_npu_nd",
        lambda tensor: (
            "AsPy native bridge requires physical ACL_FORMAT_ND (2), got format=29"
            if tensor is v_final
            else None
        ),
    )
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(if_multi_step=lambda *args: (spike_seq, v_final, v_seq)),
            None,
        ),
    )

    with pytest.raises(ValueError, match="v_final is not bridge-safe"):
        node(x)

    assert node.v == 0.0
    assert node.v_seq is None


def test_aspy_lif_malformed_extension_result_does_not_commit_state(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    x = torch.zeros(3, 2, 4)
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
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

    assert node.v == 0.0
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
def test_klif_single_multi_reset_state_and_first_order_gradients(scale_reset, decay_input):
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

    x_multi = torch.tensor([[[0.4, -0.2]], [[0.6, 0.5]], [[0.3, 0.8]]], requires_grad=True)
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (SimpleNamespace(lif_multi_step=lambda *args: None), None),
    )
    fallback = neuron.KLIFNode(backend="aspy", step_mode="m", surrogate_function=surrogate.ATan())
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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                klif_multi_step=lambda *args: calls.append(args) or (spike_seq, v_final, v_seq)
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

    failing = neuron.KLIFNode(backend="aspy", step_mode="m", surrogate_function=surrogate.ATan())
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
    original_failing_v = failing.v
    with pytest.raises(RuntimeError, match="native KLIF launch failed"):
        failing(x)
    assert failing.v is original_failing_v

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
                    torch.ones(1),
                    torch.ones_like(x[0]),
                    torch.ones_like(x),
                )
            ),
            None,
        ),
    )
    original_malformed_v = malformed.v
    original_malformed_v_seq = malformed.v_seq
    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        malformed(x)
    assert malformed.v is original_malformed_v
    assert malformed.v_seq is original_malformed_v_seq


def test_aspy_native_input_mutation_cannot_corrupt_committed_state(monkeypatch):
    node = neuron.LIFNode(
        backend="aspy",
        step_mode="m",
        store_v_seq=True,
        surrogate_function=surrogate.ATan(),
    )
    node.v = torch.full((2, 4), 0.375)
    node.v_seq = torch.full((3, 2, 4), 0.25)
    original_v = node.v
    original_v_seq = node.v_seq

    def implementation(x_seq, v_init, *args):
        v_init.add_(10.0)
        raise RuntimeError("native input mutation then failure")

    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(_aspy, "_require_npu_nd", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (SimpleNamespace(lif_multi_step=implementation), None),
    )

    with pytest.raises(RuntimeError, match="native input mutation then failure"):
        node(torch.zeros(3, 2, 4))

    assert node.v is original_v
    assert node.v_seq is original_v_seq
    torch.testing.assert_close(node.v, torch.full((2, 4), 0.375))
    torch.testing.assert_close(node.v_seq, torch.full((3, 2, 4), 0.25))


@pytest.mark.parametrize(
    ("node", "parameter_name"),
    [
        (
            neuron.ParametricLIFNode(
                backend="torch",
                step_mode="m",
                surrogate_function=surrogate.ATan(),
            ),
            "w",
        ),
        (
            neuron.KLIFNode(
                backend="torch",
                step_mode="m",
                surrogate_function=surrogate.ATan(),
            ),
            "k",
        ),
    ],
)
def test_aspy_backend_switch_rejects_non_fp32_master_parameter(
    monkeypatch, node, parameter_name
):
    node.to(dtype=torch.bfloat16)
    assert getattr(node, parameter_name).dtype == torch.bfloat16
    optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
    parameter = getattr(node, parameter_name)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert any(
        isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16
        for value in optimizer.state[parameter].values()
    )

    node.backend = "aspy"
    x = torch.zeros(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    loaded = []
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: loaded.append(True) or (SimpleNamespace(), None),
    )

    with pytest.raises(RuntimeError, match="recreate the optimizer"):
        node(x)

    assert loaded == []
    assert parameter.dtype == torch.bfloat16
    assert node.v == 0.0


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
    monkeypatch.setattr(_aspy, "_unsupported_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                plif_multi_step=lambda *args: calls.append(args) or (spike_seq, v_final, v_seq)
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
    original_malformed_v = malformed.v
    original_malformed_v_seq = malformed.v_seq
    with pytest.raises(ValueError, match="spike_seq shape mismatch"):
        malformed(x)
    assert malformed.v is original_malformed_v
    assert malformed.v_seq is original_malformed_v_seq
