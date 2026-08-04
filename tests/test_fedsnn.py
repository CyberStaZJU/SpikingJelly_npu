from types import SimpleNamespace

import pytest
import torch

from spikingjelly_npu.activation_based import _aspy, surrogate
from spikingjelly_npu.fedsnn import (
    BNTT1d,
    BNTT2d,
    DecayLIF,
    MultiStepLIF,
    PackedBNTTConvNet,
    PackedBNTTMLP,
    PoissonEncoder,
)
from spikingjelly_npu.npu.amp import npu_bf16_autocast
from spikingjelly_npu.routing import ProviderRoute


def test_poisson_encoder_shape_values_and_seed_repeatability():
    encoder = PoissonEncoder(4)
    inputs = torch.tensor([[0.0, 0.5, 1.0]])
    torch.manual_seed(9)
    first = encoder(inputs)
    torch.manual_seed(9)
    second = encoder(inputs)
    assert first.shape == (4, 1, 3)
    assert torch.equal(first, second)
    assert torch.equal(first[..., 0], torch.zeros_like(first[..., 0]))
    assert torch.equal(first[..., 2], torch.ones_like(first[..., 2]))


def test_multistep_lif_matches_manual_soft_reset():
    module = MultiStepLIF(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=None)
    currents = torch.tensor([[[0.8]], [[0.8]], [[0.0]]])
    spikes = module(currents)
    torch.testing.assert_close(spikes, torch.tensor([[[0.0]], [[1.0]], [[0.0]]]))


@pytest.mark.parametrize(
    ("module", "input_shape"),
    [
        (BNTT1d(2, 3).eval(), (2, 4, 3)),
        (BNTT2d(2, 3).eval(), (2, 4, 3, 5, 5)),
    ],
)
def test_bntt_bf16_profile_normalizes_in_fp32_and_restores_public_dtype(
    monkeypatch, module, input_shape
):
    original_autocast = torch.autocast
    disabled_calls = []

    class CPUAutocast:
        def __init__(self, **kwargs):
            if not kwargs.get("enabled", True):
                disabled_calls.append(kwargs)
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
    current = torch.randn(*input_shape, dtype=torch.bfloat16)

    with npu_bf16_autocast():
        output = module(current)

    assert output.dtype == torch.bfloat16
    assert all(layer.running_mean.dtype == torch.float32 for layer in module.layers)
    assert len(disabled_calls) == 1


def test_packed_bntt_mlp_bf16_profile_has_bf16_spikes_and_fp32_logits(monkeypatch):
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
    model = PackedBNTTMLP(4, 6, 3, 2).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    current = torch.randn(2, 4, 4, requires_grad=True)

    with npu_bf16_autocast():
        logits, spikes = model(current, return_spikes=True)
        loss = logits.square().mean()
    loss.backward()
    optimizer.step()

    assert logits.dtype == torch.float32
    assert spikes.dtype == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad.dtype == torch.float32 for parameter in model.parameters())


def test_decay_lif_exact_order_forward_gradient_and_state_dict_neutrality():
    module = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
    )
    reference_module = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
    )
    current = torch.tensor(
        [[[0.8, 0.2]], [[0.1, 0.8]], [[0.3, 0.1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_current = current.detach().clone().requires_grad_(True)

    actual = module(current)
    membrane = torch.zeros_like(reference_current[0])
    expected_spikes = []
    for t in range(reference_current.shape[0]):
        charged = membrane * reference_module.membrane_decay
        charged = charged + reference_current[t]
        spike = reference_module.surrogate_function(charged - reference_module.v_threshold)
        membrane = charged - spike.detach() * reference_module.v_threshold
        expected_spikes.append(spike)
    expected = torch.stack(expected_spikes)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(current.grad, reference_current.grad, rtol=0, atol=0)
    assert module.state_dict() == {}
    assert not hasattr(module, "membrane")
    assert module.last_backend_route.backend == "torch"


@pytest.mark.parametrize("shape", [(0, 2, 4), (3, 0, 4)])
@pytest.mark.parametrize(
    ("backend", "strict"),
    [("torch", False), ("aspy", False), ("aspy", True)],
)
def test_decay_lif_rejects_empty_public_shapes_consistently(shape, backend, strict):
    module = DecayLIF(0.5, backend=backend, backend_strict=strict)
    match = "at least one time step" if shape[0] == 0 else "at least one neuron"
    with pytest.raises(ValueError, match=match):
        module(torch.empty(shape))


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_decay_lif_rejects_non_finite_threshold(threshold):
    with pytest.raises(ValueError, match="v_threshold must be finite"):
        DecayLIF(0.5, v_threshold=threshold)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_decay_lif_rejects_invalid_atan_alpha(alpha):
    with pytest.raises(ValueError, match="ATan surrogate alpha"):
        DecayLIF(0.5, surrogate_function=surrogate.ATan(alpha=alpha))


def test_decay_lif_aspy_cpu_fallback_and_strict_pre_execution(monkeypatch):
    eager = DecayLIF(0.5, backend="torch")
    routed = DecayLIF(0.5, backend="aspy")
    eager_current = torch.rand(3, 2, 4, requires_grad=True)
    routed_current = eager_current.detach().clone().requires_grad_(True)

    expected = eager(eager_current)
    actual = routed(routed_current)
    expected.sum().backward()
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(routed_current.grad, eager_current.grad)
    assert isinstance(routed.last_backend_route, ProviderRoute)
    assert isinstance(routed.last_backend_route, _aspy.AsPyRoute)
    assert routed.last_backend_route.backend == "torch"
    assert routed.last_backend_route.logical_operation == "fedsnn.decay_lif"
    assert routed.last_backend_route.reason_code == ("aspy.fedsnn_decay_lif.unsupported_request")
    assert not routed.last_backend_route.native_launch_attempted
    assert "requires an NPU tensor" in routed.last_backend_route.reason

    load_calls = []
    monkeypatch.setattr(_aspy, "_load_extension", lambda: load_calls.append(True))
    strict = DecayLIF(0.5, backend="aspy", backend_strict=True)
    with pytest.raises(_aspy.AsPyBackendError, match="requires an NPU tensor"):
        strict(torch.ones(2, 1, 1))
    assert load_calls == []


def test_fedsnn_base_format_gate_allows_only_nd_or_rank5_ncdhw(monkeypatch):
    current = torch.zeros(4, 2, 3, 5, 5)
    monkeypatch.setattr(_aspy, "_npu_format_value", lambda tensor: (2, None))
    assert _aspy._require_fedsnn_base_format(current) is None

    monkeypatch.setattr(_aspy, "_npu_format_value", lambda tensor: (30, None))
    assert _aspy._require_fedsnn_base_format(current) is None
    reason = _aspy._require_fedsnn_base_format(torch.zeros(4, 2, 15))
    assert "rank-5 ACL_FORMAT_NCDHW (30)" in reason

    monkeypatch.setattr(_aspy, "_npu_format_value", lambda tensor: (29, None))
    reason = _aspy._require_fedsnn_base_format(current)
    assert "got format=29" in reason


def test_decay_lif_fake_native_route_returns_only_spikes(monkeypatch):
    module = DecayLIF(
        0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
    )
    current = torch.zeros(3, 2, 4)
    spike_seq = torch.ones_like(current)
    calls = []
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=lambda *args: calls.append(args) or spike_seq,
            ),
            None,
        ),
    )

    output = module(current)

    assert output is spike_seq
    assert calls[0][0] is current
    assert calls[0][1:] == (0.75, 0.7, "atan", 2.5)
    assert isinstance(module.last_backend_route, ProviderRoute)
    assert isinstance(module.last_backend_route, _aspy.AsPyRoute)
    assert module.last_backend_route.backend == "aspy"
    assert module.last_backend_route.reason_code == "aspy.fedsnn_decay_lif.native"
    assert module.last_backend_route.native_launch_attempted
    assert module.last_backend_route.accelerated
    assert "FedSNN decay-LIF" in module.last_backend_route.reason
    assert module.state_dict() == {}


@pytest.mark.parametrize("capability", [False, None])
@pytest.mark.parametrize("strict", [False, True])
def test_decay_lif_old_native_feature_gap_is_observable_before_execution(
    monkeypatch, capability, strict
):
    module = DecayLIF(0.75, backend="aspy", backend_strict=strict)
    current = torch.zeros(3, 2, 4)
    calls = []
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    extension = SimpleNamespace(
        fedsnn_decay_lif=lambda *args: calls.append(args),
    )
    if capability is not None:
        extension.supports_fedsnn_decay_lif = capability
    monkeypatch.setattr(_aspy, "_load_extension", lambda: (extension, None))

    if strict:
        with pytest.raises(
            _aspy.AsPyBackendError,
            match="does not provide FedSNN decay-LIF support",
        ):
            module(current)
    else:
        actual = module(current)
        expected = module._torch_forward(current)
        torch.testing.assert_close(actual, expected)
        assert module.last_backend_route.backend == "torch"
        assert "does not provide FedSNN decay-LIF support" in module.last_backend_route.reason
    assert calls == []


@pytest.mark.parametrize(
    ("native_result", "error_type", "match"),
    [
        ("not-a-tensor", TypeError, "result must be a tensor"),
        (torch.zeros(2, 2), ValueError, "shape mismatch"),
        (torch.zeros(3, 2, 4, dtype=torch.float64), ValueError, "device and dtype"),
    ],
)
def test_decay_lif_malformed_native_results_propagate(
    monkeypatch, native_result, error_type, match
):
    module = DecayLIF(0.75, backend="aspy")
    current = torch.zeros(3, 2, 4)
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=lambda *args: native_result,
            ),
            None,
        ),
    )

    eager_calls = []
    monkeypatch.setattr(
        module,
        "_torch_forward",
        lambda value: eager_calls.append(value) or torch.zeros_like(value),
    )
    with pytest.raises(error_type, match=match):
        module(current)
    assert eager_calls == []


def test_decay_lif_malformed_native_layout_is_rejected_without_eager_replay(
    monkeypatch,
):
    module = DecayLIF(0.75, backend="aspy")
    current = torch.zeros(3, 2, 4)
    base = torch.zeros(3, 2, 8)
    noncontiguous = base[..., ::2]
    assert noncontiguous.shape == current.shape
    assert not noncontiguous.is_contiguous()
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=lambda *args: noncontiguous,
            ),
            None,
        ),
    )
    eager_calls = []
    monkeypatch.setattr(
        module,
        "_torch_forward",
        lambda value: eager_calls.append(value) or torch.zeros_like(value),
    )

    with pytest.raises(ValueError, match="contiguous with storage offset zero"):
        module(current)
    assert eager_calls == []


def test_decay_lif_malformed_native_physical_format_is_rejected(monkeypatch):
    module = DecayLIF(0.75, backend="aspy")
    current = torch.zeros(3, 2, 4)
    spike_seq = torch.zeros_like(current)
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_require_npu_nd",
        lambda tensor: (
            "AsPy native bridge requires physical ACL_FORMAT_ND (2), got format=29"
            if tensor is spike_seq
            else None
        ),
    )
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=lambda *args: spike_seq,
            ),
            None,
        ),
    )

    with pytest.raises(ValueError, match="format=29"):
        module(current)


def test_decay_lif_bf16_uses_fp32_native_island(monkeypatch):
    module = DecayLIF(0.75, backend="aspy", backend_strict=True)
    current = torch.zeros(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    calls = []

    def implementation(native_current, *args):
        calls.append((native_current, args))
        assert native_current.dtype == torch.float32
        return native_current * 0.0 + 1.0

    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=implementation,
            ),
            None,
        ),
    )

    output = module(current)
    output.float().sum().backward()

    assert len(calls) == 1
    assert output.dtype == torch.bfloat16
    assert current.grad is not None and current.grad.dtype == torch.bfloat16
    assert module.last_backend_route.backend == "aspy"
    assert module.last_backend_route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert module.last_backend_route.dtype_conversion_bytes == 2 * current.numel() * (2 + 4)


def test_decay_lif_native_launch_failure_is_never_replayed_eager(monkeypatch):
    module = DecayLIF(0.75, backend="aspy")
    current = torch.zeros(3, 2, 4)
    native_calls = []
    eager_calls = []
    monkeypatch.setattr(_aspy, "_unsupported_stateless_reason", lambda *args: None)
    monkeypatch.setattr(_aspy, "_require_fedsnn_base_format", lambda tensor: None)
    monkeypatch.setattr(
        _aspy,
        "_load_extension",
        lambda: (
            SimpleNamespace(
                supports_fedsnn_decay_lif=True,
                fedsnn_decay_lif=lambda *args: (
                    native_calls.append(args)
                    or (_ for _ in ()).throw(RuntimeError("native launch failed"))
                ),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        module,
        "_torch_forward",
        lambda value: eager_calls.append(value) or torch.zeros_like(value),
    )

    with pytest.raises(RuntimeError, match="native launch failed"):
        module(current)
    assert len(native_calls) == 1
    assert eager_calls == []


def test_packed_convnet_matches_stepwise_reference_in_eval_and_gradients():
    torch.manual_seed(11)
    packed = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=3,
        channels=(4, 6),
        hidden_features=8,
        pooled_size=2,
    ).eval()
    stepwise = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=3,
        channels=(4, 6),
        hidden_features=8,
        pooled_size=2,
    ).eval()
    stepwise.load_state_dict(packed.state_dict())
    input_seq_a = torch.rand(3, 2, 1, 8, 8, requires_grad=True)
    input_seq_b = input_seq_a.detach().clone().requires_grad_(True)
    output_a = packed.forward_current_seq(input_seq_a)
    output_b = stepwise.forward_current_seq_stepwise(input_seq_b)
    torch.testing.assert_close(output_a, output_b, rtol=1e-5, atol=1e-6)
    output_a.sum().backward()
    output_b.sum().backward()
    torch.testing.assert_close(input_seq_a.grad, input_seq_b.grad, rtol=2e-5, atol=2e-6)
    for (_, parameter_a), (_, parameter_b) in zip(
        packed.named_parameters(), stepwise.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=2e-5, atol=2e-6)


def test_packed_convnet_bf16_profile_matches_stepwise_fp32_islands(monkeypatch):
    original_autocast = torch.autocast

    class CPUAutocast:
        def __init__(self, **kwargs):
            if kwargs.get("enabled", True):
                self.context = original_autocast(
                    device_type="cpu", dtype=torch.bfloat16, cache_enabled=False
                )
            else:
                self.context = original_autocast(device_type="cpu", enabled=False)

        def __enter__(self):
            return self.context.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: CPUAutocast(**kwargs))
    torch.manual_seed(13)
    packed = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=2,
        channels=(2, 2),
        hidden_features=4,
        pooled_size=2,
    ).eval()
    stepwise = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=2,
        channels=(2, 2),
        hidden_features=4,
        pooled_size=2,
    ).eval()
    stepwise.load_state_dict(packed.state_dict())
    input_seq_a = torch.rand(2, 2, 1, 8, 8, requires_grad=True)
    input_seq_b = input_seq_a.detach().clone().requires_grad_(True)

    with npu_bf16_autocast():
        output_a = packed.forward_current_seq(input_seq_a)
        output_b = stepwise.forward_current_seq_stepwise(input_seq_b)

    assert output_a.dtype == torch.float32
    assert output_b.dtype == torch.float32
    torch.testing.assert_close(output_a, output_b, rtol=2e-2, atol=2e-2)
    output_a.sum().backward()
    output_b.sum().backward()
    torch.testing.assert_close(input_seq_a.grad, input_seq_b.grad, rtol=4e-2, atol=4e-2)


def test_packed_convnet_input_forward_and_backward():
    torch.manual_seed(12)
    model = PackedBNTTConvNet(1, 5, 2, channels=(4, 4), hidden_features=8, pooled_size=2)
    inputs = torch.rand(3, 1, 8, 8)
    logits = model(inputs)
    assert logits.shape == (3, 5)
    logits.square().mean().backward()
    assert model.conv1.weight.grad is not None
