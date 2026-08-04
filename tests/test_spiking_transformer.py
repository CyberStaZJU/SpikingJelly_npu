import pytest
import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, neuron
from spikingjelly_npu.activation_based.layer import SpikingSelfAttention
from spikingjelly_npu.npu.amp import npu_bf16_autocast


def test_ssa_legacy_module_is_the_same_canonical_class():
    from spikingjelly_npu.activation_based.transformer import (
        SpikingSelfAttention as LegacySpikingSelfAttention,
    )

    assert LegacySpikingSelfAttention is SpikingSelfAttention


def _nonsymmetric_qkv():
    q = torch.tensor(
        [[[[[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]]]]
    )
    k = torch.tensor(
        [[[[[2.0, 0.0, 1.0], [1.0, 4.0, 0.0]]]]]
    )
    v = torch.tensor(
        [[[[[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]]]]]
    )
    return q, k, v


def test_ssa_kernel_uses_exact_vktq_order_and_fixed_scale():
    q, k, v = _nonsymmetric_qkv()
    qkv = torch.stack((q, k, v), dim=2)

    actual = SpikingSelfAttention._ssa_kernel_torch(qkv, 0.125)
    expected = ((v @ k.transpose(-2, -1)) @ q) * 0.125
    wrong_order = ((q @ k.transpose(-2, -1)) @ v) * 0.125
    wrong_scale = ((v @ k.transpose(-2, -1)) @ q) * (2 ** -0.5)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong_order)
    assert not torch.allclose(actual, wrong_scale)


def test_ssa_topology_defaults_shape_and_state_dict():
    module = SpikingSelfAttention(dim=8, num_heads=2)

    assert isinstance(module.qkv_conv_bn[0], nn.Conv1d)
    assert module.qkv_conv_bn[0].in_channels == 8
    assert module.qkv_conv_bn[0].out_channels == 24
    assert module.qkv_conv_bn[0].kernel_size == (1,)
    assert module.qkv_conv_bn[0].bias is None
    assert isinstance(module.qkv_conv_bn[1], nn.BatchNorm1d)
    assert isinstance(module.proj_conv_bn[0], nn.Conv1d)
    assert module.proj_conv_bn[0].in_channels == 8
    assert module.proj_conv_bn[0].out_channels == 8
    assert module.proj_conv_bn[0].bias is None

    for node in (module.qkv_lif, module.attn_lif, module.proj_lif):
        assert node.tau == 2.0
        assert node.detach_reset
        assert node.step_mode == "m"
    assert module.qkv_lif.v_threshold == 1.0
    assert module.attn_lif.v_threshold == 0.5
    assert module.proj_lif.v_threshold == 1.0

    output = module(torch.randn(2, 3, 8, 5))
    assert output.shape == (2, 3, 8, 5)
    assert not any(key.endswith(".v") or "v_seq" in key for key in module.state_dict())


def test_ssa_forward_matches_manual_module_flow():
    torch.manual_seed(11)
    actual_module = SpikingSelfAttention(dim=4, num_heads=2).eval()
    reference_module = SpikingSelfAttention(dim=4, num_heads=2).eval()
    reference_module.load_state_dict(actual_module.state_dict())
    x = torch.randn(3, 2, 4, 5)

    actual = actual_module(x)

    qkv = reference_module.qkv_lif(reference_module.qkv_conv_bn(x))
    qkv = qkv.reshape(3, 2, 3, 2, 2, 5)
    q, k, v = qkv.unbind(dim=2)
    expected = ((v @ k.transpose(-2, -1)) @ q) * 0.125
    expected = reference_module.attn_lif(expected).reshape(3, 2, 4, 5)
    expected = reference_module.proj_lif(reference_module.proj_conv_bn(expected))

    torch.testing.assert_close(actual, expected)


def test_ssa_bf16_profile_keeps_public_spikes_and_state_boundaries(monkeypatch):
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
    module = SpikingSelfAttention(dim=4, num_heads=2).train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    x = torch.randn(2, 2, 4, 3, requires_grad=True)

    with npu_bf16_autocast():
        output = module(x)
        loss = output.float().square().mean()
    loss.backward()
    optimizer.step()

    assert output.dtype == torch.bfloat16
    assert all(
        node.v.dtype == torch.float32
        for node in (module.qkv_lif, module.attn_lif, module.proj_lif)
    )
    assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
    assert all(parameter.grad is not None for parameter in module.parameters())
    assert all(parameter.grad.dtype == torch.float32 for parameter in module.parameters())


def test_ssa_gradients_reset_and_backend_cpu_fallback():
    torch.manual_seed(12)
    module = SpikingSelfAttention(dim=4, num_heads=2, backend="aspy").eval()
    x = torch.randn(2, 2, 4, 3, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for node in (module.qkv_lif, module.attn_lif, module.proj_lif):
        assert node.backend == "aspy"
        assert node.last_backend_route.requested_backend == "aspy"
        assert node.last_backend_route.backend == "torch"
        assert not node.last_backend_route.accelerated
        assert "requires an NPU tensor" in node.last_backend_route.reason

    functional.reset_net(module)
    for node in (module.qkv_lif, module.attn_lif, module.proj_lif):
        assert isinstance(node.v, torch.Tensor)
        torch.testing.assert_close(node.v, torch.zeros_like(node.v))

    module.backend = "torch"
    assert module.backend == "torch"
    assert all(
        node.backend == "torch"
        for node in (module.qkv_lif, module.attn_lif, module.proj_lif)
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"dim": 0, "num_heads": 1}, "dim must be a positive integer"),
        ({"dim": 8, "num_heads": 0}, "num_heads must be a positive integer"),
        ({"dim": 6, "num_heads": 4}, "must be divisible"),
    ],
)
def test_ssa_constructor_validation(args, message):
    with pytest.raises(ValueError, match=message):
        SpikingSelfAttention(**args)


def test_ssa_input_and_kernel_validation():
    module = SpikingSelfAttention(dim=8, num_heads=2)
    with pytest.raises(ValueError, match=r"\[T, N, C, L\]"):
        module(torch.randn(2, 8, 5))
    with pytest.raises(ValueError, match="expected C=8"):
        module(torch.randn(2, 1, 4, 5))
    with pytest.raises(ValueError, match="dimensions must be positive"):
        module(torch.empty(2, 0, 8, 5))
    with pytest.raises(ValueError, match="qkv must have shape"):
        module._ssa_kernel_torch(torch.randn(2, 1, 6), 0.125)


def test_ssa_rejects_unsupported_backend_transactionally():
    module = SpikingSelfAttention(dim=4, num_heads=2)
    with pytest.raises(NotImplementedError):
        module.backend = "triton"
    assert module.backend == "torch"
    assert all(
        node.backend == "torch"
        for node in (module.qkv_lif, module.attn_lif, module.proj_lif)
    )
    assert sum(isinstance(child, neuron.LIFNode) for child in module.modules()) == 3
