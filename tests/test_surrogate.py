import math

import pytest
import torch

from spikingjelly_npu.activation_based import surrogate
from spikingjelly_npu.npu.amp import npu_bf16_autocast


@pytest.mark.parametrize(
    ("factory", "expected_grad"),
    [
        (
            lambda: surrogate.Sigmoid(alpha=4.0),
            lambda x: 4.0
            * torch.sigmoid(4.0 * x)
            * (1 - torch.sigmoid(4.0 * x)),
        ),
        (
            lambda: surrogate.ATan(alpha=2.0),
            lambda x: 1.0 / (1.0 + (math.pi * x) ** 2),
        ),
        (
            lambda: surrogate.PiecewiseQuadratic(alpha=1.0),
            lambda x: torch.clamp(1.0 - x.abs(), min=0.0),
        ),
        (
            lambda: surrogate.SoftSign(alpha=2.0),
            lambda x: 1.0 / (1.0 + (2.0 * x).abs()) ** 2,
        ),
    ],
)
def test_spiking_forward_and_surrogate_gradient(factory, expected_grad):
    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    output = factory()(x)
    assert torch.equal(output.detach(), torch.tensor([0.0, 1.0, 1.0]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, expected_grad(x.detach()), rtol=1e-6, atol=1e-6)


def test_bf16_profile_evaluates_surrogate_in_fp32_and_restores_public_dtype(
    monkeypatch,
):
    original_autocast = torch.autocast
    primitive_dtypes = []

    class InspectATan(surrogate.ATan):
        @staticmethod
        def primitive_function(x, alpha):
            primitive_dtypes.append(x.dtype)
            return surrogate.ATan.primitive_function(x, alpha)

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
    x = torch.tensor([-0.25, 0.25], dtype=torch.bfloat16, requires_grad=True)

    with npu_bf16_autocast():
        output = InspectATan()(x)
    output.float().sum().backward()

    assert primitive_dtypes == [torch.float32]
    assert output.dtype == torch.bfloat16
    assert x.grad is not None and x.grad.dtype == torch.bfloat16


def test_non_spiking_mode_returns_primitive():
    x = torch.linspace(-1, 1, 5)
    fn = surrogate.ATan(alpha=2.0, spiking=False)
    expected = torch.atan(math.pi * x) / math.pi + 0.5
    torch.testing.assert_close(fn(x), expected)
