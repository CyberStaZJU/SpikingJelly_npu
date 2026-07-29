import math

import pytest
import torch

from spikingjelly_npu.activation_based import surrogate


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


def test_non_spiking_mode_returns_primitive():
    x = torch.linspace(-1, 1, 5)
    fn = surrogate.ATan(alpha=2.0, spiking=False)
    expected = torch.atan(math.pi * x) / math.pi + 0.5
    torch.testing.assert_close(fn(x), expected)
