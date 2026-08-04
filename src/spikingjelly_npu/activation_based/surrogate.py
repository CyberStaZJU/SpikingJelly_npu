"""NPU-friendly surrogate gradients implemented entirely with PyTorch operators."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..npu.amp import is_npu_bf16_autocast_active


def heaviside(x: torch.Tensor) -> torch.Tensor:
    """Return one where ``x >= 0`` and zero elsewhere, preserving dtype/device."""
    return (x >= 0).to(x)


class SurrogateFunctionBase(nn.Module):
    """Base class matching SpikingJelly's spiking/primitive mode contract."""

    def __init__(self, spiking: bool = True, **parameters: float) -> None:
        super().__init__()
        self.spiking = bool(spiking)
        self._sg_param_names = tuple(parameters)
        for name, value in parameters.items():
            setattr(self, name, value)

    @property
    def _sg_params(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self._sg_param_names}

    def set_spiking_mode(self, spiking: bool) -> None:
        self.spiking = bool(spiking)

    @staticmethod
    def primitive_function(x: torch.Tensor, **kwargs: float) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_npu_bf16_autocast_active():
            public_dtype = x.dtype
            with torch.autocast(device_type="npu", enabled=False):
                state_input = x.float()
                primitive = self.primitive_function(state_input, **self._sg_params)
                output = (
                    primitive
                    if not self.spiking
                    else heaviside(state_input).detach()
                    - primitive.detach()
                    + primitive
                )
            return output.to(dtype=public_dtype) if x.dtype == torch.bfloat16 else output

        primitive = self.primitive_function(x, **self._sg_params)
        if not self.spiking:
            return primitive
        # Straight-through construction: exact Heaviside forward, primitive gradient.
        return heaviside(x).detach() - primitive.detach() + primitive

    def extra_repr(self) -> str:
        params = ", ".join(f"{key}={value}" for key, value in self._sg_params.items())
        return f"{params}, spiking={self.spiking}" if params else f"spiking={self.spiking}"


class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha: float = 4.0, spiking: bool = True) -> None:
        super().__init__(spiking=spiking, alpha=float(alpha))

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.sigmoid(alpha * x)


class ATan(SurrogateFunctionBase):
    def __init__(self, alpha: float = 2.0, spiking: bool = True) -> None:
        super().__init__(spiking=spiking, alpha=float(alpha))

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        return torch.atan((math.pi / 2.0) * alpha * x) / math.pi + 0.5


class PiecewiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha: float = 1.0, spiking: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        super().__init__(spiking=spiking, alpha=float(alpha))

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        lower = -1.0 / alpha
        upper = 1.0 / alpha
        middle = -0.5 * alpha * alpha * x.abs() * x + alpha * x + 0.5
        return torch.where(
            x < lower,
            torch.zeros_like(x),
            torch.where(x > upper, torch.ones_like(x), middle),
        )


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha: float = 2.0, spiking: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        super().__init__(spiking=spiking, alpha=float(alpha))

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        ax = alpha * x
        return 0.5 * (ax / (1.0 + ax.abs()) + 1.0)


class SuperSpike(SurrogateFunctionBase):
    """SuperSpike primitive with derivative ``alpha / (1 + |x|)^2``."""

    def __init__(self, alpha: float = 1.0, spiking: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        super().__init__(spiking=spiking, alpha=float(alpha))

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        # Continuous primitive centered at 0.5. It need not be probability-bounded
        # for spiking mode; its derivative is the published SuperSpike gradient.
        return 0.5 + alpha * x.sign() * (1.0 - 1.0 / (1.0 + x.abs()))


__all__ = [
    "heaviside",
    "SurrogateFunctionBase",
    "Sigmoid",
    "ATan",
    "PiecewiseQuadratic",
    "SoftSign",
    "SuperSpike",
]
