"""Step-aware wrappers around common ``torch.nn`` layers."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from . import base, functional


class _StepAwareMixin(base.StepModule):
    def _set_step_mode(self, step_mode: str) -> None:
        self.step_mode = step_mode

    def _forward_step_aware(self, x: Tensor, single_step) -> Tensor:
        if self.step_mode == "s":
            return single_step(x)
        return functional.seq_to_ann_forward(x, single_step)


class Linear(nn.Linear, _StepAwareMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        step_mode: str = "s",
    ):
        super().__init__(in_features, out_features, bias)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv1d(nn.Conv1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv2d(nn.Conv2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv3d(nn.Conv3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class BatchNorm1d(nn.BatchNorm1d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class BatchNorm2d(nn.BatchNorm2d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class BatchNorm3d(nn.BatchNorm3d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class MaxPool1d(nn.MaxPool1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class MaxPool2d(nn.MaxPool2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class MaxPool3d(nn.MaxPool3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AvgPool1d(nn.AvgPool1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AvgPool2d(nn.AvgPool2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AvgPool3d(nn.AvgPool3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class Flatten(nn.Flatten, _StepAwareMixin):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, step_mode: str = "s"):
        super().__init__(start_dim, end_dim)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class VotingLayer(nn.Module, base.StepModule):
    def __init__(self, voting_size: int = 10, step_mode: str = "s"):
        super().__init__()
        if voting_size <= 0:
            raise ValueError("voting_size must be positive")
        self.voting_size = int(voting_size)
        self.step_mode = step_mode

    def single_step_forward(self, x: Tensor) -> Tensor:
        return F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        return functional.seq_to_ann_forward(x, self.single_step_forward)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def forward(self, x_seq: Tensor) -> Tensor:
        return functional.seq_to_ann_forward(x_seq, tuple(self))


__all__ = [
    "Linear", "Conv1d", "Conv2d", "Conv3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "Flatten", "VotingLayer", "SeqToANNContainer",
]
