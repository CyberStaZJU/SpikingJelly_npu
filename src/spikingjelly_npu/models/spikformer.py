"""Eager Spikformer image-classification model."""

from __future__ import annotations

import math

from torch import Tensor, nn

from ..activation_based import base, layer, neuron
from ..activation_based.transformer import SpikingSelfAttention


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive number, but got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, but got {value!r}")
    return result


def _lif_tau(value: float) -> float:
    tau = _positive_float("tau", value)
    if tau <= 1.0:
        raise ValueError(f"tau must be greater than 1, but got {value!r}")
    return tau


def _set_neuron_backend(module: nn.Module, backend: str) -> None:
    base.check_backend_library(backend)
    nodes = [child for child in module.modules() if isinstance(child, neuron.BaseNode)]
    for child in nodes:
        if backend not in child.supported_backends:
            raise NotImplementedError(
                f"{backend!r} is not a supported backend of {child._get_name()}"
            )
    for child in nodes:
        child.backend = backend


def _requested_backend(module: nn.Module) -> str:
    for child in module.modules():
        if isinstance(child, neuron.BaseNode):
            return child.requested_backend
    return "torch"


class SpikformerConv2dBN(nn.Module):
    """A stateless Conv2d-BatchNorm2d block with optional /2 max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
    ) -> None:
        super().__init__()
        in_channels = _positive_int("in_channels", in_channels)
        out_channels = _positive_int("out_channels", out_channels)
        kernel_size = _positive_int("kernel_size", kernel_size)
        stride = _positive_int("stride", stride)
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError(f"padding must be a non-negative integer, but got {padding!r}")

        modules: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if pool:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block = layer.SeqToANNContainer(*modules)

    def forward(self, x_seq: Tensor) -> Tensor:
        return self.block(x_seq)


class SpikformerConv2dBNLIF(nn.Module, base.MultiStepModule):
    """A Conv2d-BatchNorm2d-(MaxPool2d)-LIF stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool: bool = False,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.conv_bn = SpikformerConv2dBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool=pool,
        )
        self.neuron = neuron.LIFNode(
            tau=_lif_tau(tau),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    @property
    def backend(self) -> str:
        return self.neuron.requested_backend

    @backend.setter
    def backend(self, value: str) -> None:
        self.neuron.backend = value

    def forward(self, x_seq: Tensor) -> Tensor:
        return self.neuron(self.conv_bn(x_seq))


class SpikformerPatchStem(nn.Module, base.MultiStepModule):
    """Four-stage /16 spiking patch stem with residual position encoding."""

    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 256,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.img_size_h = _positive_int("img_size_h", img_size_h)
        self.img_size_w = _positive_int("img_size_w", img_size_w)
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dims = _positive_int("embed_dims", embed_dims)
        if patch_size != 16:
            raise ValueError(
                "SpikformerPatchStem requires patch_size=16 for its fixed four-stage "
                f"downsampling pipeline, but got {patch_size!r}"
            )
        if self.embed_dims % 8 != 0:
            raise ValueError(
                f"embed_dims={self.embed_dims} must be divisible by 8 for the patch stem"
            )
        self.patch_size = patch_size
        self.image_size = (self.img_size_h, self.img_size_w)
        self.grid_size = (
            self.img_size_h // self.patch_size,
            self.img_size_w // self.patch_size,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        stage_dims = (
            self.embed_dims // 8,
            self.embed_dims // 4,
            self.embed_dims // 2,
            self.embed_dims,
        )
        stages: list[nn.Module] = []
        stage_in_channels = self.in_channels
        for stage_out_channels in stage_dims:
            stages.append(
                SpikformerConv2dBNLIF(
                    in_channels=stage_in_channels,
                    out_channels=stage_out_channels,
                    kernel_size=3,
                    padding=1,
                    pool=True,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
            )
            stage_in_channels = stage_out_channels
        self.stages = nn.Sequential(*stages)
        self.positional_encoding = SpikformerConv2dBNLIF(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=3,
            padding=1,
            pool=False,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )

    @property
    def backend(self) -> str:
        return _requested_backend(self)

    @backend.setter
    def backend(self, value: str) -> None:
        _set_neuron_backend(self, value)

    def forward(self, x_seq: Tensor) -> Tensor:
        if x_seq.ndim != 5:
            raise ValueError(
                "expected patch-stem input [T, N, C, H, W], "
                f"but got shape={tuple(x_seq.shape)}"
            )
        if x_seq.shape[2:] != (self.in_channels, self.img_size_h, self.img_size_w):
            raise ValueError(
                "expected patch-stem [C, H, W]="
                f"{(self.in_channels, self.img_size_h, self.img_size_w)}, "
                f"but got {tuple(x_seq.shape[2:])}"
            )
        output = self.stages(x_seq)
        return output + self.positional_encoding(output)


class SpikformerMLP(nn.Module, base.MultiStepModule):
    """Token-last Conv1d-BN-LIF Spikformer feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        in_features = _positive_int("in_features", in_features)
        hidden_features = _positive_int("hidden_features", hidden_features)
        out_features = _positive_int("out_features", out_features)
        self.fc1 = layer.SeqToANNContainer(
            nn.Conv1d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_features),
        )
        self.neuron1 = neuron.LIFNode(
            tau=_lif_tau(tau),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )
        self.fc2 = layer.SeqToANNContainer(
            nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_features),
        )
        self.neuron2 = neuron.LIFNode(
            tau=_lif_tau(tau),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
        )

    @property
    def backend(self) -> str:
        return self.neuron1.requested_backend

    @backend.setter
    def backend(self, value: str) -> None:
        self.neuron1.backend = value
        self.neuron2.backend = value

    def forward(self, x_seq: Tensor) -> Tensor:
        if x_seq.ndim != 4:
            raise ValueError(
                "expected MLP input [T, N, C, L], "
                f"but got shape={tuple(x_seq.shape)}"
            )
        output = self.neuron1(self.fc1(x_seq))
        return self.neuron2(self.fc2(output))


class SpikformerBlock(nn.Module, base.MultiStepModule):
    """Residual SpikingSelfAttention followed by a residual spiking MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        dim = _positive_int("dim", dim)
        ratio = _positive_float("mlp_ratio", mlp_ratio)
        hidden_features = int(dim * ratio)
        if hidden_features <= 0:
            raise ValueError(
                f"int(dim * mlp_ratio) must be positive, but got {hidden_features}"
            )
        self.attn = SpikingSelfAttention(
            dim=dim,
            num_heads=num_heads,
            backend=backend,
        )
        self.mlp = SpikformerMLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )

    @property
    def backend(self) -> str:
        return self.attn.backend

    @backend.setter
    def backend(self, value: str) -> None:
        self.attn.backend = value
        self.mlp.backend = value

    def forward(self, x_seq: Tensor) -> Tensor:
        if x_seq.ndim != 5:
            raise ValueError(
                "expected block input [T, N, C, H, W], "
                f"but got shape={tuple(x_seq.shape)}"
            )
        time_steps, batch_size, channels, height, width = x_seq.shape
        if min(time_steps, batch_size, channels, height, width) <= 0:
            raise ValueError(
                "all [T, N, C, H, W] dimensions must be positive, "
                f"but got shape={tuple(x_seq.shape)}"
            )
        if channels != self.attn.dim:
            raise ValueError(
                f"expected C={self.attn.dim}, but got C={channels} in "
                f"shape={tuple(x_seq.shape)}"
            )
        tokens = x_seq.flatten(3)
        tokens = tokens + self.attn(tokens)
        tokens = tokens + self.mlp(tokens)
        return tokens.reshape(time_steps, batch_size, channels, height, width).contiguous()


class Spikformer(nn.Module, base.MultiStepModule):
    """Spikformer classifier returning one logit tensor per time step."""

    def __init__(
        self,
        T: int = 4,
        in_channels: int = 3,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        embed_dims: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        depths: int = 4,
        backend: str = "torch",
        tau: float = 2.0,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.T = _positive_int("T", T)
        self.in_channels = _positive_int("in_channels", in_channels)
        self.img_size_h = _positive_int("img_size_h", img_size_h)
        self.img_size_w = _positive_int("img_size_w", img_size_w)
        self.num_classes = _positive_int("num_classes", num_classes)
        self.embed_dims = _positive_int("embed_dims", embed_dims)
        self.num_heads = _positive_int("num_heads", num_heads)
        self.depths = _positive_int("depths", depths)
        self.mlp_ratio = _positive_float("mlp_ratio", mlp_ratio)
        if patch_size != 16:
            raise ValueError(f"Spikformer requires patch_size=16, but got {patch_size!r}")
        if self.embed_dims % 8 != 0:
            raise ValueError(
                f"embed_dims={self.embed_dims} must be divisible by 8 for the patch stem"
            )
        if self.embed_dims % self.num_heads != 0:
            raise ValueError(
                f"embed_dims={self.embed_dims} must be divisible by num_heads={self.num_heads}"
            )
        tau = _lif_tau(tau)
        base.check_backend_library(backend)

        self.patch_size = patch_size
        self.patch_embed = SpikformerPatchStem(
            img_size_h=self.img_size_h,
            img_size_w=self.img_size_w,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            backend=backend,
            tau=tau,
            detach_reset=detach_reset,
        )
        self.blocks = nn.ModuleList(
            [
                SpikformerBlock(
                    dim=self.embed_dims,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    backend=backend,
                    tau=tau,
                    detach_reset=detach_reset,
                )
                for _ in range(self.depths)
            ]
        )
        self.head = layer.Linear(self.embed_dims, self.num_classes, step_mode="m")
        self._init_weights()

    @property
    def backend(self) -> str:
        return _requested_backend(self)

    @backend.setter
    def backend(self, value: str) -> None:
        _set_neuron_backend(self, value)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Conv1d | nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def _to_sequence(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            expected = (self.in_channels, self.img_size_h, self.img_size_w)
            if tuple(x.shape[1:]) != expected:
                raise ValueError(
                    f"expected 4D input [N, C, H, W] with [C, H, W]={expected}, "
                    f"but got shape={tuple(x.shape)}"
                )
            if x.shape[0] <= 0:
                raise ValueError("input batch size must be positive")
            return x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        if x.ndim == 5:
            expected = (self.T, self.in_channels, self.img_size_h, self.img_size_w)
            actual = (x.shape[0], x.shape[2], x.shape[3], x.shape[4])
            if actual != expected:
                raise ValueError(
                    "expected 5D input [T, N, C, H, W] with "
                    f"[T, C, H, W]={expected}, but got shape={tuple(x.shape)}"
                )
            if x.shape[1] <= 0:
                raise ValueError("input batch size must be positive")
            return x
        raise ValueError(
            "expected 4D image input [N, C, H, W] or 5D sequence input "
            f"[T, N, C, H, W], but got shape={tuple(x.shape)}"
        )

    def forward_features(self, x: Tensor) -> Tensor:
        x_seq = self._to_sequence(x)
        x_seq = self.patch_embed(x_seq)
        for block in self.blocks:
            x_seq = block(x_seq)
        return x_seq.flatten(3).mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))


__all__ = [
    "Spikformer",
    "SpikformerBlock",
    "SpikformerConv2dBN",
    "SpikformerConv2dBNLIF",
    "SpikformerMLP",
    "SpikformerPatchStem",
]
