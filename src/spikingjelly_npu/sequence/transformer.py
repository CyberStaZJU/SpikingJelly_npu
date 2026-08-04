"""Dense attention and Transformer layers with standard :mod:`torch.nn` semantics."""

from contextlib import contextmanager
from threading import RLock

import torch
from torch import Tensor, nn

from ..npu.amp import is_npu_bf16_autocast_active

_MHA_FASTPATH_LOCK = RLock()


@contextmanager
def _qualified_mha_fastpath_policy():
    """Avoid torch-npu's mixed BF16/FP32 eval-only native MHA fast path."""

    if not is_npu_bf16_autocast_active():
        yield
        return
    with _MHA_FASTPATH_LOCK:
        was_enabled = torch.backends.mha.get_fastpath_enabled()
        if was_enabled:
            torch.backends.mha.set_fastpath_enabled(False)
        try:
            yield
        finally:
            if was_enabled:
                torch.backends.mha.set_fastpath_enabled(True)


class MultiheadAttention(nn.MultiheadAttention):
    """A direct :class:`torch.nn.MultiheadAttention` eager wrapper."""

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """A direct :class:`torch.nn.TransformerEncoderLayer` eager wrapper."""

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """A direct :class:`torch.nn.TransformerDecoderLayer` eager wrapper."""

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )


class TransformerEncoder(nn.TransformerEncoder):
    """A direct :class:`torch.nn.TransformerEncoder` eager wrapper."""

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                src,
                mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )


class TransformerDecoder(nn.TransformerDecoder):
    """A direct :class:`torch.nn.TransformerDecoder` eager wrapper."""

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )


class Transformer(nn.Transformer):
    """A direct :class:`torch.nn.Transformer` eager wrapper."""

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        with _qualified_mha_fastpath_policy():
            return super().forward(
                src,
                tgt,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                src_is_causal=src_is_causal,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )


__all__ = [
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]
