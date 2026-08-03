"""Standard sequence APIs with eager PyTorch and FP32 NPU recurrent fallback.

RNN, GRU, and LSTM use a narrow primitive decomposition only for actual FP32
NPU inputs; other devices and dtypes retain upstream :mod:`torch.nn` dispatch.
This is a compatibility and availability path, not an acceleration claim.
"""

from . import recurrent, transformer
from .recurrent import GRU, LSTM, RNN
from .transformer import (
    MultiheadAttention,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

__all__ = [
    "recurrent",
    "transformer",
    "RNN",
    "GRU",
    "LSTM",
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
]
