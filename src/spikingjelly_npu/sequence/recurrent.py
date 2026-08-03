"""Dense recurrent layers with the standard :mod:`torch.nn` semantics."""

from torch import nn


class RNN(nn.RNN):
    """A direct :class:`torch.nn.RNN` subclass."""


class GRU(nn.GRU):
    """A direct :class:`torch.nn.GRU` subclass."""


class LSTM(nn.LSTM):
    """A direct :class:`torch.nn.LSTM` subclass."""


__all__ = ["RNN", "GRU", "LSTM"]
