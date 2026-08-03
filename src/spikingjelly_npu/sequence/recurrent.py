"""Dense recurrent layers with the standard :mod:`torch.nn` semantics."""

from torch import nn


class RNN(nn.RNN):
    """A direct :class:`torch.nn.RNN` subclass using the eager Torch forward."""

    def forward(self, input, hx=None):
        return super().forward(input, hx)


class GRU(nn.GRU):
    """A direct :class:`torch.nn.GRU` subclass using the eager Torch forward."""

    def forward(self, input, hx=None):
        return super().forward(input, hx)


class LSTM(nn.LSTM):
    """A direct :class:`torch.nn.LSTM` subclass using the eager Torch forward."""

    def forward(self, input, hx=None):
        return super().forward(input, hx)


__all__ = ["RNN", "GRU", "LSTM"]
