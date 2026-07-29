"""FedSNN-style feedforward SNN using the public compatibility subset."""

from __future__ import annotations

import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, layer, neuron, surrogate
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu, is_npu_available


class FeedforwardSNN(nn.Module):
    def __init__(self, inputs: int = 700, hidden: int = 128, classes: int = 20) -> None:
        super().__init__()
        self.input = layer.Linear(inputs, hidden, step_mode="m")
        self.lif = neuron.LIFNode(
            tau=20.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=True,
            step_mode="m",
            backend="torch",
        )
        self.readout = nn.Linear(hidden, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        sequence = inputs.transpose(0, 1).contiguous()
        spikes = self.lif(self.input(sequence))
        return self.readout(spikes.mean(0))


def main() -> None:
    device = configure_npu("npu:0") if is_npu_available() else torch.device("cpu")
    model = FeedforwardSNN().to(device).eval()
    # This compatibility model resets persistent neuron memory inside forward.
    # Explicitly opt in only after validating the exact consumer model.
    runner = StaticGraphRunner(model, batch_size=32, assume_graph_safe=True)
    inputs = torch.rand(32, 50, 700, device=device)
    logits = runner(inputs)
    print({"shape": list(logits.shape), "backend": runner.last_route.backend})


if __name__ == "__main__":
    main()
