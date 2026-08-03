import os
import subprocess
import sys


def test_import_does_not_import_torch_npu():
    code = (
        "import sys; import torch; import spikingjelly_npu; "
        "from spikingjelly_npu import models, sequence; "
        "from spikingjelly_npu.activation_based import _aspy, neuron, recurrent, transformer; "
        "from spikingjelly_npu.activation_based.layer import SpikingSelfAttention; "
        "from spikingjelly_npu.activation_based.model.spikformer import spikformer_ti; "
        "assert SpikingSelfAttention.__name__ == 'SpikingSelfAttention'; "
        "assert recurrent.SpikingGRU.__name__ == 'SpikingGRU'; "
        "assert transformer.SpikingSelfAttention is SpikingSelfAttention; "
        "assert sequence.LSTM.__name__ == 'LSTM'; "
        "assert models.Spikformer.__name__ == 'Spikformer'; "
        "assert callable(spikformer_ti); "
        "node = neuron.IFNode(backend='aspy', step_mode='m'); "
        "klif = neuron.KLIFNode(backend='aspy', step_mode='m'); "
        "plif = neuron.ParametricLIFNode(backend='aspy', step_mode='m'); "
        "decay = spikingjelly_npu.fedsnn.DecayLIF(0.5, backend='aspy'); "
        "assert _aspy.eager_route('aspy', 'not executed').backend == 'torch'; "
        "node(torch.ones(2, 1, 1)); klif(torch.ones(2, 1, 1)); plif(torch.ones(2, 1, 1)); "
        "decay(torch.ones(2, 1, 1)); "
        "assert node.last_backend_route.backend == 'torch'; "
        "assert klif.last_backend_route.backend == 'torch'; "
        "assert plif.last_backend_route.backend == 'torch'; "
        "assert decay.last_backend_route.backend == 'torch'; "
        "assert 'torch_npu' not in sys.modules; print('ok')"
    )
    env = os.environ.copy()
    # PyTorch can auto-import installed out-of-tree device backends while importing
    # torch itself. Disable that independent mechanism so this test isolates the
    # package's own imports.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"


def test_enable_compat_canonical_imports_do_not_import_torch_npu():
    code = (
        "import sys; import spikingjelly_npu; spikingjelly_npu.enable_compat(); "
        "from spikingjelly.activation_based.layer import SpikingSelfAttention; "
        "from spikingjelly.activation_based.model import Spikformer; "
        "from spikingjelly.activation_based.model.spikformer import spikformer_s; "
        "from spikingjelly.activation_based.recurrent import SpikingLSTM; "
        "from spikingjelly.sequence.transformer import TransformerDecoderLayer; "
        "assert SpikingSelfAttention.__name__ == 'SpikingSelfAttention'; "
        "assert Spikformer.__name__ == 'Spikformer'; assert callable(spikformer_s); "
        "assert SpikingLSTM.__name__ == 'SpikingLSTM'; "
        "assert TransformerDecoderLayer.__name__ == 'TransformerDecoderLayer'; "
        "assert 'torch_npu' not in sys.modules; print('ok')"
    )
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"
