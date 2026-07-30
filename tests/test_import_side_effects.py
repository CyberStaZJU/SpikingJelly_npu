import os
import subprocess
import sys


def test_import_does_not_import_torch_npu():
    code = (
        "import sys; import torch; import spikingjelly_npu; "
        "from spikingjelly_npu.activation_based import neuron; "
        "node = neuron.IFNode(backend='aspy', step_mode='m'); "
        "plif = neuron.ParametricLIFNode(backend='aspy', step_mode='m'); "
        "decay = spikingjelly_npu.fedsnn.DecayLIF(0.5, backend='aspy'); "
        "node(torch.ones(2, 1, 1)); plif(torch.ones(2, 1, 1)); "
        "decay(torch.ones(2, 1, 1)); "
        "assert node.last_backend_route.backend == 'torch'; "
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
