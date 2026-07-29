# spikingjelly_npu

> **Upstream lineage:** This project is derived from and built upon [SpikingJelly](https://github.com/fangwei123456/spikingjelly), the open-source SNN framework created and maintained by Fang Wei and its contributors. `spikingjelly_npu` adapts its activation-based design and compatible API subset for Ascend NPU/CANN, while adding the AsPy Ascend C kernels, NPUGraph integration, runtime routing, packaging, and deployment work in this repository. It is an independent downstream project and is not an official SpikingJelly distribution.

A PyTorch-native SNN core for **Ascend NPU / CANN**, with a focused compatibility layer for SpikingJelly `activation_based` APIs, graph-safe FedSNN building blocks, and an optional fused AsPy backend.

The package contains **no CuPy, CUDA, or Triton dependency**. Ordinary functionality runs on CPU for development and dispatches PyTorch operators to Ascend through torch-npu on the server. The Python wheel is pure Python. GitHub Releases additionally provide a stack-pinned, relocatable AsPy bundle for the qualified CANN 8.5 / torch-npu 2.9 environment; native source builds remain available for development.

> Status: `0.1.0` alpha. Qualified target: Ascend 910B4, CANN 8.5.0, Python 3.10, PyTorch 2.9.0, and torch-npu 2.9.0. Run real NPU tests only on an idle or allocated device.

## Features

- SpikingJelly-style `activation_based` APIs:
  - `IFNode`, `LIFNode`, `ParametricLIFNode`;
  - `surrogate.ATan`, `Sigmoid`, `PiecewiseQuadratic`, `SoftSign`, `SuperSpike`;
  - reset/detach/backend/step-mode helpers and sequence utilities;
  - step-aware Linear, Conv1d/2d/3d, BatchNorm, pooling, flatten, and voting layers.
- Correct hard/soft reset, `detach_reset`, `store_v_seq`, persistent state, and PLIF `w` state-dict semantics.
- Packed stateless layers over `[T*N, ...]` and `StaticGraphRunner` fixed-full-batch NPUGraph with visible eager fallback.
- Optional AsPy FP32 multi-step native routes for IF, fixed-tau LIF, and dynamic-parameter PLIF on Ascend 910B4/CANN 8.5.
- AsPy first-order ATan-surrogate backward, input/carried-state gradients, and PLIF `w.grad`.
- Real fixed-shape NPUGraph capture/replay for AsPy IF, LIF, and PLIF; PLIF reciprocal tau remains a dynamic device-tensor input during replay.
- Observable AsPy fallback/strict decisions through `last_backend_route` and `AsPyBackendError`.
- CPU fallback and import safety: the package does not explicitly import `torch_npu` at import time. PyTorch may independently autoload an installed device backend unless `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is set.
- FedSNN-oriented components and a process-local opt-in alias for consumers that still import `spikingjelly`.

## Installation

### One-line GitHub Release install

```bash
curl -fsSL https://raw.githubusercontent.com/CyberStaZJU/SpikingJelly_npu/main/install.sh | bash
```

The installer verifies SHA-256 hashes and installs the pure-Python wheel. On the exact qualified native matrix—Linux aarch64, CPython 3.10, torch 2.9.0, torch-npu 2.9.0, CANN 8.5 and Ascend 910B—it also installs the relocatable AsPy bundle. Use `--require-native` to reject a mismatched environment or `--fallback-only` to install only the eager-compatible wheel. It never uses `sudo` or changes the system CANN installation.

Ordinary import remains side-effect safe:

```python
import spikingjelly_npu
```

Existing code that still imports the qualified `spikingjelly.activation_based` subset can opt into a process-local alias before importing that code:

```python
import spikingjelly_npu
spikingjelly_npu.enable_compat()
```

A dedicated Ascend process that also retains common `.cuda()` and `torch.cuda.*` conveniences may use `enable_compat(cuda=True)`. This invokes torch-npu's official `transfer_to_npu` compatibility module. It is process-global and is **not** an emulator for CuPy arrays, custom CUDA extensions, arbitrary CUDA APIs, NCCL launch semantics, or the complete SpikingJelly catalog.

### CPU development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
```

### CANN 8.5 runtime

Source the qualified CANN environment. The helper uses an existing `ASCEND_TOOLKIT_HOME`, discovers common CANN locations, or accepts `SPIKINGJELLY_NPU_CANN_ENV=/path/to/set_env.sh`:

```bash
source scripts/cann_env.sh
python -m pip install -e . --no-deps
ASCEND_DEVICE_ID=2 scripts/run_npu_tests.sh
```

The run scripts inspect the `npu-smi` process table and refuse an already occupied selected NPU. This is a safety check, not a reservation mechanism. Never kill unknown work.

## Optional AsPy build

AsPy mirrors the role of fused SpikingJelly neuron kernels; it is not a general CuPy-compatible array API. Keep all generated operator projects, libraries, extension objects, caches, and logs outside the repository:

```bash
source scripts/cann_env.sh
export TASK_QUEUE_ENABLE=1
export SPIKINGJELLY_NPU_ASPY_BUILD_ROOT="$HOME/.cache/spikingjelly_npu/aspy-$(date +%Y%m%d-%H%M%S)"
scripts/build_aspy.sh
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"
ASCEND_DEVICE_ID=7 scripts/run_aspy_tests.sh
```

`build_aspy.sh` generates and builds independent IF, fixed-tau LIF, and PLIF forward/backward ACLNN operators for `ascend910b`, then builds `_spikingjelly_npu_aspy` with torch-npu `NpuExtension`. The bridge submits on the current NPU stream through `at_npu::native::OpCommand::RunOpApiV2`, keeps workspace and ACL object owners alive in the task-queue callback, and has no explicit synchronization in the native forward/backward hot path.

Qualified native inputs are non-empty, contiguous, storage-offset-zero FP32 time-major sequences on NPU with a spiking `surrogate.ATan`. LIF requires a fixed Python float `tau > 1`; PLIF transports reciprocal tau as a one-element contiguous FP32 NPU tensor, so `w` and `w.grad` remain live. Only first-order gradients are supported. Native single-step, FP16/BF16, non-ATan or non-spiking surrogates, unsupported layouts, and arbitrary dynamic-shape graphs use observable pre-execution fallback or strict failure.

## AsPy public interface

```python
import torch
from spikingjelly_npu.activation_based import neuron, surrogate

common = dict(
    v_threshold=0.7,
    v_reset=None,
    surrogate_function=surrogate.ATan(alpha=2.0),
    detach_reset=False,
    step_mode="m",
    backend="aspy",
    backend_strict=True,
    store_v_seq=True,
)

if_node = neuron.IFNode(**common).to("npu:7")
lif_node = neuron.LIFNode(tau=2.5, decay_input=True, **common).to("npu:7")
plif_node = neuron.ParametricLIFNode(
    init_tau=2.5,
    decay_input=True,
    **common,
).to("npu:7")

x = torch.rand(8, 64, 4096, device="npu:7", requires_grad=True)
spikes = plif_node(x)
loss = spikes.square().mean() + plif_node.v.square().mean()
loss.backward()

assert plif_node.last_backend_route.backend == "aspy"
assert x.grad is not None and plif_node.w.grad is not None
```

With `backend_strict=False`, unsupported or unavailable requests fall back to the PyTorch implementation before native execution. With `backend_strict=True`, the same request raises `AsPyBackendError`. Once a native launch starts, failures propagate rather than silently replaying a stateful eager step.

## Ordinary PyTorch quick start

```python
import torch
from torch import nn
from spikingjelly_npu.activation_based import functional, layer, neuron, surrogate

net = nn.Sequential(
    layer.Linear(700, 128, step_mode="m"),
    neuron.LIFNode(
        tau=20.0,
        decay_input=True,
        surrogate_function=surrogate.ATan(alpha=2.0),
        detach_reset=True,
        step_mode="m",
        backend="torch",  # ordinary PyTorch ops; device may be CPU or NPU
    ),
    layer.Linear(128, 20, step_mode="m"),
)

x = torch.rand(50, 32, 700)  # [T, N, F]
functional.reset_net(net)
y = net(x).mean(0)
```

## NPUGraph routing

```python
from spikingjelly_npu.npu import StaticGraphRunner

model = model.to("npu:2").eval()
runner = StaticGraphRunner(model, batch_size=128)
logits = runner(full_batch)       # fixed full batch: capture/replay when qualified
logits = runner(remainder_batch)  # partial batch: eager fallback
print(runner.last_route)
```

Real fixed-shape NPUGraph is qualified for AsPy IF, LIF, and PLIF as well as the packed model path. Replay requires unchanged argument structure, shape, dtype, device, layout, `requires_grad`, train/eval state, and parameter/buffer identities and storage. Separate train/eval captures are required. Models must declare graph-safe per-forward state or callers must explicitly opt in after verifying it. Training capture is disabled by default; the qualified opt-in path requires deterministic algorithms unless the expert-only override is used. Higher-order differentiation and arbitrary dynamic shapes are unsupported.

PLIF graph replay reads reciprocal tau from a device tensor. Qualification changed both `w` and input across five fixed-shape replays and checked output, input-gradient, and `w.grad` parity against native eager.

## FedSNN migration

Preferred explicit import:

```python
from spikingjelly_npu.activation_based import functional, layer, neuron, surrogate
```

If source imports cannot yet be changed, enable the process-local alias before importing the consumer:

```python
import spikingjelly_npu
spikingjelly_npu.enable_compat()
```

This intentionally shadows an installed but not-yet-imported real SpikingJelly package for the documented allowlist. It refuses partial replacement after real SpikingJelly has already been imported. The equivalent environment-gated bootstrap is `SPIKINGJELLY_NPU_COMPAT=spikingjelly`; `SPIKINGJELLY_NPU_COMPAT=ascend` additionally enables torch-npu's official CUDA convenience transfer. See [`docs/fedsnn-integration.md`](docs/fedsnn-integration.md). This release does not claim a formal FedSNN end-to-end training, federated-round, or convergence result.

## Compatibility boundaries

- sequences are time-major `[T,N,...]`;
- neuron state persists until `reset()` / `reset_net()`;
- `v` and `v_seq` are not state-dict entries;
- PLIF stores scalar parameter `w`;
- `backend="torch"` works on CPU or NPU tensors;
- `backend="npu"` is an explicit alias for ordinary PyTorch operators on NPU;
- `backend="aspy"` is the optional IF/LIF/PLIF Ascend route, not a CuPy array backend;
- for compatible SpikingJelly constructors, `backend="cupy"` is accepted only as an observable preference alias for AsPy with eager fallback; no CuPy array/kernel API is implemented;
- CUDA CuPy kernels, Triton CUDA kernels, custom CUDA extensions, and the full SpikingJelly catalog are intentionally unsupported.

See [`docs/compatibility.md`](docs/compatibility.md) for the full contract.

## Current component evidence

All AsPy numbers below are component-level measurements on Ascend 910B4 `npu:7`, CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32. They are not formal FedSNN end-to-end results.

- **IF three paths:** fixed `[B,T,F]=[64,8,4096]`; full reset + gain + IF + linear readout + MSE + backward; 10 warmups, 50 measured iterations, synchronize after warmup and every measured iteration, three fresh processes. Median-of-run-medians: PyTorch 9.241211 ms, AsPy eager 2.685313 ms, AsPy NPUGraph 2.056798 ms. Median run speedups: 3.401488× eager and 4.606828× graph. All graph routes were true NPUGraph; output/loss were exact and maximum gain-gradient absolute error was `9.313225746e-10`.
- **PLIF:** fixed `[T,B,F]=[8,64,4096]`; forward + spike/final-state loss + backward for input and `w`; 10 warmups, 50 measured iterations, same synchronization policy, three fresh processes. Run speedups were 5.320089×, 6.337813×, and 4.311383×; median 5.320089×. Three fresh determinism hashes, three 20-step optimizer trajectories, and five dynamic-`w` true-NPUGraph replays passed.
- **LIF:** fixed-tau forward/backward, carried state, strict/fallback, and true fixed-shape NPUGraph are covered by the full real-NPU AsPy suite. No standalone LIF performance number is claimed.
- **Full AsPy NPU suite at final PLIF snapshot:** 76 passed.

An early scalar, one-core, host-synchronized IF prototype was approximately 189× slower than PyTorch. That result is retained only as historical rejection evidence and does not describe the current vectorized multi-block task-queue implementation.

See [`docs/performance.md`](docs/performance.md) for exact methodology and qualification boundaries, and [`docs/cann-8.5.md`](docs/cann-8.5.md) for deployment.

## Repository policy

Keep virtual environments, build outputs, wheel staging, compiler caches, profiler output, benchmark JSON, logs, datasets, checkpoints, and experiment results outside this source tree.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
