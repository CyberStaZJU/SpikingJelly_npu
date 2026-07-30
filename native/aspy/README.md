# AsPy native backend

This directory contains **source only** for the optional Ascend-specific fused backend. It is not a general CuPy-compatible array package. The `spikingjelly-npu` Python wheel remains pure Python and does not explicitly import `torch_npu` during ordinary package import. GitHub Releases may additionally ship a separate, stack-pinned relocatable native bundle built from this source.

## Qualified scope

The release target is Ascend 910B4 with CANN 8.5.0, Python 3.10, PyTorch 2.9.0, and torch-npu 2.9.0. The native backend provides independent forward/backward ACLNN operators for:

- FP32 time-major stateless FedSNN decay-LIF with exact multiply-then-add charge order, zero initial membrane, detached soft reset, and spike-sequence-only public output;
- FP32 time-major multi-step IF;
- FP32 time-major multi-step fixed-tau LIF, with a Python float `tau > 1`;
- FP32 time-major multi-step parametric LIF (PLIF), with reciprocal tau passed as a one-element NPU tensor so the parameter remains dynamic in a fixed-shape graph.

The public contract requires rank-two-or-higher time-major tensors with a non-empty time dimension and non-empty flattened timestep, contiguous storage-offset-zero NPU inputs and matching state, a spiking `surrogate.ATan`, and first-order differentiation. The C++ bridge accepts physical `ACL_FORMAT_ND` only. The FedSNN Python adapter may accept rank-5 `ACL_FORMAT_NCDHW` convolutional sequences and clone their flattened view into fresh ND storage before launch; format 29 and other unsupported physical layouts are rejected before extension loading. Hard and soft reset, both `detach_reset` values, both LIF/PLIF `decay_input` values, optional `v_seq`, input/carried-state gradients, and PLIF `w.grad` are covered. Higher-order gradients, FP16/BF16, non-ATan or non-spiking surrogates, native single-step execution, and arbitrary dynamic-shape NPUGraph replay are not supported.

The AIV kernels use 4096-element tiles and multiple blocks, capped at 20 cores by the current tiling policy. The Python adapter pads flattened neuron rows to the required eight-FP32 alignment and crops public outputs back to their logical shape.

## Stream and graph behavior

`native/aspy/bridge/aspy_bridge.cpp` submits ACLNN work on torch-npu's current NPU stream through `at_npu::native::OpCommand::RunOpApiV2`. NPU byte-tensor workspaces and ACL tensor/executor owners are captured by the task-queue callback. The native forward/backward hot path contains no explicit `torch.npu.synchronize`, `aclrtSynchronizeStream`, or equivalent device/stream synchronization.

Real fixed-shape NPUGraph capture/replay is qualified for IF, LIF, and PLIF. PLIF passes reciprocal tau as a tensor input; qualification changed both `w` and input across five replays and checked output, input-gradient, and `w.grad` parity against native eager. Graph argument structure, shape, dtype, device, layout, and `requires_grad` state remain fixed, and the default qualified training-capture policy requires deterministic algorithms.

## External build

The native bundle attached to `v0.1.0-alpha.1` predates the FedSNN decay-LIF symbols. For `packed_aspy`, build from a current source checkout until a newer release explicitly lists those symbols in its native manifest.

Source the verified CANN 8.5 environment and choose a new empty build directory outside the repository:

```bash
source scripts/cann_env.sh
export TASK_QUEUE_ENABLE=1
export SPIKINGJELLY_NPU_ASPY_BUILD_ROOT="$HOME/.cache/spikingjelly_npu/aspy-$(date +%Y%m%d-%H%M%S)"
scripts/build_aspy.sh
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"
```

The build script preflights Linux aarch64, CPython 3.10, torch/torch-npu 2.9, and a CANN 8.5 toolkit path. Set `PYTHON=/path/to/python3.10` to select the interpreter. It generates ACLNN projects with `msopgen`, installs the reviewed IF/LIF/PLIF/FedSNN host, tiling, and AIV kernel sources, targets `ascend910b`, installs custom-op APIs under the external root, and builds `_spikingjelly_npu_aspy` with torch-npu `NpuExtension`. Before reporting success it imports the extension and requires all eight forward/backward symbols. External `build-manifest.json` records those verified capabilities, source Git identity and dirty state, and a deterministic source-input digest over each file's path, size, and SHA-256, plus runtime/tool versions, toolkit path, compiler, and target SOC. Under Git, the digest scope is every tracked plus untracked non-ignored file; a source snapshot uses the equivalent filtered source-file scope. No generated file is written into the source repository.

The public adapter is `src/spikingjelly_npu_aspy.py`. The core router imports it lazily only after device, dtype, layout, neuron, surrogate, and autograd qualification. Unsupported or unavailable requests fall back before native execution when `backend_strict=False`, and raise `AsPyBackendError` when strict. Once native execution begins, failures propagate rather than silently replaying eager code. The stateless FedSNN helper is `spikingjelly_npu.fedsnn.DecayLIF`; it creates no persistent membrane or state-dict entries and exposes its most recent route through `last_backend_route`.

## Public interfaces

Use the package's public neuron classes rather than importing the native extension directly:

```python
import torch
from spikingjelly_npu.activation_based import neuron, surrogate
from spikingjelly_npu.fedsnn import DecayLIF

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

decay_lif = DecayLIF(
    membrane_decay=0.75,
    v_threshold=0.7,
    surrogate_function=surrogate.ATan(alpha=2.0),
    backend="aspy",
    backend_strict=True,
).to("npu:7")
if_node = neuron.IFNode(**common).to("npu:7")
lif_node = neuron.LIFNode(tau=2.5, decay_input=True, **common).to("npu:7")
plif_node = neuron.ParametricLIFNode(
    init_tau=2.5,
    decay_input=True,
    **common,
).to("npu:7")

x = torch.rand(8, 64, 4096, device="npu:7", requires_grad=True)
stateless_spikes = decay_lif(x)
spikes = plif_node(x)
loss = stateless_spikes.square().mean() + spikes.square().mean() + plif_node.v.square().mean()
loss.backward()
assert decay_lif.last_backend_route.backend == "aspy"
assert plif_node.last_backend_route.backend == "aspy"
assert x.grad is not None and plif_node.w.grad is not None
```

Single-step nodes keep an observable PyTorch fallback. Inspect `last_backend_route` for the executed route and reason.

## Tests and qualification

After activation, select only a genuinely idle NPU; the test and qualification drivers reject a selected device already listed in the `npu-smi` process table:

```bash
ASCEND_DEVICE_ID=7 scripts/run_aspy_tests.sh

export SPIKINGJELLY_NPU_ASPY_QUALIFICATION_ROOT="$HOME/.cache/spikingjelly_npu/aspy-qualification-$(date +%Y%m%d-%H%M%S)"
ASCEND_DEVICE_ID=7 scripts/run_aspy_qualification.sh
```

The test driver runs the complete NPU AsPy suite. The qualification driver writes only to the external evidence root and covers IF fresh-process determinism, three 20-step trajectories, true NPUGraph, and a three-process synchronized IF three-path benchmark; it also covers PLIF fresh-process determinism, three 20-step optimizer trajectories, five dynamic-parameter graph replays, and a three-process synchronized performance benchmark. LIF parity, gradients, carried state, strict/fallback behavior, and true NPUGraph are covered by `tests/test_aspy_npu.py`.

Synchronization in qualification is a measurement/evidence boundary after warmup and each timed iteration. It is not synchronization inside the native hot path. Generic IF/LIF/PLIF results are component qualifications. The FedSNN-specific decay-LIF has also passed a separate actual AlexNet-BNTT qualification on Ascend 910B4: full=128 and remainder=42 routes, gradients at `rtol=5e-5, atol=3e-5`, six native train/eval routes, two trainer smokes, and five-process complete-client medians of 4.3882 s for `packed_aspy`, 5.0087 s for `packed_eager`, and 5.3214 s for `legacy_stepwise`. This qualifies execution, not federated convergence.
