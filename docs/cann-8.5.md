# CANN 8.5 deployment

## Qualified target

- Ascend 910B4
- CANN 8.5.0
- Python 3.10
- PyTorch 2.9.0
- torch-npu 2.9.0

Use one consistent CANN installation in a process. `scripts/cann_env.sh` accepts an already sourced `ASCEND_TOOLKIT_HOME`, checks common toolkit locations, or sources the file named by `SPIKINGJELLY_NPU_CANN_ENV=/path/to/set_env.sh`. This repository is the authoritative code source; external builds, snapshots, logs, and test consumers belong outside it.

## Runtime probe

```bash
source scripts/cann_env.sh
python - <<'PY'
from spikingjelly_npu.npu import get_npu_info
print(get_npu_info())
PY
```

The package does not explicitly import `torch_npu` until a runtime helper is called. PyTorch may independently auto-load an installed out-of-tree device backend during `import torch`; set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` when testing the package's own import behavior in isolation.

## Release installation and optional source build

The release installer downloads the pure-Python wheel and, on the exact qualified hardware/runtime matrix, that release's relocatable native bundle containing `_spikingjelly_npu_aspy` and `libcust_opapi.so`. `--check` validates Linux aarch64, CPython 3.10, torch/torch-npu 2.9, a sourced CANN 8.5 toolkit, and an Ascend 910B device without installing. `--require-native` also probes the extracted extension and fails before installation when the FedSNN decay-LIF symbols are absent. Ordinary auto mode emits an explicit warning when the bundle supports only generic routes and cannot provide `packed_aspy`. The bundle attached to `v0.1.0-alpha.1` predates those symbols; use the source build below for `packed_aspy` until a newer release explicitly advertises support:

```bash
curl -fsSL https://raw.githubusercontent.com/CyberStaZJU/SpikingJelly_npu/main/install.sh | bash -s -- --require-native
```

The bundle is stack-specific rather than a general manylinux artifact. The loader checks it lazily only after an NPU request qualifies, preloads its packaged custom-op API library, and retains ordinary import safety.

AsPy is a focused fused SNN backend, not a general CuPy-compatible array library. Developers may instead build the Ascend extension and custom-op installation explicitly on the target host. All generated ACLNN projects, object files, libraries, bridge outputs, activation metadata, and logs must use a new external directory:

```bash
source scripts/cann_env.sh
export TASK_QUEUE_ENABLE=1
export SPIKINGJELLY_NPU_ASPY_BUILD_ROOT="$HOME/.cache/spikingjelly_npu/aspy-$(date +%Y%m%d-%H%M%S)"
scripts/build_aspy.sh
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"
ASCEND_DEVICE_ID=7 scripts/run_aspy_tests.sh
```

`build_aspy.sh` preflights the qualified Linux aarch64 / CPython 3.10 / torch and torch-npu 2.9 / CANN 8.5 matrix, consumes the checked-in operator manifest, and generates fourteen ACLNN operators across full-output IF/LIF, compact IF/LIF, learnable-`k` KLIF, dynamic-parameter PLIF, and exact stateless FedSNN decay-LIF groups, targeting `ascend910b`, then builds `_spikingjelly_npu_aspy` with `NpuExtension`. Use `PYTHON=/path/to/python3.10` to select the interpreter. Before success it imports the extension and requires every manifest-declared symbol. Compact IF/LIF now compiles and passes physical full/remainder/singleton correctness, but its five-process hotspot benchmark did not meet the promotion gate: IF was `1.0317×`, LIF `1.0093×`, and peak allocated-HBM reduction was about `10.0%` for each. Therefore compact is a correctness-qualified additive ABI, not a claimed optimization. External `build-manifest.json` records those capabilities, tool/runtime identity, and a deterministic source-input digest over each included file's path, byte size, and SHA-256. Under Git, the scope is tracked plus untracked non-ignored files; filtered snapshots use the corresponding source-file scope. The bridge submits each operator to the current NPU stream through torch-npu `OpCommand::RunOpApiV2`. It retains workspace, ACL tensors, and executor ownership in the task-queue callback and has no explicit hot-path stream/device synchronization. PLIF passes reciprocal tau as a one-element FP32 NPU tensor, not a frozen host attribute.

The native API is consumed through the public package classes, not by importing `_spikingjelly_npu_aspy` directly:

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
klif_node = neuron.KLIFNode(
    scale_reset=True,
    tau=2.5,
    decay_input=True,
    **common,
).to("npu:7")
plif_node = neuron.ParametricLIFNode(
    init_tau=2.5,
    decay_input=True,
    **common,
).to("npu:7")

x = torch.rand(8, 64, 4096, device="npu:7", requires_grad=True)
spikes = klif_node(x)
loss = spikes.square().mean() + klif_node.v.square().mean()
loss.backward()
assert klif_node.last_backend_route.backend == "aspy"
assert x.grad is not None and klif_node.k.grad is not None
```

Qualified native scope: FP32 rank-two-or-higher time-major multi-step IF/LIF/KLIF/PLIF with non-empty time and flattened-timestep dimensions; contiguous storage-offset-zero inputs/state; spiking ATan surrogate; hard or soft reset; both `detach_reset` values; LIF/KLIF/PLIF `decay_input`; KLIF `scale_reset`; first-order input/carried-state gradients; KLIF `k.grad`; and PLIF `w.grad`. BF16 public tensors use the same FP32 native scope through the explicit conversion island described below; the native ABI itself is unchanged. LIF and KLIF require fixed float `tau > 1`. KLIF native eager parity is qualified at `rtol=5e-5, atol=3e-5`; no KLIF speed or NPUGraph claim is made. Unsupported or unavailable requests pre-fallback to PyTorch unless `backend_strict=True`; in particular, strict single-step KLIF raises `AsPyBackendError` rather than silently using eager PyTorch. Native failures after launch are not hidden by replaying a stateful eager step.

## Device configuration

```python
from spikingjelly_npu.npu import configure_npu

device = configure_npu(
    "npu:7",
    jit_compile=False,
    allow_internal_format=False,
    require_bf16=True,
)
```

`jit_compile=False` selects binary ACLNN/opapi operators in torch-npu 2.9, while `allow_internal_format=False` prevents legacy internal-format ACLop kernels such as Conv2D from being selected during NPUGraph capture. In the qualified 2.9 runtime the latter is a write-only property, so it must be assigned directly rather than tested with `hasattr`.

`require_bf16=True` fails closed before process-global NPU configuration if the runtime cannot report BF16 support. The active mixed-precision profile uses BF16 for qualified Conv/Linear/MatMul and public activations. Normalization, membrane dynamics, surrogate math, reductions, final logits, master parameters, gradients, and optimizer state remain FP32. LayerNorm and Softmax are conservative FP32 targets; standard Transformer internals remain delegated to torch-npu until an explicit isolated route is physically qualified. FP8 is not enabled by this profile.

The physical tiny-shape BF16 family matrix passed 84 train/eval × full/remainder/singleton cases across standard recurrent and Transformer APIs, project spiking recurrent modules, SpikingSelfAttention, and Spikformer. Representative shapes, long trajectories, convergence, native Spikformer, graph capture, latency, memory, and FP8 remain separate qualification gates.

## NPUGraph constraints

Real NPUGraph capture/replay is qualified for fixed-shape AsPy IF, LIF, and PLIF. KLIF remains native-eager-only in the qualified scope. The bridge remains on torch-npu's current stream during capture, and PLIF's reciprocal-tau device input remains dynamic across replay. The PLIF qualification changed both `w` and input over five replays and checked output, input gradient, and `w.grad` against native eager.

General restrictions remain:

- fixed shapes, dtypes, device, logical layout, runtime-reported Ascend physical format, argument structure, and `requires_grad` state;
- tensor-only sample arguments;
- only graph-capturable ACLNN operators;
- no module hooks at capture time;
- no parameter/buffer additions or storage replacement after capture;
- no higher-order differentiation;
- autocast caching must be disabled;
- train and eval states need separate captures;
- partial batches and diagnostics remain eager only in non-strict compatibility mode; strict mode rejects them before model execution;
- hard spike thresholds can amplify small nondeterministic kernel differences.

`StaticGraphRunner` enforces the full-batch/ordinary-forward routing policy and records a fallback reason. With `strict=False`, known pre-capture rejections use observable eager fallback. With `strict=True`, every such rejection raises `GraphPreExecutionError` carrying the rejected route before any eager model call. Failure to inspect the runtime Ascend physical format is also a pre-execution rejection: non-strict mode executes eager exactly once and strict mode raises before model, capture, or replay execution. Once `torch.npu.make_graphed_callables` is entered, capture exceptions, parameter mutation, cleanup failures, and replay exceptions are fatal and poison the whole runner; no eager replay is allowed. Post-launch cleanup does not query physical formats. It restores buffers, gradients, runtime memories, mixed training modes, and CPU/NPU RNG; parameter values, storage, and versions must remain unchanged, with accidental value mutation restored only to leave the canonical eager model uncorrupted before the runner fails closed. Models must declare graph-safe per-forward state or callers must explicitly set `assume_graph_safe=True`; do not opt in models whose persistent neuron memory can be consumed by capture warmups. Training capture is disabled by default and requires both `allow_training=True` and, by default, `torch.use_deterministic_algorithms(True, warn_only=False)`. Warn-only mode is not qualified. The deterministic requirement is tracked as capture state, so changing it invalidates and rebuilds a graph. The expert-only `require_deterministic_training=False` override must not be used for formal SNN training without independent parity evidence.

## AMP

Use the explicit BF16 profile only after a fail-closed capability check:

```python
import torch
from spikingjelly_npu.npu import configure_npu, npu_bf16_autocast

device = configure_npu("npu:7", require_bf16=True)
model = model.to(device)
with npu_bf16_autocast():
    output = model(inputs.to(device))
```

`npu_bf16_autocast()` fixes `dtype=torch.bfloat16` and `cache_enabled=False`, records package-local context state without eagerly importing `torch_npu`, and restores nested/exception state transactionally. Generic `autocast(...)` remains available but does not activate the qualified model policy. The package does not silently create or control a GradScaler.

The model contract keeps FP32 master parameters, gradients, optimizer state, spiking recurrent state/gates, neuron membrane/threshold/surrogate/reset, BatchNorm/BNTT, average reductions, and final logits. Qualified Conv/Linear/MatMul operations may execute in BF16. Standard FP32-storage recurrent modules retain the eager compatibility decomposition so affine operators may autocast to BF16 while gates/state remain FP32; this avoids the unavailable FP32 `DynamicGRUV2` route. Standard Transformer wrappers delegate BF16 operator selection and internal normalization/reduction behavior to upstream torch-npu. Physical per-family qualification is mandatory before a support claim.

The AsPy native ABI and recurrence remain FP32. BF16 public IF/LIF/KLIF/PLIF and FedSNN decay-LIF tensors may use the explicit `bf16-public-fp32-aspy-island` route: inputs are materialized as bridge-safe FP32 tensors before extension loading, FP32 voltage state and PLIF/KLIF master scalars are preserved, and public spike outputs are cast back to BF16. Configure AsPy before dtype conversion and create or recreate the optimizer only after backend/dtype setup; a late backend switch with a non-FP32 learnable scalar is rejected before extension loading. Route metadata reports the conversion kind and estimated conversion bytes. This is BF16 mixed-precision interoperability, **not** a BF16-native AsPy kernel. Separately validate spike decisions, state, gradients, optimizer updates, convergence, latency, and HBM before formal use. The frozen gates are recorded in [`docs/evidence/bf16_mixed_acceptance_policy_20260804.json`](evidence/bf16_mixed_acceptance_policy_20260804.json).

## Safety

Before NPU tests:

1. inspect `npu-smi info`, including the process table for all devices;
2. use only an idle or explicitly allocated device; prefer NPU 7 only when it has no unknown process;
3. set `ASCEND_DEVICE_ID` and `DEVICE_ID` consistently;
4. write builds, wheel targets, caches, logs, profiler output, and evidence outside this repository;
5. avoid sharing a device with active FedSNN, vLLM, or unknown jobs and never kill an unknown process;
6. do not treat component qualification as a formal FedSNN end-to-end result.
