# spikingjelly_npu

> **Upstream lineage:** This project is derived from and built upon [SpikingJelly](https://github.com/fangwei123456/spikingjelly), the open-source SNN framework created and maintained by Fang Wei and its contributors. `spikingjelly_npu` adapts its activation-based design and compatible API subset for Ascend NPU/CANN, while adding the AsPy Ascend C kernels, NPUGraph integration, runtime routing, packaging, and deployment work in this repository. It is an independent downstream project and is not an official SpikingJelly distribution.

A PyTorch-native SNN core for **Ascend NPU / CANN**, with a focused compatibility layer for SpikingJelly `activation_based` APIs, graph-safe FedSNN building blocks, and an optional fused AsPy backend.

The package contains **no CuPy, CUDA, or Triton dependency**. Ordinary functionality runs on CPU for development and dispatches PyTorch operators to Ascend through torch-npu on the server. The Python wheel is pure Python. GitHub Releases additionally provide a stack-pinned, relocatable AsPy bundle for the qualified CANN 8.5 / torch-npu 2.9 environment; native source builds remain available for development.

> Status: `0.1.0` alpha. Qualified target: Ascend 910B4, CANN 8.5.0, Python 3.10, PyTorch 2.9.0, and torch-npu 2.9.0. Run real NPU tests only on an idle or allocated device.

## Features

- SpikingJelly-style `activation_based` APIs:
  - `IFNode`, `LIFNode`, `KLIFNode`, `ParametricLIFNode`;
  - `surrogate.ATan`, `Sigmoid`, `PiecewiseQuadratic`, `SoftSign`, `SuperSpike`;
  - reset/detach/backend/step-mode helpers and sequence utilities;
  - step-aware Linear, Conv1d/2d/3d, BatchNorm, pooling, flatten, and voting layers.
- FP32 semantic-alpha sequence/model APIs:
  - direct PyTorch subclasses for standard `RNN`, `GRU`, and `LSTM`, with a narrow primitive fallback for actual FP32 NPU inputs plus ordinary upstream dispatch elsewhere; multi-head attention and Transformer encoder/decoder stacks remain eager wrappers;
  - project-defined dense spiking RNN/GRU/LSTM cells and sequence modules with explicit or persistent carry state;
  - canonical token-last `SpikingSelfAttention` and eager Spikformer models/factories.
- Correct hard/soft reset, `detach_reset`, `store_v_seq`, persistent state, and PLIF `w` state-dict semantics.
- Packed stateless layers over `[T*N, ...]`, the legacy one-shape `StaticGraphRunner`, and bounded exact-PyTree `GraphBucketRunner` routing with visible eager fallback or strict pre-execution failure.
- Optional AsPy FP32 multi-step native routes for IF, fixed-tau LIF, learnable-`k` KLIF, dynamic-parameter PLIF, and an exact stateless FedSNN `membrane_decay * membrane + current` decay-LIF scan on Ascend 910B4/CANN 8.5.
- Additive compact IF/LIF capability groups for `store_v_seq=False`: fake-native/CPU integration selects compact only from a complete declared pair (or complete unversioned legacy inference), while old full-output bundles remain usable. The physical CANN 8.5 build and correctness checks passed, but the five-process hotspot benchmark reached only `1.0317×` IF and `1.0093×` LIF with about `10.0%` peak allocated-HBM reduction, so compact routing did not meet the frozen performance gate and is not promoted as an optimization.
- AsPy first-order ATan-surrogate backward, input/carried-state gradients, KLIF `k.grad`, and PLIF `w.grad`.
- Real fixed-shape NPUGraph capture/replay for AsPy IF, LIF, and PLIF; PLIF reciprocal tau remains a dynamic device-tensor input during replay.
- Observable AsPy fallback/strict decisions through `last_backend_route` and `AsPyBackendError`.
- CPU fallback and import safety: the package does not explicitly import `torch_npu` at import time. PyTorch may independently autoload an installed device backend unless `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is set.
- FedSNN-oriented components and a process-local opt-in alias for consumers that still import `spikingjelly`.

## Installation

### One-line GitHub Release install

```bash
curl -fsSL https://raw.githubusercontent.com/CyberStaZJU/SpikingJelly_npu/main/install.sh | bash
```

The installer verifies SHA-256 hashes and installs the artifacts attached to the selected release. On the exact qualified native matrix—Linux aarch64, CPython 3.10, torch 2.9.0, torch-npu 2.9.0, CANN 8.5 and Ascend 910B—it can also install that release's relocatable AsPy bundle. `--check` reports the complete matrix without installing. Use `--require-native` to reject either a mismatched environment **or a bundle that lacks the FedSNN decay-LIF capability**; use `--fallback-only` to install only the eager-compatible wheel. In ordinary auto mode, bundle probing reports KLIF and FedSNN decay-LIF separately: missing KLIF produces an explicit KLIF fallback warning, while missing decay-LIF warns that `packed_aspy` is unavailable; legacy IF/LIF/PLIF routes remain usable. It never uses `sudo` or changes the system CANN installation.

> **`packed_aspy` availability:** the existing `v0.1.0-alpha.1` release predates the FedSNN decay-LIF symbols. Until a newer release explicitly lists `packed_aspy` in its notes and native manifest, clone `main` and use the [source-build procedure](#optional-aspy-build) below. The installer probes the extracted extension before registration and prints its `klif` and `fedsnn_decay_lif` capabilities; `--require-native` fails before package or bundle installation when the FedSNN capability is absent.

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

`build_aspy.sh` preflights the qualified Linux aarch64 / CPython 3.10 / torch and torch-npu 2.9 / CANN 8.5 matrix, then consumes `native/aspy/operator_manifest.json` to generate and validate fourteen operators across seven independent capability groups: full-output IF/LIF, compact IF/LIF, learnable-`k` KLIF, PLIF, and exact FedSNN decay-LIF forward/backward. It builds `_spikingjelly_npu_aspy` with torch-npu `NpuExtension` and verifies the manifest-declared symbol set. Set `PYTHON=/path/to/python3.10` when the desired interpreter is not `python3`. The external `build-manifest.json` records that capability set, source Git identity/dirty state, and a deterministic SHA-256 over `path`, byte size, and file SHA-256 for every tracked plus untracked non-ignored build-input file (or the equivalent filtered snapshot scope outside Git), along with runtime versions, toolkit path, `msopgen`, CMake, compiler, and target SOC. The bridge submits on the current NPU stream through `at_npu::native::OpCommand::RunOpApiV2`, keeps workspace and ACL object owners alive in the task-queue callback, and has no explicit synchronization in the native forward/backward hot path. Compact IF/LIF compiled and passed physical full/remainder/singleton correctness plus post-launch no-replay checks. Its five-process `[T,B,F]=[8,64,4096]` benchmark nevertheless failed both promotion alternatives: IF achieved `1.0317×` with `10.0%` allocated-HBM reduction, and LIF achieved `1.0093×` with `10.0%` reduction, below the required `1.25×` speedup or `20%` memory reduction.

Qualified native inputs are rank-two-or-higher FP32 time-major sequences with a non-empty time dimension and non-empty flattened timestep, contiguous storage-offset-zero NPU storage, and a spiking `surrogate.ATan`. LIF and KLIF require a fixed Python float `tau > 1`; KLIF keeps public learnable scalar `k` live and expands it into an aligned device block only at the native boundary, while PLIF transports reciprocal tau as a one-element contiguous FP32 NPU tensor so `w` and `w.grad` remain live. The native bridge itself accepts only physical `ACL_FORMAT_ND` storage. The FedSNN adapter additionally recognizes real convolutional rank-5 `ACL_FORMAT_NCDHW` outputs, reshapes and clones them into fresh ND storage before launch, and rejects other physical formats before loading or calling the extension. The FedSNN-specific stateless path preserves the application's exact charge ordering, detached soft reset, ATan derivative, and zero membrane at each public forward; it is exposed as `spikingjelly_npu.fedsnn.DecayLIF`. Only first-order gradients are supported. FP16/BF16, non-ATan or non-spiking surrogates, unsupported layouts, and arbitrary dynamic-shape graphs use observable pre-execution fallback or strict failure. Stateful IF/LIF/PLIF nodes in single-step mode intentionally remain on PyTorch even when `backend_strict=True`; strict native enforcement applies to qualified multi-step requests such as `packed_aspy`.

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

For qualified multi-step requests, `backend_strict=False` falls back to the PyTorch implementation before native execution and `backend_strict=True` raises `AsPyBackendError`. For IF/LIF with `store_v_seq=False`, the adapter uses the compact ABI only when capability metadata declares a complete callable `if_compact`/`lif_compact` pair, or an unversioned bundle exposes the complete pair; valid versioned metadata is authoritative, so undeclared raw compact symbols are ignored. `store_v_seq=True` always uses the full-output ABI, and missing compact support also uses the complete full-output ABI rather than forcing eager fallback. Single-step IF/LIF/PLIF remains an intentional observable PyTorch compatibility path under either setting. Single-step KLIF also falls back when non-strict, but `backend="aspy", backend_strict=True` rejects it because no native single-step KLIF route exists. Once a native launch starts, failures propagate rather than silently replaying a stateful eager step.

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

## Sequence and model semantic alpha

```python
import torch
from spikingjelly_npu import sequence
from spikingjelly_npu.activation_based.recurrent import SpikingGRU
from spikingjelly_npu.activation_based.model import spikformer_ti

# Standard modules preserve torch.nn constructors, parameters, state dicts,
# PackedSequence behavior, masks, and eager first-order autograd. FP32 NPU
# recurrent inputs use a primitive availability fallback; other routes stay upstream.
gru = sequence.GRU(64, 128, batch_first=True)
encoder_layer = sequence.TransformerEncoderLayer(
    d_model=128, nhead=8, batch_first=True
)

# Project-defined spiking recurrent modules use dense fixed-length sequences.
spiking_gru = SpikingGRU(64, 128, batch_first=True, stateful=True)
output, state = spiking_gru(torch.randn(8, 32, 64))
spiking_gru.detach()  # explicit TBPTT boundary

# Spikformer accepts [N,C,H,W] or explicit [T,N,C,H,W].
model = spikformer_ti(T=4, num_classes=1000, backend="torch")
```

This release stage establishes eager FP32 semantics, public namespaces, state-dict behavior, CPU tests, observable routing, and import safety. For standard RNN/GRU/LSTM, actual FP32 NPU inputs use a functional decomposition because the examined CANN 8.5 fused `DynamicGRUV2` route rejects FP32. The decomposition precomputes per-layer/direction input affine work and keeps hidden recurrence in eager primitives; it is a compatibility/availability path, not an acceleration claim, and it does not use `DynamicGRUV2`. FP16/BF16 and every non-NPU route delegate upstream unchanged. Graph capture, arbitrary dynamic-shape, compiled execution, and higher-order-gradient claims are excluded. Spiking recurrent equations are project-defined extensions. Spikformer factories provide architecture/configuration compatibility; strict external checkpoint-key/layout compatibility is not claimed. TorchScript is also outside the current recurrent semantic-alpha contract because support varies across PyTorch releases.

Physical small-shape FP32 adaptation on the qualified stack passed `18/18` cases for the standard recurrent fallback, `18/18` project spiking recurrent cases, and `18/18` standard Transformer cross-attention/encoder/decoder cases across train/eval and full/remainder/singleton batches. These are eager torch-npu availability/parity results, not native or performance claims. Tiny Spikformer eager tests passed public output/logit, loss, persistent-buffer, initial-node-state-gradient, and continuous final-state checks. The user-approved `rtol=2e-4, atol=5e-4` applies **only** to continuous BaseNode final states: the observed trigger was about `3.61e-4`, while public outputs, loss, gradients, updates, and state dict retain `rtol=2e-4, atol=2e-5`. Spikformer training remains unqualified because the input gradient differed by `8.23e-4` in 32 elements and 7/29 parameter gradients failed, with a worst absolute difference of `1.33e-2`. See [`docs/sequence-acceleration-contract.md`](docs/sequence-acceleration-contract.md), [`sequence_acceptance_policy.json`](docs/evidence/sequence_acceptance_policy.json), the narrow [`sequence_acceptance_addendum_20260803.json`](docs/evidence/sequence_acceptance_addendum_20260803.json), and the sanitized [`sequence_physical_qualification_20260803.json`](docs/evidence/sequence_physical_qualification_20260803.json).

Reusable fresh-process sequence benchmark entrypoints are provided for standard and spiking recurrent modules, standard Transformer encoder/decoder stacks, and SpikingSelfAttention/Spikformer. The formal defaults enforce five fresh processes, alternating candidate/baseline first position, a cold call, warmup, at least five seconds of synchronized measured work per implementation/process, deterministic input/state hashes, raw process records, and median-of-five aggregation. `--smoke` permits tiny CPU checks but marks the output non-evidence. Benchmark JSON must be written outside the checkout. See [`docs/performance.md`](docs/performance.md#sequence-benchmark-protocol) for commands and schema boundaries.

## NPUGraph routing

```python
from spikingjelly_npu.npu import GraphBucketRunner, GraphBucketSpec

model = model.to("npu:2").eval()
runner = GraphBucketRunner(
    model,
    [
        GraphBucketSpec((full_batch,), name="full"),
        GraphBucketSpec((remainder_batch,), name="remainder"),
    ],
)
logits = runner(full_batch)       # exact allowlisted signature
logits = runner(remainder_batch)  # separately captured exact signature
print(runner.last_route)
```

`StaticGraphRunner` remains the one-fixed-batch compatibility facade. In non-strict mode it preserves eager fallback for every known rejection that is resolved before capture/replay (diagnostic arguments, partial batches, CPU execution, graph-unsafe modules, disallowed or nondeterministic/RNG-sensitive training, hooks, signature mismatch, and failure to inspect the Ascend physical format). With `strict=True`, those same paths raise `GraphPreExecutionError` with the rejected `GraphRoute` before any eager model call. `GraphBucketRunner` uses a bounded explicit allowlist and matches the complete tensor PyTree, keyword insertion order and static values, shape, dtype, device, layout, `requires_grad`, stride, storage offset, PyTorch memory format, runtime-reported Ascend physical format, alias groups, mode, and parameter/buffer execution state. `StaticGraphRunner` also binds the runtime-reported Ascend physical format for its fixed input. Format inspection and all decision-capable signatures are resolved before graph execution; non-strict rejection executes eager exactly once, while strict rejection executes neither the model nor the graph. Once `make_graphed_callables` is entered, any exception, parameter mutation, cleanup failure, or replay failure poisons the runner and is fatal; eager is never re-executed after graph capture/replay launch. Capture warmups restore buffer, gradient, runtime-memory, mode, RNG, and unchanged-parameter state without performing a post-launch physical-format query.

Real fixed-shape NPUGraph is qualified for the previously documented AsPy IF, LIF, and PLIF paths. The new general bucket machinery currently has CPU static-buffer replay tests, but it still requires a physical CANN/torch-npu canary for each newly claimed model path. Separate train/eval captures are required. Models must declare graph-safe per-forward state or callers must explicitly opt in after verifying it. Training capture is disabled by default; the qualified opt-in path requires deterministic algorithms unless the expert-only override is used. Higher-order differentiation and arbitrary dynamic shapes are unsupported.

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

This intentionally shadows an installed but not-yet-imported real SpikingJelly package for the documented allowlist. It refuses partial replacement after real SpikingJelly has already been imported. The equivalent environment-gated bootstrap is `SPIKINGJELLY_NPU_COMPAT=spikingjelly`; `SPIKINGJELLY_NPU_COMPAT=ascend` additionally enables torch-npu's official CUDA convenience transfer. See [`docs/fedsnn-integration.md`](docs/fedsnn-integration.md).

The FedSNN AlexNet-BNTT integration backend `packed_aspy` combines `[T*N,...]` ANN packing with the exact stateless decay-LIF AsPy operator. It has passed the qualified Ascend 910B4/CANN 8.5 actual-model gates for full batch 128, remainder 42, train/eval, diagnostic eager fallback, six native LIF routes, BNTT buffers, two SGD updates, and two real trainer smokes. The accepted actual-shape gradient gate is `rtol=5e-5, atol=3e-5`; this is numerical-tolerance equivalence, not bitwise equivalence.

After building and activating AsPy, enable it explicitly in a consumer application that implements the documented six-layer adapter. These YAML keys are from the qualified FedSNN integration seam, not configuration parsed by `spikingjelly_npu` itself:

```yaml
model:
  name: fedsnn_alexnet_bntt
  execution_backend: packed_aspy
  execution_backend_strict: true
```

Strict mode is recommended for qualification and deployment because the consumer must reject silent fallback unless all six spiking layers use native AsPy in both train and eval. See the [FedSNN integration guide](docs/fedsnn-integration.md) for the adapter contract, route-diagnosis example, fallback, physical-format, diagnostic, and identity rules.

For CIFAR-10 Dirichlet α=0.3 seed-2 client-0, T=4, batch 128, LE=5, 1706 samples, five balanced fresh-process measurements produced median complete-client times of 5.3214 s for `legacy_stepwise`, 5.0087 s for `packed_eager`, and 4.3882 s for `packed_aspy`. Thus `packed_aspy` reduced median wall time by 12.4% versus `packed_eager` and 17.5% versus `legacy_stepwise`. Every measured AsPy run used 420 native LIF calls with no fallback. This qualifies the execution path for explicit adoption on the tested stack; it is not a federated convergence or multi-seed accuracy claim. Existing running experiments must retain their recorded code/config identity, and `npugraph` remains unqualified for this formal shape. These were externally collected FedSNN qualification results; audit the checked-in [`docs/evidence/packed_aspy_manifest.json`](docs/evidence/packed_aspy_manifest.json) and raw summary JSON, and see [`docs/performance.md`](docs/performance.md) for the exact identity and reproduction boundary.

## Compatibility boundaries

- activation-based sequences are time-major `[T,N,...]`; standard `torch.nn` wrappers also preserve `batch_first` and `PackedSequence` behavior;
- the standard recurrent decomposition is selected only for actual FP32 NPU input storage; it is not a `DynamicGRUV2`, speedup, graph, arbitrary dynamic-shape, FP16, or BF16 claim;
- spiking recurrent modules accept dense fixed-length sequences only; ragged/packed inputs and projected spiking LSTM are deferred;
- neuron and explicitly stateful recurrent state persists until `reset()` / `reset_net()`;
- Spikformer factory compatibility does not imply universal external checkpoint-key compatibility;
- recurrent TorchScript is not part of the current semantic-alpha claim;
- the SpikingSelfAttention/Spikformer `rtol=2e-4, atol=5e-4` addendum applies only to continuous BaseNode final states in tiny-shape FP32 eager parity; public outputs/logits, loss, all gradients, optimizer updates, state dict, and any discrete-spike exactness gate are unchanged;
- tiny-shape Spikformer forward evidence does not qualify training/backward, optimizer trajectories, convergence, native routing, representative shapes, or acceleration;
- `v` and `v_seq` are not state-dict entries;
- KLIF stores scalar parameter `k`; PLIF stores scalar parameter `w`;
- `backend="torch"` works on CPU or NPU tensors;
- `backend="npu"` is an explicit alias for ordinary PyTorch operators on NPU;
- `backend="aspy"` is the optional IF/LIF/KLIF/PLIF Ascend route, not a CuPy array backend;
- for compatible SpikingJelly constructors, `backend="cupy"` is accepted only as an observable preference alias for AsPy with eager fallback; no CuPy array/kernel API is implemented;
- CUDA CuPy kernels, Triton CUDA kernels, custom CUDA extensions, and the full SpikingJelly catalog are intentionally unsupported.

See [`docs/compatibility.md`](docs/compatibility.md) for the full contract.

## Current component evidence

All AsPy numbers below are component-level measurements on Ascend 910B4 `npu:7`, CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32. They are not formal FedSNN end-to-end results.

- **IF three paths:** fixed `[B,T,F]=[64,8,4096]`; full reset + gain + IF + linear readout + MSE + backward; 10 warmups, 50 measured iterations, synchronize after warmup and every measured iteration, three fresh processes. Median-of-run-medians: PyTorch 9.241211 ms, AsPy eager 2.685313 ms, AsPy NPUGraph 2.056798 ms. Median run speedups: 3.401488× eager and 4.606828× graph. All graph routes were true NPUGraph; output/loss were exact and maximum gain-gradient absolute error was `9.313225746e-10`.
- **PLIF:** fixed `[T,B,F]=[8,64,4096]`; forward + spike/final-state loss + backward for input and `w`; 10 warmups, 50 measured iterations, same synchronization policy, three fresh processes. Run speedups were 5.320089×, 6.337813×, and 4.311383×; median 5.320089×. Three fresh determinism hashes, three 20-step optimizer trajectories, and five dynamic-`w` true-NPUGraph replays passed.
- **LIF:** fixed-tau forward/backward, carried state, strict/fallback, and true fixed-shape NPUGraph are covered by the full real-NPU AsPy suite. No standalone LIF performance number is claimed.
- **KLIF:** real-NPU qualification covers full/remainder/singleton shapes, hard/soft reset, both `detach_reset`, `decay_input`, and `scale_reset` values, carried state, input gradient, and scalar `k.grad` at `rtol=5e-5, atol=3e-5`. No KLIF speed or NPUGraph claim is made.
- **Full AsPy NPU suite:** see the current CANN 8.5 verification evidence; older PLIF-only snapshot counts do not include KLIF.

- **FedSNN AlexNet-BNTT `packed_aspy`:** Ascend 910B4, CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32, CIFAR-10 Dirichlet α=0.3 seed-2 client-0, T=4, batch 128, LE=5. Five balanced fresh-process measurements, each after one complete-client warmup epoch and synchronized once before/after the measured client, gave medians `legacy_stepwise=5.3214 s`, `packed_eager=5.0087 s`, `packed_aspy=4.3882 s`. The AsPy route was native for all six spiking layers and all 70 measured forwards per run. Actual-model gradients passed at the accepted `rtol=5e-5, atol=3e-5`; this is tolerance equivalence, not bitwise equivalence.

An early scalar, one-core, host-synchronized IF prototype was approximately 189× slower than PyTorch. That result is retained only as historical rejection evidence and does not describe the current vectorized multi-block task-queue implementation.

See [`docs/performance.md`](docs/performance.md) for exact methodology and qualification boundaries, and [`docs/cann-8.5.md`](docs/cann-8.5.md) for deployment.

## Repository policy

Keep virtual environments, build outputs, wheel staging, compiler caches, profiler output, benchmark JSON, logs, datasets, checkpoints, and experiment results outside this source tree.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
