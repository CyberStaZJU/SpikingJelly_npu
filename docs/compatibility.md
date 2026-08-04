# Compatibility contract

## Target

The stable native contract still covers the activation-based subset required by FedSNN and common feedforward SNNs. Its historical compatibility baseline is SpikingJelly `0.0.0.0.14`, the version pinned by the current FedSNN lockfile; the KLIF gap audit additionally checked upstream source snapshot `f67935114ab178300be623297b41adef6727622f`.

The unreleased semantic alpha adds standard and spiking recurrent/Transformer APIs. Their external semantic audit used current SpikingJelly `master` commit `6de16e441f60e37fce28bc9e6b11ac25039ee239` and standard PyTorch APIs. The qualified accelerator stack remains Ascend 910B4, CANN 8.5.0, Python 3.10, PyTorch 2.9.0, and torch-npu 2.9.0. Physical CANN 8.5 qualification found the fused standard GRU route rejects FP32, so standard FP32 NPU recurrent availability uses eager primitive decomposition. The active precision direction is BF16 mixed precision: qualified Conv/Linear/MatMul and public activations use BF16, while normalization, recurrent/spiking state, reductions, master parameters, gradients, and optimizer state remain FP32. LayerNorm and Softmax are FP32 targets, but current standard Transformer internals remain delegated to torch-npu pending explicit isolation qualification. The 84-case tiny-shape BF16 eager matrix passed across standard recurrent and Transformer APIs, project spiking recurrent modules, SpikingSelfAttention, and Spikformer. These are compatibility and dtype-contract—not performance or convergence—results; FP8 remains unqualified.

## Supported APIs

| Module | Supported |
|---|---|
| `base` | `StepModule`, `SingleStepModule`, `MultiStepModule`, `MemoryModule`, memory traversal |
| `neuron` | `BaseNode`, `IFNode`, `LIFNode`, `KLIFNode`, `ParametricLIFNode` |
| `surrogate` | `heaviside`, `Sigmoid`, `ATan`, `PiecewiseQuadratic`, `SoftSign`, `SuperSpike` |
| `functional` | reset/detach/configuration and sequence helpers |
| `layer` | Linear, Conv1d/2d/3d, BatchNorm1d/2d/3d, max/average/adaptive-average pooling, Flatten, VotingLayer, SeqToANNContainer, `SpikingSelfAttention` |
| `sequence.recurrent` | Direct `torch.nn.RNN`, `GRU`, and `LSTM` subclasses with standard dense/`PackedSequence` semantics and a primitive FP32 NPU compatibility fallback |
| `sequence.transformer` | Eager `MultiheadAttention`, encoder/decoder layers and stacks, and top-level `Transformer` |
| `activation_based.recurrent` | Project-defined `SpikingRNN/GRU/LSTM` cells and dense fixed-length sequence wrappers |
| `activation_based.model.spikformer` | Eager patch stem, blocks, classifier, and `spikformer_ti` / `spikformer_s` presets |
| `npu.graph` | Legacy `StaticGraphRunner` and bounded exact-signature `GraphBucketRunner` |

Support levels differ: standard recurrent, standard Transformer, project spiking recurrent modules, SpikingSelfAttention, and Spikformer have bounded tiny-shape BF16 eager train/eval qualification. Spikformer additionally has shape-specific representative evidence at `T=4`, batch `64`, `224x224`, embedding `384`, six heads, and four blocks: three seeds × twenty finite optimizer steps, BF16-vs-FP32 training latency/HBM measurements, and exact fixed-shape evaluation NPUGraph replay. A batch-8 fixed-synthetic-batch 200-update run is an optimization-health canary only. Existing neuron AsPy capabilities retain their independently qualified native status through a BF16-public/FP32-native island. No dataset convergence, native Spikformer operator, training NPUGraph, FP8, arbitrary-variant, or family-wide speed qualification is implied. See `sequence-acceleration-contract.md` for the exact boundary.

## Semantics

- Single step uses `[N, ...]`; multi-step uses `[T, N, ...]`.
- Heaviside fires at `x >= 0`.
- IF charge: `v = v + x`.
- LIF charge:
  - `decay_input=True`: `v += (x - (v - v_reset)) / tau`.
  - otherwise: `v -= (v - v_reset) / tau; v += x`.
- Hard reset uses a float `v_reset`; soft reset uses `None`.
- `detach_reset` detaches only the spike used by reset.
- `store_v_seq=True` stores post-reset voltage for every timestep.
- Runtime voltage memories are not persistent state-dict entries.
- Standard recurrent/Transformer wrappers preserve PyTorch constructor, parameter, state-dict, mask, state, dropout, and eager first-order-autograd semantics.
- Standard RNN/GRU/LSTM calls use a functional primitive decomposition only when the actual input tensor is FP32 on `npu:*`. Dense and packed, unbatched/batched, multilayer/bidirectional, PyTorch GRU reset placement, projected LSTM, supplied/default state, sort/unsort, gradients, and parameter identity remain in scope. All other devices/dtypes delegate immediately to upstream PyTorch.
- Spiking recurrent modules use dense fixed-length `[T,N,F]` (or `batch_first`) inputs, PyTorch-style top-level recurrent parameter names, and non-persistent optional carry state. Their equations are project-defined and versioned in `sequence-acceleration-contract.md`.
- `SpikingSelfAttention` uses `[T,N,C,L]` and the exact softmax-free `(V @ K^T) @ Q * 0.125` order. Spikformer accepts 4D images or explicit 5D time-major image sequences.
- Spikformer presets claim architecture/configuration compatibility only; strict loading of arbitrary external checkpoint keys/layouts is not claimed.
- The user-approved `rtol=2e-4, atol=5e-4` addendum remains limited to continuous BaseNode final states in the historical FP32 parity evidence. It does not relax BF16 public outputs/logits, loss, gradients, optimizer updates, state dict, finite-value checks, or discrete-spike gates.
- KLIF applies `h = relu(k * h_pre)` after ordinary LIF charge, fires from `h`, optionally resets using `h / k` and `v_threshold / k` when `scale_reset=True`, and exposes learnable checkpoint key `k` initialized to `1.0`.
- PLIF initializes `w = -log(init_tau - 1)`, uses `sigmoid(w)` as reciprocal tau, and exposes checkpoint key `w`.

## Intentional extensions

- `backend="npu"` is an explicit alias for pure PyTorch operators on an NPU device.
- `backend="aspy"` is available on multi-step `IFNode`, fixed-tau `LIFNode`, learnable-`k` `KLIFNode`, and `ParametricLIFNode`. It requests the optional fused Ascend C backend; it is not a CuPy-compatible array API.
- On those replacement neurons, `backend="cupy"` is accepted as a migration preference alias and normalized to the same observable AsPy request. Unsupported/native-missing cases follow the normal strict/fallback policy. Actual CuPy imports, arrays, raw kernels, or CUDA execution are not emulated.
- The AsPy native ABI accepts non-empty, contiguous, storage-offset-zero FP32 time-major tensors on an Ascend NPU with a spiking `surrogate.ATan`. Public tensors may be FP32 or BF16. BF16 uses an explicit FP32 native island: input is converted and physical ND format is revalidated before extension loading, voltage state remains FP32, and the public spike sequence is cast back to BF16. LIF and KLIF additionally require a fixed Python float `tau > 1`. KLIF keeps scalar `k` FP32 and dynamic without changing the public parameter shape; PLIF transports its one-element contiguous FP32 reciprocal-tau tensor as a device input, so `w` remains an FP32 master parameter. Configure AsPy before dtype conversion and create the optimizer only after backend/dtype setup; a late backend switch with a non-FP32 `k`/`w` is rejected before extension loading, so existing optimizer state cannot be silently left at BF16.
- AsPy supports hard/soft reset, both `detach_reset` values, LIF/KLIF/PLIF `decay_input`, both KLIF `scale_reset` values, optional FP32 `v_seq`, carried-state gradients, BF16 or FP32 input gradients matching the public input dtype, FP32 KLIF `k.grad`, and FP32 PLIF `w.grad` in the qualified scope. BF16 route records use `dtype_conversion="bf16-public-fp32-aspy-island"` and estimated conversion traffic.
- Unsupported or unavailable requests fall back before native execution when `backend_strict=False`. With `backend_strict=True`, the same condition raises `AsPyBackendError`. Once a native function has been invoked, failures propagate and are not silently replayed through a stateful eager step. `last_backend_route` records the implementation and reason.
- Step-aware `Linear` explicitly packs `[T, N]` through `seq_to_ann_forward`; this is numerically equivalent to N-dimensional `nn.Linear` and makes the NPU execution policy explicit.
- BatchNorm1d multi-step accepts both `[T,N,C]` and `[T,N,C,L]`, matching modern PyTorch usefulness rather than preserving the old release's restrictive shape check.
- `MemoryModule.reset()` restores no-grad/eval tensor memories in place where safe, helping static graph address stability.

## AsPy and NPUGraph

The bridge submits ACLNN work through torch-npu `OpCommand::RunOpApiV2` on the current NPU stream. It does not call `torch.npu.synchronize`, `aclrtSynchronizeStream`, or an equivalent synchronization in the forward/backward hot path. Workspace and ACL tensor/executor owners are retained through the task-queue callback.

Real fixed-shape NPUGraph capture/replay is qualified for AsPy IF, LIF, and PLIF. PLIF replay reads reciprocal tau from a device tensor and has been tested while changing `w` and input across five replays, including forward, input-gradient, and `w.grad` parity. KLIF is qualified for native eager multi-step execution only; no KLIF NPUGraph claim is made. Normal graph restrictions still apply: fixed argument structure, shape, dtype, device, layout and `requires_grad` state; separate train/eval captures; and deterministic algorithms for the default qualified training-capture policy.

## Unsupported

- Recurrent speedup claims for the FP32 NPU primitive fallback. It is an availability path, does not use `DynamicGRUV2`, and has no graph, arbitrary dynamic-shape, or compiled-execution claim.
- A custom FP16/BF16 fused recurrent implementation. Non-FP32 public storage still delegates upstream unchanged. Under `npu_bf16_autocast()`, standard FP32-storage recurrent modules retain the eager compatibility decomposition: affine operators may autocast to BF16 while recurrent gates/state stay FP32, avoiding the unavailable FP32 `DynamicGRUV2` path. Project spiking recurrent gates/state also use FP32 sensitive islands. Physical family qualification remains required.
- Standard Transformer native or speed claims: the completed 18/18 physical cases are eager torch-npu parity only, with no distinct native candidate or five-process performance qualification.
- Spikformer claims beyond the frozen `T=4`, batch-64, `224x224`, embedding-384, six-head, four-block BF16 eager training shape and the explicitly tested batch-8/batch-64 evaluation graph buckets. Dataset convergence/generalization, custom native execution, training NPUGraph, FP8, arbitrary variants, and family-wide acceleration remain unsupported. The batch-8 200-update fixed-synthetic-batch canary is not an accuracy result, and the batch-64 evaluation graph did not pass the latency gate.
- TorchScript compatibility for the standard recurrent subclasses in this semantic alpha; support differs across PyTorch versions.
- `PackedSequence`/ragged inputs for project-defined spiking recurrent modules, projected spiking LSTM, arbitrary dynamic-shape graphs, and incremental Transformer KV cache.
- Universal external Spikformer checkpoint-key/layout compatibility; an explicit converter and frozen external fixture would be required.
- General CuPy-compatible arrays, CUDA kernels, or Triton CUDA kernels.
- Native AsPy single-step execution. Single-step requests use the observable PyTorch fallback when non-strict; strict single-step KLIF raises `AsPyBackendError` because the requested native route is unavailable. IF/LIF/PLIF retain their documented strict-compatible single-step eager path.
- FP16 AsPy, true BF16-native AsPy kernel I/O/arithmetic, non-contiguous/storage-offset inputs, non-ATan surrogates, non-spiking surrogate mode, empty time dimensions, or arbitrary dynamic-shape NPUGraph replay. BF16 public tensors are supported only through the observable FP32 AsPy island described above.
- Higher-order AsPy gradients. The native IF, LIF, KLIF, and PLIF paths support first-order ATan-surrogate differentiation only.
- Full SpikingJelly layer/neuron catalog.
- Whole-model pickle compatibility across module paths.
- Higher-order gradients through NPUGraph.

## Backend naming

`backend="torch"` means the implementation uses PyTorch operations. It does not mean CPU-only. When the module and tensors are on `npu:*`, those operations dispatch through torch-npu.

Direct unsupported backend assignment raises `NotImplementedError`. `functional.set_backend` warns and keeps the existing backend when a module cannot support the requested value.
