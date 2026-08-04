# Sequence and model acceleration contract

## Status and reference identities

This document freezes the **semantic-alpha** contract for the first RNN and Transformer expansion of `spikingjelly_npu`. It is a correctness and routing contract, not a new native-performance claim.

- Downstream implementation baseline: `SpikingJelly_npu` commit `289c2ac82c759da0a01e7fd798972cb17f2f6e9b` plus the changes described here.
- External API/semantic audit reference: SpikingJelly `master` commit `6de16e441f60e37fce28bc9e6b11ac25039ee239` (observed 2026-08-03).
- Standard dense sequence authority: the installed PyTorch `torch.nn` implementations and, for physical qualification, torch/torch-npu 2.9.0.
- First physical target: Ascend 910B4, CANN 8.5.0, CPython 3.10, torch/torch-npu 2.9.0, with BF16 mixed precision and FP32 stability/master state.

The active precision contract uses BF16 for qualified Conv/Linear/MatMul and public activations. LayerNorm, Softmax, normalization, recurrent and spiking state, surrogate math, reductions, final logits, master parameters, gradients, and optimizer state remain FP32. AsPy accepts BF16 public tensors through its unchanged FP32 native ABI. FP8 is not enabled and requires an independent capability, autograd, numerical, and performance qualification. The 84-case tiny-shape physical matrix passed across standard recurrent and Transformer layers/stacks/top-level APIs, project spiking recurrent modules, SpikingSelfAttention, and Spikformer. Any native, graph, memory, latency, convergence, representative-shape, or family-wide claim remains capability- and shape-specific.

## Provider and failure model

The sequence work uses four logical provider requests:

- `torch`: authoritative eager PyTorch semantics on CPU or any supported PyTorch device;
- `vendor`: an exact torch-npu/CANN recurrent or attention implementation, when independently qualified;
- `aspy`: a versioned custom Ascend C capability, initially limited to narrow temporal/state scans;
- `auto`: select only a provider that has passed the relevant correctness and performance gates, otherwise use `torch`.

Provider selection must be observable. A structured route records the logical operation, requested and actual provider, stable reason code, train/eval mode, ABI/schema, native region, exact graph bucket, format conversion, and whether a native launch began.

All validation and unsupported-capability rejection occurs before native or graph launch. Once a native launch or graph replay begins, any failure is fatal for that call; the implementation must not rerun the stateful operation through eager PyTorch or another native ABI. Public state is committed only after all returned tensors have passed shape, dtype, device, layout, finite-scalar, and physical-format validation.

Ordinary package import must not explicitly import or initialize `torch_npu`.

## Standard recurrent API

`spikingjelly_npu.sequence` exposes direct subclasses of:

- `torch.nn.RNN`;
- `torch.nn.GRU`;
- `torch.nn.LSTM`.

The wrappers override `forward` only to select a narrow FP32 NPU compatibility path. They preserve the standard constructor, direct bases, exact parameter objects and names, state-dict layout, dense and `PackedSequence` inputs, unbatched/batched inputs, `batch_first`, bias, layers, directions, inter-layer dropout, RNN `tanh`/`relu`, PyTorch GRU gate/reset placement, ordinary and projected LSTM semantics, supplied/default and sorted/unsorted states, and eager first-order gradients.

When the actual dense tensor, or `PackedSequence.data`, has `device.type == "npu"` and dtype `torch.float32`, the wrapper bypasses the fused recurrent call and uses eager PyTorch primitives. Each layer/direction precomputes the input affine transform, while the temporal hidden recurrence is reconstructed functionally without in-place state slices. This is the CANN 8.5 FP32 availability fallback: the physically examined fused GRU route (`DynamicGRUV2`) rejected FP32, so this path does not use or claim `DynamicGRUV2`. Every other device or storage dtype delegates immediately to the upstream `torch.nn` `forward`. Under `npu_bf16_autocast()`, the same FP32-storage decomposition remains active: affine `Linear` operators may autocast to BF16, then gate inputs, supplied state, hidden state, and cell state are promoted to FP32 before nonlinear recurrence. Parameters, gradients, optimizer state, outputs, and final recurrent state remain FP32 by contract. This source-level dispatch rule does not itself qualify physical BF16 execution.

This decomposition is a compatibility and availability path, not a recurrent acceleration claim. It does not establish graph capture, arbitrary dynamic-shape execution, higher-order autograd, or a new native recurrent provider. A future vendor/native route may only replace an operation when all relevant gate order, state, reserve, dropout, bidirectional, projection, and backward semantics are exact.

Physical small-shape FP32 qualification passed 18/18 RNN/GRU/LSTM cases across train/eval and full/remainder/singleton batches. The worst maximum absolute differences were `1.1921e-7` for output/final state, `9.3132e-9` for input gradient, `1.1176e-8` for supplied initial-state gradient, `5.9605e-8` for parameter gradient, and `0` for optimizer/state update. The tested snapshot was explicitly dirty and is bound by base commit `08b7826`, patch SHA-256 `c1450bd4...`, and evidence SHA-256 `72abb1bc...`; this supports availability/parity only.

### TorchScript boundary

TorchScript is **not** part of this semantic-alpha compatibility claim. The wrappers re-declare PyTorch-style dense and `PackedSequence` overload metadata on each overridden `forward`, but TorchScript execution is still outside this claim. Some supported PyTorch releases reject scripting a recurrent subclass with `Overloads are not usable when a module is redeclared`, and the Python fallback itself is not a separately qualified scripted implementation. Eager, `PackedSequence`, state-dict, gradient, and optimizer semantics remain tested independently. Any future TorchScript support requires a version-specific qualification rather than a broad skip being counted as evidence.

## Project-defined spiking recurrent API

`spikingjelly_npu.activation_based.recurrent` exposes:

- `SpikingRNNCell`, `SpikingGRUCell`, `SpikingLSTMCell`;
- `SpikingRNN`, `SpikingGRU`, `SpikingLSTM`.

These are project-defined extensions. They are not presented as a unique upstream SpikingJelly recurrent definition.

All sequence wrappers accept dense fixed-length tensors only. Inputs are `[T,N,F]` or `[N,T,F]` when `batch_first=True`. `PackedSequence`, ragged input, zero-length sequences, LSTM projection, and arbitrary dynamic-shape native execution are outside the spiking recurrent alpha contract.

The frozen equations are:

```text
SpikingRNN:
  h_t = S(W_ih x_t + b_ih + W_hh h_(t-1) + b_hh)

SpikingGRU (PyTorch-efficient reset placement):
  r_t = S1(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)
  z_t = S1(W_iz x_t + b_iz + W_hz h_(t-1) + b_hz)
  n_t = S2(W_in x_t + b_in + r_t * (W_hn h_(t-1) + b_hn))
  h_t = (1 - z_t) * n_t + z_t * h_(t-1)

SpikingLSTM:
  (i_t, f_t, g_t, o_t) = split(W_ih x_t + b_ih + W_hh h_(t-1) + b_hh)
  i_t = S1(i_t); f_t = S1(f_t); g_t = S2(g_t); o_t = S1(o_t)
  c_t = clamp_max(f_t * c_(t-1) + i_t * g_t, 1)
  h_t = o_t * c_t
```

`S`, `S1`, and `S2` are configurable surrogate-spike modules and default to `surrogate.ATan`. There is no `tanh(c_t)` in the LSTM output equation.

Top-level recurrent parameter names follow PyTorch's `weight_ih_l*`, `weight_hh_l*`, `bias_ih_l*`, `bias_hh_l*`, and `_reverse` convention. Runtime carry state is non-persistent and does not enter the state dict.

Two state modes are explicit:

- passing `hx` makes the call stateless with respect to module memory;
- `stateful=True` carries the returned state only when `hx` is omitted, and only for unidirectional modules.

`reset()` clears carried state. `detach()` detaches it for truncated backpropagation. A changed batch shape, dtype, or device requires a reset rather than implicit state conversion.

Physical small-shape FP32 qualification passed 18/18 project-defined spiking RNN/GRU/LSTM cases across train/eval and full/remainder/singleton batches. Visible spike/output and final-state maximum absolute differences were zero; the worst input-gradient, initial-state-gradient, parameter-gradient, and update differences were `1.7462e-10`, `1.4552e-11`, `2.9802e-8`, and `4.6566e-10`. This is eager torch-npu semantic evidence only, not custom-native or performance qualification.

## Standard attention and Transformer API

`spikingjelly_npu.sequence` exposes eager subclasses of:

- `MultiheadAttention`;
- `TransformerEncoderLayer` and `TransformerDecoderLayer`;
- `TransformerEncoder` and `TransformerDecoder`;
- `Transformer`.

They preserve PyTorch parameter/state-dict namespaces and forward boolean/additive attention masks, key-padding masks, causal hints, self-attention, cross-attention, `batch_first`, `norm_first`, relu/gelu, dropout, train/eval, and first-order autograd semantics.

The semantic-alpha decoder is full-sequence teacher-forced execution. Incremental decoding, KV cache, native dropout, ragged/nested sequence acceleration, and arbitrary dynamic-shape graph capture are not claimed.

Under `npu_bf16_autocast()`, GEMM and projection operators may use qualified BF16 torch-npu/vendor Cube paths. LayerNorm is a conservative FP32 target, and Softmax is also targeted for FP32; however, the current standard Transformer route still delegates these internals to torch-npu until an explicit isolated route passes physical qualification. Parameters, gradients, optimizer state, and master weights remain FP32. A custom full attention implementation is not justified merely to add a native symbol, and FP8 GEMM is not enabled without a separate runtime and numerical qualification.

Physical tiny-shape BF16 qualification covers MultiheadAttention, encoder/decoder layers, encoder/decoder stacks built from upstream layers, and top-level Transformer in train/eval and full/remainder/singleton batches. This is eager torch-npu execution and dtype-contract evidence only; it does not qualify a custom native attention path, graph capture, representative shapes, convergence, FP8, or speed.

## SpikingSelfAttention and Spikformer

The canonical attention class is `spikingjelly_npu.activation_based.layer.SpikingSelfAttention`; `activation_based.transformer.SpikingSelfAttention` is the same class object.

Its public tensor convention is `[T,N,C,L]`. The frozen softmax-free kernel is evaluated in this order:

```text
q, k, v = split(project_and_spike(x))
a = (v @ transpose(k)) @ q
a = a * 0.125
output = project_and_spike(attention_spike(a))
```

The implementation must not silently replace this with standard softmax SDPA, change the two matrix-multiplication order, or collapse the token axis into an unrelated packed dimension.

The canonical model is `spikingjelly_npu.activation_based.model.spikformer.Spikformer`. `spikingjelly_npu.models.spikformer` is a backward-compatible alias. It provides a four-stage `/16` convolutional patch stem, residual positional encoding, residual SpikingSelfAttention/MLP blocks, per-timestep logits, 4D image repetition, and explicit 5D `[T,N,C,H,W]` input.

The `spikformer_ti` and `spikformer_s` factories claim architecture/configuration and tensor-semantic compatibility only. This alpha uses a combined QKV projection and local parameter namespaces. It does **not** claim strict checkpoint-key or tensor-layout compatibility with every external Spikformer repository. External checkpoints require a separately specified and tested conversion contract.

### Physical SpikingSelfAttention/Spikformer boundary

The active BF16 profile keeps Conv/Linear/MatMul and public spike activations in BF16 where qualified. BatchNorm, membrane, threshold, surrogate/reset math, spatial and temporal reductions, final logits, master parameters, gradients, and optimizer state remain FP32. SpikingSelfAttention is softmax-free, so the general Transformer Softmax FP32 rule does not add an operation to this architecture.

The tiny-shape physical matrix covers SpikingSelfAttention and Spikformer in train/eval and full/remainder/singleton batches as part of the 84-case BF16 suite. Public SSA spikes are BF16; persistent node state and Spikformer logits are FP32. The historical user-approved `rtol=2e-4, atol=5e-4` exception remains confined to continuous BaseNode final-state comparison and does not relax BF16 outputs, loss, gradients, updates, or discrete spikes.

A separate representative qualification freezes Spikformer at `T=4`, batch `64`, `224x224`, embedding `384`, six heads, and four blocks. Three seeds completed twenty BF16 optimizer steps with finite losses and FP32 parameters, gradients, and optimizer state. A bounded 200-update fixed-synthetic-batch canary at batch `8` reduced its first-20 mean loss from `2.1058` to a last-20 mean of `8.433e-4`; this is optimization-health evidence, not dataset convergence or generalization. Five alternating fresh-process training measurements at the frozen batch-64 shape found BF16 `1.0808x` faster by median latency, with `15.7%` lower peak allocated HBM and `18.8%` lower peak reserved HBM than FP32. The same policy was slower at batch `8`, so the performance claim is shape-specific.

Fixed-shape BF16 evaluation NPUGraph capture and changed-input replay are exact against eager at batches `8` and `64`, with strict unknown-batch rejection. Batch `8` replay was `2.42x` faster in five fresh processes, but the frozen batch-64 replay was about `5%` slower than eager and therefore carries correctness—not acceleration—qualification. Training capture failed after launch and poisoned the runner without eager replay, so Spikformer training NPUGraph remains unsupported.

No custom native Spikformer operator is promoted. The first plausible future candidate is an exact Sigmoid-surrogate BF16-public/FP32-state multi-step LIF scan, but only if representative profiling proves the recurrence is launch-bound and physical-format conversion plus block/model gates pass. Native Spikformer execution, dataset convergence, arbitrary architecture variants, FP8, training NPUGraph, and family-wide acceleration remain unqualified. The package no longer exposes or requires a Conv HF32 configuration policy.

## Public namespaces and compatibility alias

The intended public namespaces are:

- `spikingjelly_npu.sequence`;
- `spikingjelly_npu.models`;
- `spikingjelly_npu.activation_based.recurrent`;
- `spikingjelly_npu.activation_based.transformer`;
- `spikingjelly_npu.activation_based.layer.SpikingSelfAttention`;
- `spikingjelly_npu.activation_based.model.spikformer`.

The opt-in process-local `spikingjelly` alias registers these modules canonically in `sys.modules`. Imports through the alias and through `spikingjelly_npu` must return the same module and class objects, independent of import order. The alias remains an allowlisted compatibility subset, not the complete upstream package.

## Exact-shape graph buckets

`GraphBucketRunner` admits a finite explicit allowlist of exact tensor PyTree signatures. The signature covers:

- positional/keyword tree structure, caller keyword insertion order, and immutable static values;
- tensor shape, dtype, device, layout, `requires_grad`;
- stride, storage offset, contiguous memory-format classification, and the runtime-reported Ascend physical format;
- alias/storage groups;
- train/eval state, deterministic-training policy, module structure, parameter and buffer identity/version.

If the Ascend physical format cannot be inspected before capture or replay, that
call is rejected before graph execution: non-strict routing performs one observable
eager call, while strict routing raises `GraphPreExecutionError` without invoking
the model. No format query is deferred until after graph launch.

The default maximum is eight declared buckets. Unknown signatures use observable eager fallback or strict pre-execution failure. `StaticGraphRunner(strict=True)` applies the same pre-execution rule to every known rejection before capture/replay: it raises `GraphPreExecutionError` carrying the rejected `GraphRoute` and never calls the eager model. Non-strict mode keeps the compatibility fallback only while graph execution is known not to have started. Entry into `torch.npu.make_graphed_callables` is the capture-launch boundary: any exception after that point poisons the whole runner and never triggers eager replay, even in non-strict mode. Parameter object, storage, version, and bitwise value are snapshotted before launch; capture-time mutation is restored where safely possible and then fails closed. Buffer, gradient, runtime-memory, mixed training-mode, CPU/NPU RNG, and module-structure cleanup failures likewise poison the runner. Replay failure after launch also poisons the runner. No post-launch validation re-queries an Ascend physical format.

CPU tests use a static-buffer replay test double to check changed-input copying, nested PyTree reconstruction, aliases, and poisoning. A separate tiny fixed-shape FP32 evaluation canary on NPU7 passed 11/11 checks, including runtime physical format, exact signature rejection, poisoning, and no eager replay after capture/replay launch. That canary does not qualify standard recurrent, Transformer, Spikformer, training capture, arbitrary dynamic shapes, or any undeclared bucket; every newly claimed model/bucket still needs its own physical evidence.

## Compact IF/LIF capability boundary

The compact IF/LIF ABI is integrated as two additive ABI-1 capability groups, `if_compact` and `lif_compact`, for `store_v_seq=False`:

- public forward returns spikes and final voltage while retaining only the private `h_seq` history required by first-order backward;
- backward omits the public full `grad_v_seq` input path while preserving input and initial-voltage gradients;
- compact execution is selected only when the complete forward/backward capability pair is declared by a valid versioned bundle, or inferred as a complete pair from an unversioned transitional bundle;
- valid versioned metadata is authoritative: undeclared raw compact symbols are ignored;
- partial, missing/non-callable, malformed, or unsupported-version compact capabilities fail closed before launch, while a complete legacy full-output IF/LIF pair remains usable;
- `store_v_seq=True` always uses the full-output ABI;
- compact launch or backward failure is fatal and must not retry legacy or eager execution;
- route metadata exposes the executed native region as `if_compact` or `lif_compact`.

CPU and fake-native integration, manifest generation, packaging coverage, padding/cropping, and first-order failure semantics are implemented. Physical CANN 8.5 compilation plus full/remainder/singleton correctness and post-launch no-replay checks passed. In five fresh processes at `[T,B,F]=[8,64,4096]`, every IF and LIF worker reported zero maximum absolute difference for loss, final voltage, input gradient, and initial-voltage gradient versus the full-output native ABI. The performance promotion gate failed: IF full/compact median speedup was `1.0317×` and LIF `1.0093×`; peak allocated HBM fell only `10.0%` for each. Both are below the frozen `1.25×` speed or `20%` memory alternative. Compact capabilities remain compatible and callable when explicitly requested by capability metadata, but must not be advertised or auto-promoted as a performance optimization. Existing full-output IF/LIF bundles remain compatible.

## Unsupported or deferred in this alpha

- custom fused FP16/BF16 recurrent kernels; outside the explicit BF16 profile those dtypes delegate upstream unchanged, and inside it standard recurrent modules still delegate operator policy to torch-npu;
- BF16 family-wide claims before physical train/eval, batch-shape, trajectory, convergence, latency, and HBM qualification;
- AMP-native sequence kernels;
- graph capture, compiled/dynamic recurrent execution, or arbitrary dynamic-shape claims for the FP32 NPU recurrent fallback;
- higher-order native or graph gradients;
- arbitrary dynamic shapes or unbounded graph caches;
- native recurrent/attention dropout;
- spiking recurrent `PackedSequence` or ragged input;
- projected spiking LSTM;
- incremental decoder KV cache;
- universal external Spikformer checkpoint loading;
- family-wide acceleration inferred from one primitive, one model, or one execution mode;
- revival of retired FedSNN SHD, Idea A/C, AFedSNN, or asynchronous consumers.

## Evidence levels

Evidence must be reported separately:

1. CPU eager semantics and fake-native routing;
2. CANN host/kernel compilation and capability import;
3. physical NPU forward/backward/update parity;
4. primitive hotspot timing/memory;
5. complete block timing/memory;
6. representative model timing/memory and short trajectory.

Passing a lower level does not imply a higher one. Numerical comparisons are tolerance equivalence, never bitwise equivalence unless exact equality is explicitly tested.
