# Sequence and model acceleration contract

## Status and reference identities

This document freezes the **semantic-alpha** contract for the first RNN and Transformer expansion of `spikingjelly_npu`. It is a correctness and routing contract, not a new native-performance claim.

- Downstream implementation baseline: `SpikingJelly_npu` commit `289c2ac82c759da0a01e7fd798972cb17f2f6e9b` plus the changes described here.
- External API/semantic audit reference: SpikingJelly `master` commit `6de16e441f60e37fce28bc9e6b11ac25039ee239` (observed 2026-08-03).
- Standard dense sequence authority: the installed PyTorch `torch.nn` implementations and, for physical qualification, torch/torch-npu 2.9.0.
- First physical target: Ascend 910B4, CANN 8.5.0, CPython 3.10, torch/torch-npu 2.9.0, FP32.

The public Python APIs are available before physical NPU qualification. Any native, graph, memory, or latency claim remains capability- and shape-specific until it passes the acceptance policy in `docs/evidence/sequence_acceptance_policy.json`.

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

The semantic-alpha wrappers intentionally do not override recurrent `forward`. They preserve the standard constructor, top-level parameter names, state-dict layout, dense and `PackedSequence` inputs, unbatched/batched inputs, `batch_first`, layers, directions, dropout, RNN `tanh`/`relu`, LSTM projection, initial/final states, and eager first-order gradients.

These wrappers currently request no custom recurrent kernel. On an NPU they execute through ordinary PyTorch/torch-npu dispatch. A future vendor/native route may only replace an operation when all relevant gate order, state, reserve, dropout, bidirectional, projection, and backward semantics are exact.

### TorchScript boundary

TorchScript is **not** part of this semantic-alpha compatibility claim. The wrappers retain PyTorch's inherited overload metadata, but some supported PyTorch releases reject scripting a recurrent subclass with `Overloads are not usable when a module is redeclared`. Eager, `PackedSequence`, state-dict, gradient, and optimizer semantics remain tested independently. Any future TorchScript support requires a version-specific qualification rather than a broad skip being counted as evidence.

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

## Standard attention and Transformer API

`spikingjelly_npu.sequence` exposes eager subclasses of:

- `MultiheadAttention`;
- `TransformerEncoderLayer` and `TransformerDecoderLayer`;
- `TransformerEncoder` and `TransformerDecoder`;
- `Transformer`.

They preserve PyTorch parameter/state-dict namespaces and forward boolean/additive attention masks, key-padding masks, causal hints, self-attention, cross-attention, `batch_first`, `norm_first`, relu/gelu, dropout, train/eval, and first-order autograd semantics.

The semantic-alpha decoder is full-sequence teacher-forced execution. Incremental decoding, KV cache, native dropout, ragged/nested sequence acceleration, and arbitrary dynamic-shape graph capture are not claimed.

Standard GEMM, projection, softmax, normalization, and attention primitives should remain on qualified torch-npu/vendor Cube paths when exact. A custom full attention implementation is not justified merely to add a native symbol.

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
- stride, storage offset, contiguous memory-format classification;
- alias/storage groups;
- train/eval state, deterministic-training policy, module structure, parameter and buffer identity/version.

The default maximum is eight declared buckets. Unknown signatures use observable eager fallback or strict pre-execution failure. `StaticGraphRunner(strict=True)` applies the same pre-execution rule to every known rejection before capture/replay: it raises `GraphPreExecutionError` carrying the rejected `GraphRoute` and never calls the eager model. Non-strict mode keeps the compatibility fallback. An exception from the capture attempt itself propagates unchanged; subsequent strict calls report a structured prior-capture rejection. Capture failure is isolated to that bucket in `GraphBucketRunner`. Failure to restore buffers, gradients, runtime memories, training state, RNG, or module structure poisons the runner. Replay failure after launch also poisons the runner and never triggers eager replay.

CPU tests use a static-buffer replay test double to check changed-input copying, nested PyTree reconstruction, aliases, and poisoning. This does not replace a physical CANN/torch-npu graph canary. No arbitrary dynamic-shape graph claim is made.

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

CPU and fake-native integration, manifest generation, packaging coverage, padding/cropping, and first-order failure semantics are implemented. The compact ABI is not physically qualified until it compiles on CANN 8.5 and passes physical NPU correctness, memory, and performance gates. Existing full-output IF/LIF bundles remain compatible.

## Unsupported or deferred in this alpha

- FP16, BF16, AMP-native sequence kernels;
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
