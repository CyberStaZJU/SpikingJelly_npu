# FedSNN integration

## Repository and experiment boundary

- Keep the application/FedSNN repository authoritative for its own code and research records.
- This standalone repository is authoritative only for `spikingjelly_npu`.
- Treat compute-host checkouts as disposable test snapshots unless your project defines otherwise.
- Keep datasets, formal results, logs, checkpoints, and generated native builds outside this source repository.

## Recommended adoption order

### 1. Compatibility canary

Use the existing SHD backend seam first. Replace:

```python
from spikingjelly.activation_based import functional, layer, neuron, surrogate
```

with:

```python
from spikingjelly_npu.activation_based import functional, layer, neuron, surrogate
```

The required constructor signatures, time-major tensors, state reset, and state-dict keys are supported.

If changing imports is temporarily undesirable:

```python
import spikingjelly_npu
spikingjelly_npu.enable_compat()
```

Call this before importing `fedsnn.shd_model`. On a dedicated Ascend process that still contains common CUDA convenience calls, `enable_compat(cuda=True)` also invokes torch-npu's official `transfer_to_npu` layer. It does not make arbitrary CuPy/CUDA extensions portable.

### 2. Real NPU parity

On an idle device, compare legacy/pure and NPU-library backends using identical state dicts and encoded inputs:

- logits;
- hidden spikes;
- input and parameter gradients;
- one optimizer update;
- reset repeatability;
- full and remainder batches;
- eager and any explicitly qualified accelerated routes. Do not use NPUGraph for the formal AlexNet T=4, batch=128 shape unless that shape is requalified.

### 3. MNIST BNTT model

Use `BNTT1d`, `BNTT2d`, and graph-safe `MultiStepLIF` to port the small image path. Preserve existing state-dict names in the FedSNN-side model adapter rather than renaming modules.

### 4. AlexNet+BNTT ordinary forward

Port only the ordinary logits path first. Keep diagnostic keyword paths eager. Wrap the final model with `StaticGraphRunner(batch_size=<actual full batch>)`.

### 5. Diagnostics and host synchronization

Only after ordinary training parity:

- compact pattern drift on device;
- defer `.item()` and `.cpu()` metrics;
- inspect full local-client and federated-round timing;
- add AMP if correctness and convergence gates pass.

## Critical semantic gates

- Preserve Poisson RNG policy. For strict comparisons, precompute `[T,N,C,H,W]` encoded inputs outside the captured network.
- Preserve independent per-timestep BNTT modules and running buffers.
- Preserve soft reset used by the active image models.
- Preserve exact ATan derivative and `detach_reset=True` behavior.
- Do not replace `Parameter` objects after graph capture. Load state into existing tensors.
- Capture train and eval paths separately.
- Before any opted-in training capture, call `torch.use_deterministic_algorithms(True, warn_only=False)`; warn-only mode is not qualified and the runner otherwise stays eager.
- Treat `require_deterministic_training=False` as an expert qualification override, not an optimization switch.
- Never route diagnostic kwargs through the ordinary graph silently.

## Qualified AlexNet-BNTT exact decay-LIF path

FedSNN's image model charges membrane in this exact order:

```text
charged = membrane * membrane_decay
charged = charged + current
spike = ATan(charged - threshold)
membrane = charged - spike.detach() * threshold
```

A generic fixed-tau `LIFNode` is not a threshold-exact substitution because its subtraction/division operation order differs. The dedicated stateless `spikingjelly_npu.fedsnn.DecayLIF` therefore preserves the application order, starts each public forward from zero membrane, returns the spike sequence, and performs the reverse-time ATan-surrogate recurrence in its optional AsPy backward.

FedSNN may expose this as an explicit `packed_aspy` backend: stateless Conv/Linear/pool still use `[T*N,...]` packing, while each layer's temporal decay-LIF scan is submitted as one qualified native forward/backward route. The native request remains FP32 and requires a non-empty contiguous storage-offset-zero NPU tensor plus a spiking `surrogate.ATan`. A BF16 public current sequence may enter an explicit FP32 AsPy island, after which the spike sequence is cast back to BF16 and route metadata records `bf16-public-fp32-aspy-island` plus estimated conversion bytes. Unsupported requests fall back before extension loading unless strict mode is requested. Diagnostic keyword calls stay on the authoritative eager model path. This mixed route is not a BF16-native decay-LIF kernel.

Qualification result on Ascend 910B4/CANN 8.5/torch-npu 2.9:

- native source generation, external build, extension import, and all eight expected forward/backward symbols: passed;
- CPU semantics, state-dict neutrality, import safety, old-bundle fallback, observable routing, and shipped integration tests: passed;
- physical-NPU forward/backward and actual AlexNet full=128, remainder=42, singleton, train/eval, BNTT-buffer, two-SGD-update, diagnostic-eager, and six-native-route gates: passed;
- actual-model gradients: passed at `rtol=5e-5, atol=3e-5`, explicitly as numerical-tolerance equivalence rather than bitwise equivalence;
- balanced complete-client benchmark: passed, with medians `legacy_stepwise=5.3214 s`, `packed_eager=5.0087 s`, and `packed_aspy=4.3882 s` across five fresh processes per backend;
- two consecutive exact-YAML real trainer smokes: passed, with finite metrics and six native AsPy routes in both train and eval.

`packed_aspy` is therefore qualified for explicit adoption on the tested stack and workload. It does not retroactively change running/completed experiment identities and is not a federated convergence claim. `npugraph` remains isolation-only for this workload because formal `T=4,batch=128` capture is not qualified.

### Application-side configuration

The `v0.1.0-alpha.1` release bundle predates this operator and cannot provide `packed_aspy`. Until a newer release explicitly advertises the FedSNN decay-LIF symbols, build and activate AsPy from the current source tree with `scripts/build_aspy.sh`. An older bundle remains importable, but strict mode rejects it and non-strict mode reports an observable fallback.

`packed_aspy` is a **consumer-application backend**, not a YAML parser implemented by this library. The public primitive here is `spikingjelly_npu.fedsnn.DecayLIF`; a consumer must insert it into the six temporal layers, aggregate `last_backend_route`, preserve diagnostic eager forwarding and state-dict keys, and enforce its own strict policy. The qualified adapter was evaluated against the FedSNN workspace based on Git HEAD `386c9a418f010dd5a90bac9519b78cdc2e708765`; those integration changes were not yet a clean public FedSNN commit at qualification time, so the configuration below is an integration example rather than a promise that every FedSNN checkout accepts these keys.

After building and activating a capable native bundle, configure a consumer that implements this seam explicitly:

```yaml
model:
  name: fedsnn_alexnet_bntt
  execution_backend: packed_aspy
  execution_backend_strict: true
```

Use strict mode for qualification, smoke, and deployments where silent loss of acceleration is unacceptable. It requires NPU execution and all six AlexNet spiking layers to report native AsPy routes in both training and evaluation. For exploratory compatibility, `execution_backend_strict: false` permits an observable `packed_aspy_fallback` before native execution when the extension, dtype, layout, surrogate, or physical format is unsupported.

The C++ bridge is ND-only. Real packed convolution/BNTT output can arrive as rank-5 `ACL_FORMAT_NCDHW`; the Python adapter reshapes and clones that tensor into fresh physical ND storage before native launch. Internal format 29 and other unsupported formats are rejected before extension loading. Diagnostic keyword forwards remain on the authoritative legacy eager path, while ordinary remainder batches are supported by the native stateless decay-LIF route.

### Route diagnosis in a consumer

`last_backend_route` is per `DecayLIF` call. A model-level `packed_aspy` adapter should collect one route for each of its six spiking layers on every ordinary forward:

```python
routes = tuple(layer.last_backend_route for layer in model.decay_lif_layers)
details = [
    {
        "index": index,
        "requested": route.requested_backend,
        "backend": route.backend,
        "accelerated": route.accelerated,
        "reason": route.reason,
    }
    for index, route in enumerate(routes)
]
all_native = len(routes) == 6 and all(
    route.requested_backend == "aspy" and route.backend == "aspy"
    for route in routes
)
if strict and not all_native:
    raise RuntimeError(f"expected six native decay-LIF routes: {details}")
```

Expected ordinary native entries have `backend="aspy"` and a reason naming the fused FedSNN decay-LIF kernel. BF16 mixed entries additionally report `dtype_conversion="bf16-public-fp32-aspy-island"` and `dtype_conversion_bytes`; this byte count estimates **forward boundary traffic only** (public inputs converted to FP32 plus FP32 spikes converted back to BF16). Consumers should aggregate it with measured backward and framework traffic rather than treating it as total training traffic. An old bundle, FP16 input, non-ATan surrogate, non-contiguous input, or unsupported physical format reports `backend="torch"` plus its pre-execution reason in non-strict mode; strict mode raises instead. Diagnostic forwards are a separate, intentional consumer-level `eager_diagnostic` route and should not overwrite or masquerade as an ordinary six-layer native result. Log tensor shape, dtype, device, and `torch_npu.get_npu_format(tensor)` together with these route records when diagnosing fallback.

## What this repository does not change

It does not modify active configurations, running queues, experiment results, datasets, or the machine-wide CANN installation. Application-side adoption must remain explicit and must follow that application's own identity and experiment-lifecycle rules.
