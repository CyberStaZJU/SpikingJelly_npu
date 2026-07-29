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
- eager and NPUGraph routes.

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

## What this repository does not change

It does not modify an application repository, active configurations, running queues, experiment results, datasets, or the machine-wide CANN installation.
