# Compatibility contract

## Target

This release implements the activation-based subset required by FedSNN and common feedforward SNNs. Its compatibility contract is SpikingJelly `0.0.0.0.14`, the version pinned by the current FedSNN lockfile; the KLIF gap audit additionally checked upstream source snapshot `f67935114ab178300be623297b41adef6727622f`. The qualified accelerator stack is Ascend 910B4, CANN 8.5.0, Python 3.10, PyTorch 2.9.0, and torch-npu 2.9.0.

## Supported APIs

| Module | Supported |
|---|---|
| `base` | `StepModule`, `SingleStepModule`, `MultiStepModule`, `MemoryModule`, memory traversal |
| `neuron` | `BaseNode`, `IFNode`, `LIFNode`, `KLIFNode`, `ParametricLIFNode` |
| `surrogate` | `heaviside`, `Sigmoid`, `ATan`, `PiecewiseQuadratic`, `SoftSign`, `SuperSpike` |
| `functional` | reset/detach/configuration and sequence helpers |
| `layer` | Linear, Conv1d/2d/3d, BatchNorm1d/2d/3d, max/average/adaptive-average pooling, Flatten, VotingLayer, SeqToANNContainer |

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
- KLIF applies `h = relu(k * h_pre)` after ordinary LIF charge, fires from `h`, optionally resets using `h / k` and `v_threshold / k` when `scale_reset=True`, and exposes learnable checkpoint key `k` initialized to `1.0`.
- PLIF initializes `w = -log(init_tau - 1)`, uses `sigmoid(w)` as reciprocal tau, and exposes checkpoint key `w`.

## Intentional extensions

- `backend="npu"` is an explicit alias for pure PyTorch operators on an NPU device.
- `backend="aspy"` is available on multi-step `IFNode`, fixed-tau `LIFNode`, learnable-`k` `KLIFNode`, and `ParametricLIFNode`. It requests the optional fused Ascend C backend; it is not a CuPy-compatible array API.
- On those replacement neurons, `backend="cupy"` is accepted as a migration preference alias and normalized to the same observable AsPy request. Unsupported/native-missing cases follow the normal strict/fallback policy. Actual CuPy imports, arrays, raw kernels, or CUDA execution are not emulated.
- The qualified AsPy request is a non-empty, contiguous, storage-offset-zero FP32 time-major tensor on an Ascend NPU, with matching contiguous initial voltage and a spiking `surrogate.ATan`. LIF and KLIF additionally require a fixed Python float `tau > 1`. KLIF keeps scalar `k` dynamic and expands it into an aligned native device block without changing the public parameter shape; PLIF transports its one-element contiguous FP32 reciprocal-tau tensor as a device input, so `w` remains dynamic across fixed-shape NPUGraph replays.
- AsPy supports hard/soft reset, both `detach_reset` values, LIF/KLIF/PLIF `decay_input`, both KLIF `scale_reset` values, optional `v_seq`, carried-state gradients, input gradients, KLIF `k.grad`, and PLIF `w.grad` in the qualified scope.
- Unsupported or unavailable requests fall back before native execution when `backend_strict=False`. With `backend_strict=True`, the same condition raises `AsPyBackendError`. Once a native function has been invoked, failures propagate and are not silently replayed through a stateful eager step. `last_backend_route` records the implementation and reason.
- Step-aware `Linear` explicitly packs `[T, N]` through `seq_to_ann_forward`; this is numerically equivalent to N-dimensional `nn.Linear` and makes the NPU execution policy explicit.
- BatchNorm1d multi-step accepts both `[T,N,C]` and `[T,N,C,L]`, matching modern PyTorch usefulness rather than preserving the old release's restrictive shape check.
- `MemoryModule.reset()` restores no-grad/eval tensor memories in place where safe, helping static graph address stability.

## AsPy and NPUGraph

The bridge submits ACLNN work through torch-npu `OpCommand::RunOpApiV2` on the current NPU stream. It does not call `torch.npu.synchronize`, `aclrtSynchronizeStream`, or an equivalent synchronization in the forward/backward hot path. Workspace and ACL tensor/executor owners are retained through the task-queue callback.

Real fixed-shape NPUGraph capture/replay is qualified for AsPy IF, LIF, and PLIF. PLIF replay reads reciprocal tau from a device tensor and has been tested while changing `w` and input across five replays, including forward, input-gradient, and `w.grad` parity. KLIF is qualified for native eager multi-step execution only; no KLIF NPUGraph claim is made. Normal graph restrictions still apply: fixed argument structure, shape, dtype, device, layout and `requires_grad` state; separate train/eval captures; and deterministic algorithms for the default qualified training-capture policy.

## Unsupported

- General CuPy-compatible arrays, CUDA kernels, or Triton CUDA kernels.
- Native AsPy single-step execution. Single-step requests use the observable PyTorch fallback when non-strict; strict single-step KLIF raises `AsPyBackendError` because the requested native route is unavailable. IF/LIF/PLIF retain their documented strict-compatible single-step eager path.
- FP16/BF16 AsPy kernels, non-contiguous/storage-offset inputs, non-ATan surrogates, non-spiking surrogate mode, empty time dimensions, or arbitrary dynamic-shape NPUGraph replay.
- Higher-order AsPy gradients. The native IF, LIF, KLIF, and PLIF paths support first-order ATan-surrogate differentiation only.
- Full SpikingJelly layer/neuron catalog.
- Whole-model pickle compatibility across module paths.
- Higher-order gradients through NPUGraph.

## Backend naming

`backend="torch"` means the implementation uses PyTorch operations. It does not mean CPU-only. When the module and tensors are on `npu:*`, those operations dispatch through torch-npu.

Direct unsupported backend assignment raises `NotImplementedError`. `functional.set_backend` warns and keeps the existing backend when a module cannot support the requested value.
