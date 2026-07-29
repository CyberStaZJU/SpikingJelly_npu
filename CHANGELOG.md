# Changelog

## 0.1.0 - 2026-07-29

- Added a pure-PyTorch activation-based compatibility subset for IF, LIF, PLIF, surrogates, state helpers, common step-aware layers, and graph-safe FedSNN building blocks.
- Added optional source-only AsPy custom operators for Ascend 910B4/CANN 8.5: FP32 multi-step IF, fixed-tau LIF, and dynamic-parameter PLIF forward/backward, including hard/soft reset, both `detach_reset` values, both LIF/PLIF `decay_input` values, stored voltage sequence, carried-state/input gradients, and PLIF `w.grad`.
- Added vectorized multi-tile AIV kernels with multi-block tiling, plus external-only `msopgen`/CMake/NpuExtension build state. The published wheel remains pure Python and contains no compiled AsPy library.
- Changed the native bridge to allocate NPU byte workspaces compatible with torch-npu 2.9 and submit IF/LIF/PLIF ACLNN calls on the current NPU stream through `OpCommand::RunOpApiV2`. Workspace, executor, and ACL tensor owners are retained by the task-queue callback; no explicit native hot-path stream/device synchronization is used.
- Added lazy AsPy routing with import safety, observable pre-execution fallback, `backend_strict=True` errors, transactional state commit, and failure propagation after a native launch begins.
- Added explicit first-order-only ATan-surrogate autograd guards and qualified contiguous, storage-offset-zero FP32 multi-step inputs. Native single-step, FP16/BF16, non-ATan/non-spiking surrogates, unsupported layouts, and arbitrary dynamic-shape graphs remain outside scope.
- Added true fixed-shape NPUGraph capture/replay for AsPy IF, LIF, and PLIF. PLIF passes reciprocal tau as a device tensor, allowing `w` and input to change across fixed-shape replays while preserving output, input-gradient, and `w.grad` parity.
- Expanded the external AsPy qualification driver to cover IF fresh-process hashes, three 20-step trajectories, true NPUGraph, and a three-process synchronized three-path benchmark; and PLIF fresh-process hashes, three optimizer trajectories, five dynamic-parameter graph replays, and a three-process synchronized benchmark.
- Qualified IF on Ascend 910B4/CANN 8.5 with fixed `[B,T,F]=[64,8,4096]`, 10 warmups, 50 iterations, per-iteration measurement synchronization, and three fresh processes: PyTorch 9.241211 ms, AsPy eager 2.685313 ms, and AsPy NPUGraph 2.056798 ms median-of-run-medians; median run speedups were 3.401488× and 4.606828×.
- Qualified implemented PLIF on fixed `[T,B,F]=[8,64,4096]` with the same warmup/iteration/process policy: per-run speedups 5.320089×, 6.337813×, and 4.311383×, median 5.320089×; the final complete AsPy NPU suite passed 76 tests.
- Retained the initial scalar, one-core, host-synchronized IF result (approximately 189× slower than PyTorch) only as historical rejection evidence; it no longer describes the current vectorized multi-block task-queue implementation.
- Added lazy Ascend runtime probing and graph-friendly device configuration, including explicit torch-npu 2.9 ACLNN selection with `jit_compile=False` and base-format selection with `allow_internal_format=False`.
- Added fixed-full-batch `StaticGraphRunner` with eager partial/diagnostic fallback, explicit graph-safe model qualification, default-disabled training capture, a deterministic-algorithm requirement for the qualified training path, visible route metadata, and fatal buffer/gradient/CPU-RNG/NPU-RNG rollback protection.
- Diagnosed compact-proxy nondeterministic training-graph parity failures as NPU kernel differences crossing hard spike thresholds; deterministic algorithms passed bounded fresh-process checks and three 20-step trajectories exactly, while capture-end synchronization alone did not resolve the issue.
- Qualified the deterministic compact training proxy at 1.35× versus stepwise and 1.22× versus packed eager, below the 1.5× local-training acceptance target.
- Added a one-line GitHub Release installer with SHA-256 verification, pure-Python wheel installation, and a stack-pinned relocatable AsPy bundle for Linux aarch64 / CPython 3.10 / torch and torch-npu 2.9 / CANN 8.5.
- Added opt-in process-local `spikingjelly` import aliasing, a `cupy`→AsPy migration preference for qualified neurons, and optional delegation to torch-npu's official `transfer_to_npu` compatibility layer for dedicated Ascend processes.
- Added CPU and marked NPU tests, examples, benchmark harnesses, CANN 8.5 scripts, and migration documentation.
- No component benchmark in this release is a formal FedSNN end-to-end training, federated-round, or convergence claim.
