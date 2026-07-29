# Project instructions

- The Mac checkout is the authoritative source for this repository. Compute-host copies are temporary test snapshots only.
- Keep datasets, virtual environments, build outputs, benchmark JSON, logs, caches, checkpoints, and NPU profiler output outside this repository.
- Preserve SpikingJelly-compatible public semantics for the documented compatibility subset.
- All acceleration paths must retain an eager PyTorch fallback and must not explicitly import `torch_npu` at package import time. Tests that isolate this behavior must disable PyTorch's independent device-backend autoload.
- Test CPU semantics first. Run NPU tests only in a correctly bootstrapped CANN environment and only on an idle or explicitly allocated device.
- Do not claim performance improvements without recording the device, software stack, shapes, warmup, synchronization policy, and benchmark command.
