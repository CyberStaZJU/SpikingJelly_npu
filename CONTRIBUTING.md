# Contributing

1. Keep the package importable without `torch_npu`.
2. Add CPU tests for every public semantic change.
3. Mark real accelerator tests with `@pytest.mark.npu`.
4. Run NPU work only in a correctly bootstrapped CANN environment on an idle or allocated device.
5. Preserve the documented SpikingJelly compatibility subset unless a change is clearly recorded as an extension.
6. Keep generated state outside the repository: environments, compiler caches, profiler data, benchmark JSON, datasets, checkpoints, and logs.
7. Performance claims require a reproducible command and complete runtime/shape/timing metadata.

Before submitting:

```bash
ruff check .
python -m pytest -q tests -m 'not npu'
python -m build --outdir "$HOME/.cache/spikingjelly_npu/dist"

# Optional native source is built only on the qualified server and all generated
# state must use an empty directory outside the checkout:
# SPIKINGJELLY_NPU_ASPY_BUILD_ROOT="$HOME/.cache/spikingjelly_npu/aspy-..." \
#   scripts/build_aspy.sh
```

## Adding or changing an AsPy operator

A native change is complete only when every layer of the public contract is updated:

1. add or update the `msopgen` definition and stable ACLNN/operator names;
2. validate shape, dtype, scalar attributes, alignment, and physical format in host inference/tiling and the C++ bridge;
3. document tile length, core cap, padding, workspace, current-stream submission, and callback lifetime assumptions in the kernel/bridge code;
4. register every host/kernel source in `scripts/build_aspy.sh` and `native/aspy/op_kernel/CMakeLists.txt`;
5. expose pybind symbols and a positive capability flag so older bundles remain importable and fall back or fail before launch;
6. keep adapter reshape/padding/cropping differentiable and guard higher-order gradients unless explicitly implemented;
7. route unsupported requests before extension loading, preserve observable reasons, and never replay eager code after native launch;
8. add CPU semantics, fake-native adapter, malformed-result, old-bundle, direct bridge-failure, physical-NPU, full-shape, and remainder tests;
9. record numerical tolerances as tolerance equivalence, plus complete workload, warmup, synchronization, device, stack, route, and raw timing evidence;
10. update native bundle manifests and prominently state when an older GitHub Release lacks the new capability.

When changing native code, also inspect the source distribution rather than only the wheel. `MANIFEST.in` must keep `native/aspy/`, the external build scripts, tests, and developer documentation available to downstream developers:

```bash
python -m build --sdist --outdir "$HOME/.cache/spikingjelly_npu/dist"
tar -tzf "$HOME/.cache/spikingjelly_npu/dist"/spikingjelly_npu-*.tar.gz
```
