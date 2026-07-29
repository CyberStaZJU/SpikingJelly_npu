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
