# Performance methodology

## Optimization hierarchy

1. Pack stateless ANN operators across `[T*N, ...]`.
2. Use fused AsPy IF/LIF/PLIF only inside its qualified Ascend scope.
3. Capture repeated fixed-shape full batches with NPUGraph.
4. Keep remainders and diagnostics eager.
5. Keep explicit host synchronization outside the model hot path and use it only at measurement or correctness boundaries.
6. Qualify FP16 AMP separately; AsPy is FP32-only in this release.

## Packed-model benchmark

```bash
python benchmarks/benchmark_packed_convnet.py \
  --device npu:2 \
  --batch-size 128 \
  --time-steps 4 \
  --warmup 20 \
  --iterations 100 \
  --mode forward

# Deterministic complete training-step comparison and parity diagnostics:
python benchmarks/benchmark_packed_convnet.py \
  --device npu:2 \
  --batch-size 128 \
  --time-steps 4 \
  --warmup 20 \
  --iterations 100 \
  --mode train \
  --require-training-parity
```

The benchmark uses an encoded current sequence so packed and stepwise paths consume identical inputs. Forward mode uses `torch.no_grad`; train mode enables `torch.use_deterministic_algorithms(True, warn_only=False)` and measures `zero_grad(set_to_none=True)`, forward, squared-mean loss, backward, and SGD step. Treat cold capture separately from steady state.

## AsPy qualification commands

Build and activate the optional extension externally, then use only an idle NPU:

```bash
source scripts/cann_env.sh
export TASK_QUEUE_ENABLE=1
export SPIKINGJELLY_NPU_ASPY_BUILD_ROOT="$HOME/.cache/spikingjelly_npu/aspy-$(date +%Y%m%d-%H%M%S)"
scripts/build_aspy.sh
source "$SPIKINGJELLY_NPU_ASPY_BUILD_ROOT/activate_aspy.sh"

export SPIKINGJELLY_NPU_ASPY_QUALIFICATION_ROOT="$HOME/.cache/spikingjelly_npu/aspy-qualification-$(date +%Y%m%d-%H%M%S)"
ASCEND_DEVICE_ID=7 scripts/run_aspy_qualification.sh
```

The driver refuses an occupied selected NPU and a non-empty evidence directory. It creates fresh Python processes for the repeated IF and PLIF measurements. Its explicit `torch.npu.synchronize` calls are outside the native model hot path: once after warmup and after each measured iteration so wall-clock samples include completed device work. The AsPy bridge itself submits through `OpCommand::RunOpApiV2` and has no explicit hot-path synchronize.

## Required report metadata

- hardware and selected device;
- CANN, torch, torch-npu, Python, and package versions;
- tensor shapes and whether they are `[B,T,F]` or `[T,B,F]`;
- dtype, train/eval state, and complete measured workload;
- warmup and measured iteration counts;
- synchronization policy;
- number of fresh processes and aggregation rule;
- native route and NPUGraph capture status/fallback reason;
- numerical parity and gradient tolerances;
- confirmation that results are component-level rather than formal FedSNN end-to-end evidence.

## Packed ConvNet evidence

On Ascend 910B4 `npu:6` with CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32, `T=4`, batch 128, 20 warmups, 100 measured iterations, and synchronization after every iteration, the compact ConvNet eval/forward proxy measured 9.471 ms stepwise, 8.296 ms packed eager, and 4.740 ms NPUGraph median latency. That is 2.00× NPUGraph versus explicit stepwise and 1.75× versus packed eager.

The deterministic complete-training-step proxy used the same stack, shape, warmup, iteration count, and synchronization policy with SGD learning rate 0.01. It measured 54.669 ms stepwise, 49.184 ms packed eager, and 40.435 ms NPUGraph: 1.35× versus stepwise and 1.22× versus packed eager, with exact one-step logits, loss, parameter gradients, BNTT/buffers, and post-SGD parameters. Three deterministic fresh-process 20-step trajectories also matched packed eager exactly. This remains a compact proxy below the 1.5× local-training acceptance gate, not a formal FedSNN result.

Earlier nondeterministic complete-training runs produced higher timings-based speedups but failed some numerical seeds. Capture-end synchronization did not fix the discrepancy; deterministic algorithms did. Those nondeterministic numbers are historical diagnostics, not the qualified training state.

## AsPy IF three-path evidence

The current implemented IF kernel was measured on Ascend 910B4 `npu:7`, CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32, with fixed input shape `[B,T,F]=[64,8,4096]`. Each sample covered a complete training step: membrane reset, trainable gain, IF forward, linear readout, MSE loss, and backward. Each of three fresh processes used 10 warmups and 50 measured iterations, with `torch.npu.synchronize` after warmup and every measured iteration.

Median-of-run-medians results:

| Path | Median latency | Speedup versus PyTorch |
|---|---:|---:|
| PyTorch IF | 9.241211 ms | 1.000000× |
| AsPy native eager | 2.685313 ms | 3.401488× median run speedup |
| AsPy true NPUGraph | 2.056798 ms | 4.606828× median run speedup |

All three graph runs reported `backend="npugraph"`, `captured=true`, and `expected_batch_size=64`; the inner node reported the AsPy native route. Output and loss matched exactly, and the maximum trainable-gain gradient absolute error was `9.313225746e-10` for native eager and graph paths.

## AsPy PLIF evidence

The implemented PLIF kernel was measured on the same qualified stack and device with fixed shape `[T,B,F]=[8,64,4096]`, FP32. Each sample covered PLIF forward, a loss using spike output and final voltage, backward for the input, and backward to parameter `w`. Each of three fresh processes used 10 warmups and 50 measured iterations, with synchronization after warmup and every measured iteration.

| Fresh process | PyTorch PLIF | AsPy PLIF | Speedup |
|---:|---:|---:|---:|
| 1 | 15.344447 ms | 2.884246 ms | 5.320089× |
| 2 | 13.400400 ms | 2.114357 ms | 6.337813× |
| 3 | 16.060696 ms | 3.725185 ms | 4.311383× |

The median speedup is `5.320089×`. Input-gradient and `w.grad` parity passed at `rtol=3e-5`, `atol=3e-6`. Three fresh determinism processes produced identical hashes including `w_gradient`; three independent 20-step optimizer trajectories passed. A true fixed-shape NPUGraph test changed both `w` and input across five replays; every replay matched native eager for output, input gradient, and `w.grad`, and each changed value produced changed hashes.

## Qualification boundaries

These IF and PLIF timings are implemented-kernel, fixed-shape component measurements. They do not include a FedSNN client data pipeline, communication, aggregation, full model, local epochs, federated rounds, or convergence. No formal FedSNN end-to-end run is claimed. LIF is functionally qualified through the full AsPy NPU suite and true NPUGraph tests, but no standalone LIF performance claim is made here.

The initial scalar, one-core, host-synchronized IF prototype measured approximately 189× slower than PyTorch. That rejection was useful historical evidence, but it does **not** describe the current vectorized multi-block `RunOpApiV2` implementation and must not be used as current release status.

## End-to-end acceptance gate

Do not call the project a formal FedSNN acceleration until the actual trainer demonstrates:

- at least 1.5× steady-state local-training throughput versus eager FP32;
- material five-local-epoch client wall-time reduction after capture amortization;
- representative federated-round reduction with control-plane time accounted for;
- equivalent gradients, state/buffer updates, and acceptable multi-seed convergence.
