# Performance methodology

## Sequence benchmark protocol

`benchmarks/_protocol.py` implements the measurement policy frozen in `docs/evidence/sequence_acceptance_policy.json`. The representative entrypoints are:

- `benchmarks/benchmark_sequence_recurrent.py`: standard RNN/GRU/LSTM and project-defined spiking RNN/GRU/LSTM;
- `benchmarks/benchmark_sequence_transformer.py`: standard Transformer encoder and decoder stacks;
- `benchmarks/benchmark_spikformer.py`: SpikingSelfAttention and complete Spikformer.

Formal invocations default to exactly five fresh worker processes. Within every worker the candidate and baseline run once each, with alternating first position across workers. Each implementation records a synchronized cold call, required warmup, per-iteration synchronized samples until at least five seconds of measured work, median/mean/p90 latency, optional peak allocated/reserved device memory, provider-route/native-region/format-conversion hooks, runtime identity, and source/build identity. The orchestrator rejects input, initial-state, workload, schema, or order drift, retains every raw worker record, and reports the median of the five process medians. Formal output from a dirty source tree is explicitly marked evidence-ineligible. Candidate and baseline are intentionally separate hooks even when the current semantic-alpha entrypoints instantiate identical eager references; a future optimization must replace only the candidate builder while preserving the deterministic payload and validation contract.

Example CPU development smoke (not performance evidence):

```bash
python -m benchmarks.benchmark_sequence_recurrent \
  --case gru --device cpu \
  --time-steps 4 --batch-size 2 --input-size 8 --hidden-size 8 \
  --fresh-processes 1 --warmup-iterations 1 \
  --minimum-measured-work-seconds 0 --minimum-measured-iterations 2 \
  --maximum-measured-iterations 4 --smoke \
  --output "$HOME/.cache/spikingjelly_npu/sequence-gru-smoke.json"
```

Omit the smoke overrides for the frozen five-process/five-second protocol and use the reference shapes from the acceptance policy. Store every generated JSON outside the checkout. A CPU smoke validates orchestration and schema only; it does not qualify an NPU route or support a performance claim.

## Sequence physical adaptation result

On Ascend 910B4/CANN 8.5.0/Python 3.10.20/torch and torch-npu 2.9.0 in FP32, small-shape eager physical adaptation passed:

- standard RNN/GRU/LSTM FP32 primitive fallback: 18/18 train/eval × full/remainder/singleton cases;
- project-defined spiking RNN/GRU/LSTM: 18/18 cases with the same mode/batch coverage;
- standard cross-attention, Transformer encoder, and Transformer decoder: 18/18 cases with the same mode/batch coverage.

These are device availability/parity checks only. The candidate routes were eager PyTorch operations dispatched by torch-npu, not distinct optimized implementations, so no latency or memory claim follows from them.

Under the runtime's default Conv HF32 policy, the localized tiny Spikformer training case kept public output and loss inside the original tight gate but failed input-gradient and parameter-gradient checks: maximum input-gradient difference was `8.2284e-4`, and 7/29 parameter gradients failed, worst `1.3299e-2`. A first HF32-off attempt was invalid because replacing `PYTHONPATH` removed the CANN `tbe` module before model execution. With CANN paths preserved, explicit Conv HF32-off repaired the localized case and a clean-source rerun through `configure_npu(allow_conv_hf32=False)` passed all 12 SpikingSelfAttention/Spikformer train/eval × full/remainder/singleton cases, including two SGD updates per training case. The worst observed differences were `2.3693e-6` for parameter gradients, `8.9407e-7` for continuous node final state, `3.3528e-8` for input gradient, and `2.9802e-8` for public output. This is tiny-shape eager parity only: no representative-shape, convergence, native, graph, latency, memory, or family-wide acceleration claim follows. Detailed sanitized identities and hashes are in [`sequence_physical_qualification_20260803.json`](evidence/sequence_physical_qualification_20260803.json).

## Compact IF/LIF five-process negative result

The compact ABI was compared with the full-output native ABI at `[T,B,F]=[8,64,4096]`. Each IF and LIF hotspot used five fresh processes, alternating candidate/baseline first position, at least five seconds of synchronized work per implementation, and synchronization before and after every timed call. All workers reported zero maximum absolute difference for loss, final voltage, input gradient, and initial-voltage gradient.

| Hotspot | Full median | Compact median | Speedup | Peak allocated-HBM reduction | Gate |
|---|---:|---:|---:|---:|---|
| IF | 2.8254 ms | 2.7386 ms | 1.0317× | 10.0% | Fail |
| LIF | 2.8312 ms | 2.8052 ms | 1.0093× | 10.0% | Fail |

The frozen hotspot gate required either at least `1.25×` speedup, or at least `20%` allocated-memory reduction with no more than `5%` latency regression. Neither hotspot passed. This is a correctness result and a negative performance result; compact routing is not promoted as an optimized public route.

## Optimization hierarchy

1. Pack stateless ANN operators across `[T*N, ...]`.
2. Use fused AsPy IF/LIF/KLIF/PLIF or the exact FedSNN decay-LIF only inside their qualified Ascend scope; KLIF currently has parity qualification but no performance claim.
3. Capture repeated fixed-shape full batches with NPUGraph only where actual-shape capture is separately qualified.
4. Keep diagnostics eager; use the qualified stateless decay-LIF for FedSNN remainder batches, while generic graph remainders stay eager.
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

On Ascend 910B4 with CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32, `T=4`, batch 128, 20 warmups, 100 measured iterations, and synchronization after every iteration, the compact ConvNet eval/forward proxy measured 9.471 ms stepwise, 8.296 ms packed eager, and 4.740 ms NPUGraph median latency. That is 2.00× NPUGraph versus explicit stepwise and 1.75× versus packed eager.

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

## FedSNN AlexNet-BNTT diagnosis and qualification

The real-client workload is CIFAR-10 Dirichlet α=0.3, seed 2, client 0, `fedsnn_alexnet_bntt`, T=4, batch 128, LE=5, 1706 samples (13 full batches plus remainder 42). Complete-client timing includes data loading, H2D, exact four-draw Poisson encoding, forward, loss, backward, SGD, remainder handling, and terminal synchronization.

### Evidence identity and reproduction boundary

The actual-model evidence was collected externally on the external FedSNN qualification workflow rather than by a standalone harness shipped in this repository. The consumer workspace was based on FedSNN Git HEAD `386c9a418f010dd5a90bac9519b78cdc2e708765`; the library base HEAD was `9b7fa67832df01303d5a2037ef3848a6052e36d3`, with the qualified implementation identified by tree digest `928428318bded45ca1277c9d3449cb146e099a2e4179aba0ba280eee0fd21104`. The frozen FedSNN qualification archive digest was `b16ed36cb100348d881e3ff95583d64141f115842368b2f0013aaeecc6c1c18c`, the resolved smoke config SHA-256 was `12c7dcab52b7ba68ff20986c259fc50e5f75bb55f5a678ab47cc0e0bc6ce2395`, and the native build root identity was `6db7254534e8381224f0a6a3089ffb5fce0640282d8c65fe1611f9d5653fb0f4`.

The consumer adapter/test changes were not a clean public FedSNN commit at qualification time, so these results must be described as externally collected qualification, not as a benchmark reproducible from this repository alone. Small immutable summaries are checked in under [`docs/evidence/`](evidence/):

- [`packed_aspy_manifest.json`](evidence/packed_aspy_manifest.json) records the library tree, consumer archive, config, and native-build identities, plus SHA-256 identifiers for retained parity/route/smoke logs; those external logs are not shipped in this repository;
- [`fedsnn_client_benchmark.json`](evidence/fedsnn_client_benchmark.json) contains the balanced raw process records, order, seeds, routes, synchronization policy, state/config hashes, and timings;
- [`stage_profile_summary.json`](evidence/stage_profile_summary.json) contains the actual-shape stage samples;
- [`actual_model_acceptance_policy.json`](evidence/actual_model_acceptance_policy.json) records the approved gradient and update interpretation.

To reproduce rather than merely audit the summaries, a developer needs the corresponding FedSNN consumer adapter, the exact YAML, CIFAR-10 partition/cache, CANN environment, and external benchmark/parity/smoke harnesses. Until that integration is published as a clean FedSNN commit, use the public `DecayLIF` primitive and the adapter contract in [`fedsnn-integration.md`](fedsnn-integration.md), then rerun the same gates: full/remainder/singleton parity, six train/eval native routes, BNTT buffers, two SGD updates, two smoke runs, one complete-client warmup per fresh process, balanced backend order, and one synchronization immediately before/after each measured client.

The original `packed_eager` path removed T-fold submissions for stateless Conv/Linear/pool work, but each forward still performed 24 timestep-specific BNTT calls and 24 temporal LIF recurrence/autograd chains across six spiking layers. A dedicated stateless AsPy decay-LIF now fuses the exact `decay * membrane + current`, threshold, detached reset, and reverse-time ATan-surrogate recurrence. It does not change Poisson, timestep-specific BNTT, cumulative readout, or state-dict semantics.

### Actual-shape stage profile

On Ascend 910B4, CANN 8.5.0, Python 3.10.20, torch/torch-npu 2.9.0, FP32, T=4 and batch 128, two synchronized actual-model profiles per backend produced these means:

| Stage | `packed_eager` | `packed_aspy` | Change |
|---|---:|---:|---:|
| Packed ANN | 6.201 ms | 6.193 ms | approximately unchanged |
| Timestep BNTT | 7.495 ms | 7.645 ms | approximately unchanged |
| Membrane recurrence | 10.217 ms | 4.724 ms | **−53.8% / 2.16×** |
| Encoded forward total | 26.030 ms | 20.755 ms | **−20.3%** |
| Backward | 28.741 ms | 21.707 ms | **−24.5%** |
| Complete profiled batch wall | 57.112 ms | 44.796 ms | **−21.6% / 1.275×** |

ANN+BNTT time staying flat while recurrence and backward fall confirms that the gain comes from fusing the remaining temporal LIF chain, not from re-labeling or re-packing the ANN path.

### Balanced complete-client benchmark

Each backend used five fresh processes in balanced order. Every process ran one complete client epoch as warmup, then measured one complete LE=5 client run. Exactly one `torch.npu.synchronize` occurred immediately before and after each complete measured run. All processes started from the same state hash and used the same benchmark policy; every `packed_aspy` measurement reported 70 packed forwards × six native LIF calls = 420 native calls with no fallback.

| Backend | Five samples (s) | Median (s) | Versus `packed_aspy` |
|---|---|---:|---:|
| `legacy_stepwise` | 5.2822, 5.3214, 5.5683, 5.3926, 5.1510 | 5.3214 | `packed_aspy` **17.5% lower wall / 1.213×** |
| `packed_eager` | 4.8656, 5.1222, 5.0087, 4.9834, 5.4152 | 5.0087 | `packed_aspy` **12.4% lower wall / 1.141×** |
| `packed_aspy` | 4.9416, 4.0022, 4.3882, 4.3772, 4.4097 | **4.3882** | — |

Actual-model qualification covered full batch 128, remainder 42, singleton, train/eval, diagnostic eager routing, logits, loss, six spike patterns, BNTT buffers, two SGD updates, and all six native routes. Gradients passed the accepted `rtol=5e-5, atol=3e-5` gate. This is numerical-tolerance equivalence, not bitwise equivalence. Two consecutive exact-YAML real trainer smokes also passed with finite metrics and native train/eval routes.

The result qualifies `packed_aspy` for this tested execution path. It does not establish federated-round speedup, multi-seed convergence, or accuracy improvement, and it must not be applied retroactively to an existing experiment identity.

## Qualification boundaries

The IF and PLIF timings above are implemented-kernel, fixed-shape component measurements. They do not include a FedSNN client data pipeline, communication, aggregation, federated rounds, or convergence. LIF is functionally qualified through the full AsPy NPU suite and true NPUGraph tests, but no standalone generic-LIF performance number is claimed here. The FedSNN-specific decay-LIF result is a separate actual-model complete-client execution qualification; it still does not establish federated convergence or communication/control-plane speedup.

The initial scalar, one-core, host-synchronized IF prototype measured approximately 189× slower than PyTorch. That rejection was useful historical evidence, but it does **not** describe the current vectorized multi-block `RunOpApiV2` implementation and must not be used as current release status.

## Remaining end-to-end acceptance gate

The actual-model client execution path is qualified and materially faster, but do not call the project a complete federated-training acceleration until the trainer additionally demonstrates:

- representative federated-round reduction with communication and control-plane time accounted for;
- stable multi-seed convergence under the approved numerical-tolerance boundary;
- unchanged experiment identity and protocol semantics for every reported comparison.

The earlier aspirational 1.5× local-training threshold is not claimed: the measured complete-client speedup is 1.141× versus `packed_eager` and 1.213× versus `legacy_stepwise`. Report those measured values rather than substituting component-kernel speedups.
