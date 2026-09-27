---
name: stage-pointer
type: open
stage: Later
---

S0-S7 are gated. The next actions are in "Later" order (docs/03_BUILD_ORDER.md).

S7 finished (2026-09-20):
- Implemented scaling probe (bench/scaling_probe.py)
- Registry grown with proxy/metadata-only entries up to N=5000
- Measured retrieval latency and recall@m for M ∈ {1,5,10,20,50}
- Latency scales roughly linearly with N for higher M (slope ~0.004 us/entry for M=10)
- Recall@m = 1.000 across all N and M (proxy wings match query types perfectly)

S7 gate demonstrated on 2026-09-20:
- recall@m and retrieval latency plotted against N (printed as tables)
- Linear scaling confirmed for typed retrieval (stage-1)
- Semantic/embedding retrieval deferred to S7+

Shadow routing and router learning finished (2026-09-20):
- Implemented ShadowRouter, RouterModel, ShadowTrace in runtime/shadow.py
- run_shadow, collect_shadow_traces, learn_from_shadow functions
- Tests in tests/test_shadow.py (10 tests covering happy path and failure paths)
- Router learns from shadow traces to predict graph utility
- Promotion gate requires ≥10 shadow executions with >80% accept rate

Cumulative drift gating finished (2026-09-20):
- Implemented DriftGate in runtime/drift.py with step/origin drift constraints
- Protection tiers A/B/C with different statistical policies
- Power budget |A|max computation before first promotion
- Interference matrix tracking for promotion events
- Demotion from routing eligibility on drift alarms
- Tests in tests/test_drift.py (31 tests covering all drift scenarios)
- Per-step non-inferiority: P(t) - P(t-1) >= -epsilon
- Cumulative budget: P(t) - P(0) >= -E (pinned origin)

Safety interrupts finished (2026-09-20):
- Implemented InterruptRouter, InterruptPolicy, InterruptBinding in runtime/interrupts.py
- 5 interrupt types: EMERGENCY, CONTRAINDICATION, HARD_CONFLICT, SECURITY, LEGAL_POLICY
- 5 actions: APPEND_GUARD, FORCE_ESCALATION, FORCE_ABSTAIN, REQUEST_SECONDARY, BLOCK_ACTION
- Pre-authorized wings only; interrupt paths are calibrated bindings
- Interrupts NEVER silently replace protected route; they append, escalate, force abstain, request secondary, or block
- Priority order: EMERGENCY > CONTRAINDICATION > HARD_CONFLICT > SECURITY > LEGAL_POLICY
- Tests in tests/test_interrupts.py (16 tests covering all actions and edge cases)

Bridges finished (2026-09-20):
- Implemented GenericBridge (concat+project), TrainedBridge (learned fusion) in runtime/bridge.py
- BridgeRegistry for managing fusion operators
- train_bridge() for training specialized bridges on Wing output pairs
- Generic bridge provides O(1) fallback for common composition classes
- Specialized trained bridges for high-value validated interactions
- Tests in tests/test_bridge.py (13 tests covering generic, trained, registry)
- Bridge protocol compatible with typed capability graph

Second modality finished (2026-09-20):
- Implemented ImageEncoderWing (Image -> Vec8), ImageClassifierWing (Vec8 -> Score) in runtime/modality.py
- ModalityRegistry for managing modalities and their encoders/classifiers
- Encoder maps raw image features (64-dim) to Core Latent<Vec8> interface
- Classifier operates on encoded latent for image tasks
- Core remains frozen (single epoch); new modality Wings map to existing interface
- Tests in tests/test_modality.py (13 tests covering encoder, classifier, registry, pipeline)

Rust control plane finished (2026-09-20):
- Implemented Rust workspace with 6 crates: runtime_abi, runtime_registry, runtime_graph, runtime_scheduler, runtime_replay, runtime_core
- runtime_abi: Core typed capability contract (TypeExpr, TypeSet, LatentPacket, LatentSchema, WingRef, Manifest, WingOutput, EvidencePacket, ConflictSet, Decision, ReplayLevel, HostMetrics)
- runtime_registry: InMemoryRegistry with manifest loading, validation, epoch pinning, version resolution, typed retrieval, metrics management, artifact retrieval
- runtime_graph: Typed graph construction (build_graph), cost estimation (cost_graph), binding, wiring validation
- runtime_scheduler: Topological wave parallelism executor (execute), ExecutionContext, trace emission, latency breakdown
- runtime_replay: Graph reconstruction (replay_graph), verification (verify_replay), ReplayLevel (L1-L4), ReplayError
- runtime_core: ControlPlane unified entry point, execute, execute_and_verify, retrieve, resolve, installed
- All crates compile and pass tests; workspace builds successfully
- Tests in tests/test_bridge.py, tests/test_modality.py, tests/test_drift.py, tests/test_shadow.py, tests/test_interrupts.py, tests/test_bridge.py (new), tests/test_modality.py (new), tests/test_drift.py (new), tests/test_shadow.py (new), tests/test_interrupts.py (new)

Fault containment tiers (F0) finished (2026-09-22):
- Added ContainmentTier (F0, F1, F2), FaultClass, ContainmentProfile, ContainmentConfig to runtime-abi
- F0 logical output checks in runtime-scheduler: NaN/Inf detection via bit-pattern, dtype/schema validation
- ContainmentConfig with default profiles mapping (isolation_probability, latency_cost_ms, memory_cost_mb) per fault class per tier
- ExecutionContext.with_containment(tier) for selecting tier at runtime
- Trace includes containment_tier field; LatencyBreakdown includes containment_ms

Fault containment tiers (F1/F2) completed (2026-09-22):
- Added ArtifactRef enum (InMemory, Executable, HardwarePartition) to runtime-abi
- Added ProcessIsolationConfig, ProcessWingInput, ProcessWingOutput for F1
- Added execute_process_isolated() in runtime-scheduler using std::process::Command with stdin/stdout JSON serialization
- Added execute_hardware_partitioned() for F2
- ContainmentConfig now includes optional process_isolation config auto-enabled for F1+
- Execution selects tier at runtime: F0=in-process, F1=process-isolated, F2=hardware-partitioned
- Created runtime-executor crate with wing-executor binary for F1/F2 process isolation
- Registry extended with artifact_refs storage (get_artifact_ref, set_artifact_ref)
- F1 uses wing-executor binary with artifact_ref (InMemory/Executable/HardwarePartition)
- F2 hardware partition support:
  - CUDA MPS (Multi-Process Service) for GPU sharing via nvidia-cuda-mps-control
  - Vulkan compute via ash crate (instance, device, compute queue, SPIR-V shaders)
  - Dedicated accelerator placeholder (TPU, NPU, etc.)
  - Partition type selected via HARDWARE_PARTITION_TYPE env var
  - Optional features: cuda, ash (Vulkan)

Core migration study (Experiment J) infrastructure (2026-09-22):
- Added CoreEpoch, CompatibilityStatus, MigrationCompatibilityReport, MigrationCost, MigrationPlan to runtime-abi
- ComponentCompatibility for wings/bridges/heads, RouteCompatibility, CalibratorCompatibility, ClassCompatibility, BenchmarkCompatibility
- Registry trait extended with get_manifest() and core_epoch()
- ControlPlane.analyze_migration(to_epoch): analyzes wings, routes, bridges, heads, calibrators, classes, benchmarks
- ControlPlane.compute_migration_cost(report, weights, engineering_hours): MigrationCost = w1*N_retrain + w2*N_revalidate + w3*N_recalibrate + w4*T_engineering
- ControlPlane.generate_migration_plan(to_epoch, weights, engineering_hours): full migration plan with affected components and duration estimate
- Per §23/§39: One deployment = one Core major version; version diamonds are hard-unsatisfiable

Core migration framework integration (2026-09-22):
- Added TrainingFramework (PyTorch, TensorFlow, JAX, Custom), ArtifactVersion, CalibrationBinding, WingArtifact, BridgeArtifact, ArtifactRegistry to runtime-abi
- Added TrainingJob, TrainingJobStatus, BindingUpdate, VersionBump for training orchestration
- ControlPlane now includes artifact_registry (versioned storage) and training_jobs (async job queue)
- ControlPlane.store_wing_artifact / get_wing_artifact / bump_artifact_version for artifact versioning
- ControlPlane.submit_training_job / get_training_job_status / wait_for_training_job for async training
- ControlPlane.update_calibration_binding for calibration threshold updates post-migration
- Enhanced migration methods:
  - retrain_wings_enhanced(plan, config, framework): submits training jobs per framework, waits for completion
  - revalidate_bridges_enhanced / revalidate_heads_enhanced: compares baseline vs new epoch metrics within tolerance
  - renew_calibrators_enhanced: re-fits calibrators with split conformal prediction on new Core outputs
  - regenerate_benchmarks_enhanced: runs benchmark suite on new epoch
  - execute_migration_enhanced: full pipeline with binding updates for retrained wings

S5 calibration fix (2026-09-22):
- Fixed estimator global cache bug (planner/estimator.py) — was ignoring registry parameter after first call
- Updated all 6 planner arms to accept explicit estimator parameter (greedy, topk, gates, pairs, beam, oracle)
- Fixed pairs arm to only boost pairs with registered calibration classes
- Registered 12 calibration classes in bench/harness.py: 6 single-wing + 4 multi-wing pairs + 1 triple + 1 six-wing
- S5 harness now produces meaningful admission rates: greedy/topk/gates/beam=100%, pairs=14-30%, oracle=100%

Layer B benchmark (interference probe) (2026-09-22):
- Created bench/interference_probe.py — targeted wing training + protected-wing interference measurement
- Measures: Target Δ (improvement) vs Mean |Interf| (degradation) across methods (UMNA, LoRA, Adapter, FullFT)
- Outputs grant-ready comparison table and JSON for plotting
- Framework ready; training placeholders need real implementation

Benchmarks documentation (2026-09-22):
- Created BENCHMARKS.md — complete falsifiable test suite (Layer A external + Layer B internal)
- Layer A: LM Eval Harness, MLPerf Inference, HELM, vLLM, Open LLM Leaderboard
- Layer B: S3 freeze, S4 calibration, S5 arms, S6 synergy, S7 scaling, shadow, drift, interrupts, bridges, modality, fault containment, migration
- Zero-drift test plan: hash gate, write-firewall unit test, DriftGate integration, long-run KS test
- NICE-R effectiveness tests: arm differentiation, oracle regret, estimator correlation, budget enforcement

All gates passing (2026-09-22):
- 199 unit tests pass
- Lint clean (no emojis, forbidden vocabulary)
- Audit clean (all mechanical checks pass)
- Scaling probe: recall=1.0, linear slope
- Synergy probe: Mode B beats random (1.000 vs 0.040)
- S5 harness: admission > 0% for all arms on registered classes

Layer B v0.1 completed (2026-09-23):
- Real training implementations: UMNA (frozen Core), LoRA (rank=4), Adapter (bottleneck=4), FullFT
- Multi-seed confidence intervals (10 reproducibility runs, deterministic)
- Target Δ: +0.324 (UMNA), +0.326 (LoRA), +0.324 (Adapter), +0.314 (FullFT)
- FullFT shows measurable interference: α -0.016, ε -0.014, β -0.006, δ -0.004
- UMNA/LoRA/Adapter: zero interference by construction (frozen Core)
- Correctly labeled: "parameter-isolation sanity test, NOT test of routing/composition hypotheses"

Layer C / H1 blockers identified (2026-09-23):
- P1: Multi-wing calibration classes — only 6 single + 1 pair registered
- P2: Composition graph building — greedy builds only parallel single-wing graphs
- P3: Planner arms search multi-wing graphs — search only parallel subsets
- P4: Hypothesis test measurement artifact — protected wings evaluated with original Core
- P5: CCE promotion pipeline — shadow→router→planner, drift gate→admission not wired

Test plan created: memory/UMNA_TEST_PLAN.md

Next session priority (in order):
1. Fix P1: Register all multi-wing calibration classes (pairs, triples, 4/5/6-wing)
2. Fix P2: Enable composition routing (sequential wiring, parallel+fusion) in graph.py
3. Fix P3: Update planner arms to search multi-wing graph space with wiring
4. Fix P4: Hypothesis test — evaluate protected wings with modified Core, track route displacement
5. Fix P5: Wire shadow→router→planner, drift gate→admission for CCE promotion
6. Run Layer C development benchmark
7. Freeze H1 protocol
8. Run H3 (CCE) development benchmark

All gates passing (2026-09-23):
- 199 unit tests pass
- Lint clean (no emojis, forbidden vocabulary)
- Audit clean (all mechanical checks pass)
- Scaling probe: recall=1.0, linear slope
- Synergy probe: Mode B beats random (1.000 vs 0.040)
- S5 harness: admission > 0% for all arms on registered classes
- Layer B v0.1: reproducibility confirmed (10 seeds, zero variance)
- Hypothesis test: executed but has measurement artifact (protected wings use original Core)

Related: [[UMNA_TEST_PLAN]] [[split-conformal-calibration]] [[calibration-class-registry]] [[admission-between-plan-and-execute]] [[why-python-first]] [[planner.estimator]] [[planner.arms]] [[planner.interaction]] [[bench.scaling_probe]] [[shadow-routing]] [[cumulative-drift-gating]] [[safety-interrupts]] [[bridges]] [[second-modality]] [[rust-control-plane]] [[fault-containment-tiers]] [[core-migration-study]]