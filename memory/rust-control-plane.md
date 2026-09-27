# Rust Control Plane

Implemented as a Cargo workspace with 6 crates in `/home/navian/UMNA/`:

## Crate Overview

### runtime_abi
Core typed capability contract - the shared types used across all crates.
- **TypeExpr/TypeSet**: Typed capability expressions (e.g., `Latent<Vec8>`)
- **LatentPacket/LatentSchema**: Typed data packets exchanged between Wings
- **WingRef**: Pinned reference to installed Wing version (name@version)
- **Manifest**: Wing declaration (types, capabilities, dependencies, resource ceilings)
- **WingOutput**: Wing execution result (latent, evidence, local_score)
- **EvidencePacket/ConflictSet**: Scored claims with polarity
- **Decision/ReplayLevel**: Execution decisions and replay fidelity levels
- **HostMetrics**: Host-measured routing attributes (latency, memory, reliability, FLOPs)

### runtime_registry
InMemoryRegistry - the single source of truth for installed Wings.
- **Manifest loading/validation**: Strict allow-list parsing, host-owned field rejection
- **Epoch pinning**: Core epoch locking, incompatible wing rejection
- **Version resolution**: Semver range resolution, diamond detection
- **Typed retrieval**: Stage-1 retrieval by input type (O(N) scan)
- **Metrics management**: Host-measured metrics attachment with ceiling validation
- **Artifact retrieval**: Wing artifact lookup for execution

### runtime_graph
Typed graph construction and cost estimation.
- **build_graph**: Type-checked graph construction from manifests + wiring
- **cost_graph**: Level-synchronous cost estimation (compute, latency, memory)
- **Binding**: Calibration attachment point (core_epoch, graph, head, distribution, hardware)
- **Wiring validation**: Type unification, cycle detection, dependency satisfaction

### runtime_scheduler
Topological wave-parallel executor.
- **execute**: Wave-parallel execution with latency breakdown (8 terms)
- **ExecutionContext**: Registry, binding, admission, pipeline timing
- **Trace emission**: Full execution record with latency breakdown, evidence, conflicts
- **Error handling**: Panic-safe execution, failure isolation per wing

### runtime_replay
Trace replay and route verification.
- **replay_graph**: Graph reconstruction from trace wing_versions
- **verify_replay**: Multi-level verification (L1 bitwise → L4 decision-equivalent)
- **ReplayLevel**: L1 bitwise, L2 numerically_close, L3 route_equivalent, L4 decision_equivalent
- **ReplayError**: Reconstruction, GraphMismatch, InputHashMismatch, LatencyTolerance, CostVerification

### runtime_core
Unified control plane entry point.
- **ControlPlane**: Unified facade over all subsystems
- **execute**: Single graph execution with trace
- **execute_and_verify**: Execution + replay verification
- **retrieve/resolve/installed**: Registry operations

## Key Design Decisions

1. **Arc<dyn Registry>**: Dynamic dispatch for registry enables testing and future swap
2. **Arc<dyn Wing>**: Wing artifacts stored as trait objects for dynamic dispatch
3. **Typed keys**: All HashMaps use String keys (owned) to avoid borrow issues
4. **Error conversion**: ExecutionError → ReplayError via From impl for error propagation
4. **Trace cloning**: Evidence cloned for both Output and Trace to avoid move issues
5. **Panic safety**: std::panic::catch_unwind for wing execution isolation

## Testing
- All 199 Python tests pass
- All Rust unit tests pass (cargo test)
- All benchmarks run successfully (harness, synergy_probe, scaling_probe)
- Audit and lint pass

## Integration
- Python benchmarks use Rust crates via FFI (not yet - Python still uses pure Python implementations)
- Future: Python bindings via PyO3 for benchmark integration