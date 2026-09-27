# UMNA Test Plan — Required Tests for NICE-R and CCE Validation

## Current State (2026-09-23)

### Working (Layer A/B)
- **Layer A**: Standard capability benchmarks (external) — not yet run
- **Layer B v0.1**: Capability-Isolation Sanity Test — PASS
  - Interference probe with 4 methods (UMNA, LoRA, Adapter, FullFT)
  - 10 reproducibility runs, deterministic
  - Multi-seed CI computation
  - All 199 tests pass, lint/audit clean

### Missing for Layer C (H1 — Routing Pareto)

1. **Multi-wing calibration classes** — Only 6 single-wing + 1 pair (gamma+delta) registered
   - Need: All synergistic pairs, triples, 4-wing, 5-wing, 6-wing combinations
   - Need: Per-task calibration for each composition

2. **Composition routing (multi-wing graphs)** — Greedy arm builds only parallel single-wing graphs
   - Need: Sequential wiring (A→B→C), parallel fusion, branching
   - Need: Planner to select multi-wing graphs

3. **NICE-R arms with multi-wing graphs** — Arms only search parallel subsets
   - Need: Bounded-pair to find complementary pairs via wiring
   - Need: Beam to search multi-wing graph space
   - Need: Oracle to search restricted multi-wing graph space

4. **Route displacement measurement** — No tracking of route changes
   - Need: Compare selected graph before/after capability addition
   - Need: Measure protected-wing behavioral drift under new routes

5. **Hypothesis test measurement artifact** — Protected wings evaluated with original Core
   - Need: Evaluate protected wings with same Core that new routes use
   - Need: Measure behavioral drift from route changes, not parameter changes

### Missing for H3 (CCE — Conservative Promotion)

1. **Shadow routing evidence collection** — Works but doesn't feed planner
   - Need: Shadow traces → router learning → planner utility updates

2. **Drift gating integration with promotion** — DriftGate exists but not wired to promotion
   - Need: Promotion only if drift gate passes
   - Need: Demotion on drift alarm

3. **Sequential promotion experiments** — Not implemented
   - Need: V0 → V1 → V2 → ... capability additions
   - Need: Cumulative drift tracking (step + origin)

## Required Fixes (Priority Order)

### P1: Multi-Wing Calibration Classes
**File**: `bench/harness.py` → `_register_calibration_classes()`
- Register all pairs, triples, 4-wing, 5-wing, 6-wing for relevant tasks
- Each needs: graph, binding, combined score function, conformal fit

### P2: Composition Graph Building
**File**: `runtime/graph.py` → `build()` and `Wiring`
- Support sequential wiring (source_output → target_input)
- Support parallel + fusion (multiple inputs → bridge/fusion)
- Ensure `build()` validates wiring types

### P3: Planner Arms Search Multi-Wing Graphs
**File**: `planner/estimator.py` → search functions
- `_greedy_search`: Add wings with wiring (not just parallel)
- `_bounded_pair_search`: Search pairs with wiring, not just parallel
- `_beam_search`: Beam over graph space with wiring
- `_oracle_search`: Exhaustive over restricted graph space with wiring

### P4: Hypothesis Test Fix
**File**: `bench/hypothesis_test.py`
- Use modified Core for ALL wing evaluations when FullFT
- Track route displacement (selected graph before/after)
- Measure protected wing accuracy under production routing

### P5: CCE Promotion Pipeline
**File**: `runtime/shadow.py` + `runtime/drift.py` + `runtime/admission.py`
- Shadow traces → router learning → planner utility
- DriftGate check before admission to production
- Demotion on drift alarm

## Test Matrix for NICE-R + CCE

| Test | Status | Blockers |
|------|--------|----------|
| Layer C v0.1 (H1 development) | MISSING | P1, P2, P3 |
| H1 final (frozen protocol) | MISSING | Layer C v0.1 working |
| H2 (Synergy) | OK | — |
| H3 v0.1 (CCE development) | MISSING | P4, P5 |
| H3 final | MISSING | H3 v0.1 working |
| H4 (Contract Utility) | MISSING | Requires LoRA fleet baseline |
| H5 (Scaling) | OK | — |
| Sequential Drift (K) | MISSING | P5 + multi-promotion |

## Next Actions

1. Implement P1-P3 to enable Layer C development
2. Run Layer C development benchmarks
3. Freeze H1 protocol
4. Implement P4-P5 for CCE
5. Run H3 development benchmarks