# UMNA: Unified Modular Neural Architecture for Conservatively Extensible Capability Systems

**Authors:** Koushik Das  
**Version:** 0.1.0 (pinned)  
**Date:** 2026-09-24

---

## Abstract

We present UMNA, a host-controlled capability-graph runtime where a frozen Core executes independently owned capability modules (Wings) selected per input by a planner (NICE-R) under hard compute/latency/memory budgets. The central hypothesis: **when new capabilities are added and routing is allowed to change, does the system control behavioral drift better than parameter-isolation-only baselines?**

UMNA makes three architectural commitments:

1. **Write isolation, not gradient isolation.** The Core is frozen (hash-pinned); Wings train through it with a parameter write firewall. Gradients flow through the Core — this is correct and unavoidable. What is forbidden is a *write* to Core parameters.

2. **Calibration-class admission.** No graph reaches production without a registered calibration class (split conformal on task heads, α=0.1). Graphs outside every class run shadow-only for data collection.

3. **Conservative Capability Extension (CCE).** Promotion from shadow to production requires: (a) shadow evidence ≥10 executions with >80% accept rate, (b) drift gates pass (step drift ≥ −ε, origin drift ≥ −E) for all protected capabilities, (c) tier-A capacity not exceeded. Demotion on drift alarm.

---

## 1. Architecture

### 1.1 Components

| Component | Responsibility | Key Property |
|-----------|----------------|--------------|
| **Core** | Frozen trunk, single epoch, hash-pinned | Trained once, never modified |
| **Wings** | Capability modules (transform + head) | Independent ownership, versioned, provenance-tracked |
| **Registry** | Manifest validation, host metrics, epoch pinning | Rejects vendor-declared routing numbers |
| **Planner (NICE-R)** | Graph selection under hard budgets | Marginal-value stopping, complementarity modeling |
| **Admission** | Calibration-class lookup + CCE drift gate | Shadow-only if no class or demoted wing |
| **Executor** | Wave-parallel, traced, replayable | 8-term latency decomposition |
| **CCE Pipeline** | Shadow → Router → DriftGate → Promotion | Conservative by construction |

### 1.2 Data Flow

```
Input → Retrieve candidates → Plan (NICE-R) → Admit (calibration + CCE) → Execute (wave-parallel) → Trace
                                                              ↑
                                              ShadowRouter collects evidence
                                                              ↓
                                              DriftGate checks step/origin drift
                                                              ↓
                                              Promotion if all gates pass
```

### 1.3 Typed Capability Contract

Every Wing declares: `input_types`, `output_types`, `capabilities`, `dependencies`, `resource_ceilings`.  
**Never declares:** reliability, accuracy, latency, evidence-gain, relevance. Those are host-measured.

---

## 2. NICE-R v0.2: Marginal-Value Planning

### Objective
```
J(S,x) = Û(S,x) - λ_C C(S) - λ_R R(S) + λ_K K(S)
```
- `C(S) = α·F + β·L + γ·M + η·|S|` (FLOPs, latency, memory, size)
- `R(S) = Σ max(0, -κ(i,j))` redundancy penalty
- `K(S) = Σ max(0, κ(i,j))` complementarity bonus
- `κ(i,j) = U({i,j}) - U({i}) - U({j})` pair synergy

### Expansion Rule
```
max_w [ UCB(ΔU) - λ_C ΔC - λ_R ΔR + λ_K ΔK ] > τ
UCB(w|S) = ΔÛ(w|S) + κ_ucb · σ(w|S)
Stop when max_w ≤ τ
```

### Search Structure
```
Stage 1: Top-m singletons (m=6) by utility
Stage 2: Top-b complementary pairs (b=4) by U + λ_K·max(0,κ)
Stage 3: Beam (width=4) over pairs with 3rd-wing additions
Stage 4: Marginal-value stopping with UCB
Stage 5: Pareto pruning (coverage vs size)
Stage 6: Final evaluation
```

### Frozen Hyperparameters (v0.2)
| Param | Value |
|-------|-------|
| λ_C (α,β,γ,η) | (1e-5, 0.01, 0.01, 0.2) |
| λ_R | 1.0 |
| λ_K | 1.0 |
| τ | 0.05 |
| κ_ucb | 0.5 |
| beam_width | 4 |
| max_wings | 6 |

---

## 3. Conservative Capability Extension (CCE)

### Promotion Protocol
1. **Shadow evidence:** ≥10 executions, >80% accept rate
2. **Drift gates pass** for all protected capabilities:
   - Step drift: P(t) - P(t-1) ≥ -ε
   - Origin drift: P(t) - P(0) ≥ -E
3. **Tier-A capacity** not exceeded (|A| < |A|_max)
4. **Demotion** on any drift alarm

### Tiers
| Tier | α | ε | E | Min Samples | Use Case |
|------|---|---|---|-------------|----------|
| A | 0.05 | 0.02 | 0.05 | 100 | Critical capabilities |
| B | 0.10 | 0.03 | 0.10 | 50 | Important capabilities |
| C | 0.20 | 0.05 | 0.15 | 30 | Exploratory |

---

## 4. Calibration & Admission

- **Split conformal** on binary heads, α=0.1, holdout coverage verified
- **Binding = (Core epoch, graph, head, distribution, hardware)** — any change = new calibrator
- **Admission:** `admit(graph, binding)` → PRODUCTION if class exists + CCE eligible, else SHADOW_ONLY
- **No path reaches execute without admit**

---

## 5. Safety Interrupts

5 types × 5 actions, pre-authorized wings only, calibrated bindings, priority order:
| Type | Actions |
|------|---------|
| EMERGENCY | APPEND_GUARD, FORCE_ESCALATION, FORCE_ABSTAIN, REQUEST_SECONDARY, BLOCK_ACTION |
| CONTRAINDICATION | ... |
| HARD_CONFLICT | ... |
| SECURITY | ... |
| LEGAL_POLICY | ... |

**Interrupts NEVER silently replace protected route** — they append, escalate, force abstain, request secondary, or block.

---

## 6. Fault Containment

| Tier | Mechanism | Cost |
|------|-----------|------|
| F0 | In-process NaN/Inf bit-pattern check, dtype/schema validation | ~0.01ms |
| F1 | Process-isolated (wing-executor binary, JSON stdin/stdout) | ~1ms |
| F2 | Hardware-partitioned (CUDA MPS, Vulkan compute, dedicated accelerator) | ~0.5ms |

---

## 7. Core Migration Framework

- **One deployment = one Core major version** (version diamonds hard-unsatisfiable)
- **Migration analysis:** `analyze_migration(to_epoch)` → compatibility report
- **Cost model:** `w1·N_retrain + w2·N_revalidate + w3·N_recalibrate + w4·T_engineering`
- **Automated pipeline:** retrain → revalidate → recalibrate → regenerate benchmarks → execute

---

## 8. Research Hypotheses

| ID | Hypothesis | Test |
|----|------------|------|
| **H1** (Routing Pareto) | NICE-R achieves Oracle-level utility with fewer Wings | Layer C benchmark |
| **H2** (Synergy) | Complementarity modeling finds useful pairs | Synergy probe (Mode B vs random) |
| **H3** (CCE) | Conservative promotion controls drift under routing changes | Sequential promotion + drift |
| **H4** (Contract Utility) | UMNA + NICE-R + CCE beats LoRA/Adapter fleets at equal compute | Fleet benchmark |
| **H5** (Scaling) | Typed retrieval latency ~linear in N | Scaling probe (N=5000) |

---

## 9. Current Status (Layer B v0.1 Complete)

| Layer | Status | Evidence |
|-------|--------|----------|
| S0–S7 | GATED | All gates passed |
| Shadow routing | GATED | 10 tests |
| Drift gating | GATED | 31 tests |
| Safety interrupts | GATED | 16 tests |
| Bridges | GATED | 13 tests |
| Second modality | GATED | 13 tests |
| Rust control plane | GATED | 6 crates |
| Fault containment F0/F1/F2 | GATED | Integration tests |
| Core migration | GATED | Analysis + cost model |
| **Layer B v0.1** | **GATED** | **Interference probe: 10 seeds, 0 variance** |

**Layer B v0.1 Result (Capability-Isolation Sanity Test):**
| Method | Target Δ | Mean \|Protected Δ\| | Worst Regression |
|--------|----------|---------------------|------------------|
| UMNA | +0.324 | 0.0000 | 0.0000 |
| LoRA | +0.326 | 0.0000 | 0.0000 |
| Adapter | +0.324 | 0.0000 | 0.0000 |
| FullFT | +0.314 | 0.0084 | -0.0160 |

> **Interpretation:** UMNA achieves target improvement comparable to LoRA/Adapter while preserving protected capabilities. FullFT shows measurable cross-capability interference. Layer B validates the substrate; it does NOT establish unique UMNA superiority.

---

## 10. Next: Layer C (H1) — Dynamic Routing + Composition

| Milestone | Target |
|-----------|--------|
| Multi-wing calibration classes (all pairs/triples/4–6-wing) | DONE |
| Composition routing (sequential, parallel+fusion) | DONE |
| NICE-R arms search multi-wing graph space | DONE |
| Route displacement measurement (JS divergence) | DONE |
| Held-out planner test (multiple seeds, fresh tasks) | **Next** |
| Freeze H1 protocol | After Layer C |
| Sequential CCE promotion experiments | After H1 |

---

## 11. Reproducibility

- **Version:** 0.1.0 (pinned, never bumps)
- **Commits:** GPG-signed, local only, owner commits
- **Benchmarks:** `python3 -m bench.harness`, `bench.interference_multi_seed`, `bench.hypothesis_test`, `bench.scaling_probe`, `bench.synergy_probe`
- **Tests:** 197 passing, lint clean
- **Artifacts:** Content-addressed by hash, outside git