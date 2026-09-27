# UMNA-BENCHMARK

**UMNA — Unified Modular Neural Architecture**

Status: Working research prototype + active validation.

---

## What UMNA Is

UMNA is a research architecture for adding, coordinating, evaluating, and rolling back capabilities through explicit modular boundaries rather than treating capability growth as an uncontrolled modification of one system.

A frozen Core (single epoch, hash-pinned) provides shared computation. Each capability is a **Wing**: an independently versioned package with typed inputs/outputs, its own parameters, and its own resource ceilings. For each input, a planner called **NICE-R** picks a small typed graph of Wings to execute under hard budgets, and a risk layer decides whether the answer is good enough to serve or whether the system abstains.

The central research hypothesis: **When new capabilities are added and routing is allowed to change, does the system control behavioral drift better than parameter-isolation-only baselines?**

---

## Research Status (Explicit)

- Working prototype exists (Python MVP, Rust control plane in progress)
- Preliminary development experiments exist (Layer B v0.1 GATED)
- Core research hypotheses (H1–H5) are **not yet independently established**
- Held-out and independent validation remains in progress
- **This repository must not be interpreted as evidence of production safety**
- No claim that UMNA universally outperforms LoRA, Adapters, MoE, or other baselines

---

## Implemented Components (Confirmed in Code)

| Component | Location | Status |
|-----------|----------|--------|
| Frozen Core (epoch 1.x) | `core/` | Trained, hash-pinned, write-firewalled |
| 6 Wings (typed, versioned) | `wings/` | alpha, beta, gamma, delta, epsilon, zeta |
| Registry (manifest validation, epoch pinning, typed retrieval) | `runtime/registry.py` | Complete |
| Typed capability contracts (ABI) | `runtime/abi.py` | Complete |
| Graph construction & validation | `runtime/graph.py` | Complete |
| Execution with 8-term latency trace | `runtime/execute.py` | Complete |
| NICE-R v0.2 planner (7 arms + estimator) | `planner/` | Complete |
| Calibration (split conformal, α=0.1) | `risk/conformal.py` | Complete |
| Calibration-class admission | `runtime/admission.py`, `risk/classes.py` | Complete |
| CCE pipeline (shadow → router → drift gate → promotion) | `runtime/cce.py`, `runtime/drift.py` | Complete |
| Fault containment tiers (F0/F1/F2) | `runtime/` | Defined |
| Safety interrupts (5 types × 5 actions) | `runtime/interrupts.py` | Defined |
| Benchmark harness (equal tuning budget) | `bench/harness.py` | Complete |
| Interference/synergy/scaling probes | `bench/` | Complete |

---

## Not Yet Implemented (Per Build Order)

- Bridges (multi-wing fusion modules)
- Second modality (image)
- Rust control plane (S6+) — separate crate repo planned
- Process/hardware fault isolation (F1/F2)
- Core migration framework
- Marketplace
- GPU multi-tenancy

---

## Quick Start

### Requirements
- Python 3.11+
- numpy, scipy, pyyaml (see `requirements.txt`)

### Install
```bash
git clone https://github.com/setunext2028/UMNA-BENCHMARK
cd UMNA-BENCHMARK
pip install -r requirements.txt
```

### Run Tests
```bash
python3 -m pytest
# or
python3 -m unittest discover -s tests
```

### Run Benchmark Harness (S5 Comparison)
```bash
python3 -m bench.harness
```
Expected output: per-arm table with coverage, p50/p95 latency, graph size, admission rate, oracle regret.

### Run Hypothesis Test (Layer B Interference Probe)
```bash
python3 bench/hypothesis_test.py
```
Tests: UMNA vs LoRA vs Adapter vs FullFT on target improvement and protected-wing drift.

### Run Interference Multi-Seed
```bash
python3 bench/interference_multi_seed.py
```
10 seeds, zero variance — validates substrate.

---

## Key Results (Preliminary / Development Evidence)

### Capability-Isolation Sanity Test (Layer B v0.1)
| Method | Target Δ | Mean \|Protected Δ\| | Worst Regression |
|--------|----------|---------------------|------------------|
| UMNA   | +0.324   | 0.0000              | 0.0000           |
| LoRA   | +0.326   | 0.0000              | 0.0000           |
| Adapter| +0.324   | 0.0000              | 0.0000           |
| FullFT | +0.314   | 0.0084              | -0.0160          |

**Interpretation:** UMNA achieves target improvement comparable to LoRA/Adapter while preserving protected capabilities. FullFT shows measurable cross-capability interference. This is a **substrate sanity result**, not proof of UMNA superiority.

### NICE-R v0.2 Development Benchmark
| Arm | Coverage | Avg Wings |
|-----|----------|-----------|
| NICE-R v0.2 | 0.990 | 3.7 |
| Restricted Oracle | 0.990 | 3.0 |
| Greedy | 0.985 | ~6.0 |
| Top-k | 0.985 | ~6.0 |

**Interpretation:** Development evidence. Not held-out proof.

### CCE Functional Test
- 60 shadow traces
- 3 candidate graphs
- 47% acceptance rate
- Predeclared promotion threshold >80%
- **Promotion rejected** (fail-closed behavior confirmed)

### Synergy Probe
Current bounded-pair experiment is **negative** for the tested family. Retained and reported.

---

## What Is Not Yet Established

- H1 (Routing Pareto): NICE-R achieves Oracle-level utility with fewer Wings
- H2 (Synergy): Complementarity modeling finds useful pairs
- H3 (CCE): Conservative promotion controls drift under routing changes
- H4 (Contract Utility): UMNA + NICE-R + CCE beats LoRA/Adapter fleets at equal compute
- H5 (Scaling): Typed retrieval latency ~linear in N
- No production-safety claim
- No claim of complete neural interpretability

---

## Reproducibility

- Version: 0.1.0 (pinned, never bumps)
- Random seeds documented in experiment scripts
- Benchmarks: `python3 -m bench.harness`, `bench.interference_multi_seed`, `bench.hypothesis_test`, `bench.scaling_probe`, `bench.synergy_probe`
- Tests: 199 passing, lint clean
- Artifacts: Content-addressed by hash (e.g., `core/artifacts/core-1.x-8d.npz`)

---

## Repository Layout

```
UMNA-BENCHMARK/
├── core/              # Frozen Core: definition, training, artifact + hash
├── wings/             # Capability modules (manifest.yaml, model.py, params.npz)
├── runtime/           # Python MVP control plane
│   ├── abi.py         # Typed capability contracts
│   ├── registry.py    # Manifest validation, resolution, metrics
│   ├── graph.py       # Typed graph construction
│   ├── execute.py     # Wave-parallel execution, tracing, replay
│   ├── admission.py   # Calibration-class admission
│   ├── cce.py         # Conservative Capability Extension
│   ├── drift.py       # Step/origin drift gating
│   └── ...
├── planner/           # NICE-R v0.2 + arms
│   ├── estimator.py   # Utility estimator, search algorithms
│   └── arms/          # greedy, topk, gates, pairs, beam, oracle, nice_r
├── risk/              # Calibration & conformal prediction
├── bench/             # Harness, probes, tasks
├── tests/             # 199 unit/integration tests
├── tools/             # audit.py, lint_human.py
├── spec/              # Formal specification (mathematics, algorithms, lifecycle)
├── docs/              # Developer documentation (00_READ_ORDER.md first)
└── memory/            # Project decisions, invariants, traps
```

---

## Documentation Entry Point

Start with `docs/00_READ_ORDER.md` — it lists the 12 short docs in reading order.

Key docs:
- `docs/01_WHAT_AND_WHY.md` — Architecture motivation and vocabulary
- `docs/02_MVP_SCOPE.md` — What ships in first milestone
- `docs/03_BUILD_ORDER.md` — Sequential stages with gates
- `docs/05_SEAMS_AND_API.md` — Every interface (contracts)
- `docs/UMNA_MAIN_PAPER.md` — Full technical paper (v0.1.0)

---

## License

Proprietary research prototype. Not licensed for production use.