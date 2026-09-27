# NICE-R v0.2 — Frozen Record

**Status:** SEALED
**Date:** 2026-09-24
**Commit:** (to be filled by owner)

---

## Algorithm Version

NICE-R v0.2 — Marginal-value stopping with complementarity, redundancy, Pareto pruning, UCB exploration.

---

## Objective Equation

```
J(S, x) = Û(S, x) - λ_C C(S) - λ_R R(S) + λ_K K(S)
```

where:
- `Û(S, x)` = predicted utility from linear estimator
- `C(S) = α·F(S) + β·L(S) + γ·M(S) + η·|S|` (FLOPs, latency, memory, size)
- `R(S) = Σ_{i<j∈S} max(0, -κ(i,j))` redundancy penalty
- `K(S) = Σ_{i<j∈S} max(0, κ(i,j))` complementarity bonus
- `κ(i,j) = U({i,j}) - U({i}) - U({j})` pair synergy

**Expansion continues while:**
```
max_w [ UCB(ΔU(w|S)) - λ_C ΔC(w|S) - λ_R ΔR(w|S) + λ_K ΔK(w|S) ] > τ
```

**UCB:**
```
UCB(w|S) = ΔÛ(w|S) + κ_ucb · σ(w|S)
```

---

## Frozen Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| λ_C (α, β, γ, η) | (1e-5, 0.01, 0.01, 0.2) | Cost weights: F, L, M, |S| |
| λ_R | 1.0 | Redundancy penalty |
| λ_K | 1.0 | Complementarity bonus |
| τ | 0.05 | Stopping threshold |
| κ_ucb | 0.5 | UCB exploration |
| beam_width | 4 | Beam size |
| max_wings | 6 | Hard cap |
| pair_seeds | 4 | Top complementary pairs |

---

## Search Structure

```
Stage 1: Top-m singletons (m=6)
Stage 2: Top-b complementary pairs (b=4)
Stage 3: Beam over pairs with 3rd-wing additions
Stage 4: Marginal-value stopping with UCB
Stage 5: Pareto pruning (coverage vs size)
Stage 6: Final evaluation
```

---

## Candidate Budget

- Stage 1: all candidates → top 6 by singleton utility
- Stage 2: all pairs from top 6 → top 4 by (utility + λ_K·max(0,κ))
- Stage 3: beam width = 4, expand with remaining candidates
- Hard cap: max_wings = 6

---

## Stopping Criterion

```
A(w|S) = UCB(ΔU) - λ_C ΔC - λ_R ΔR + λ_K ΔK
Stop when max_w A(w|S) ≤ τ (τ = 0.05)
```

---

## Redundancy / Synergy Settings

```
κ(i,j) = U({i,j}) - U({i}) - U({j})  (measured via parallel graphs)
R(S) = Σ max(0, -κ)
K(S) = Σ max(0, κ)
```

All κ(i,j) negative on current benchmark → all redundancy, no synergy.

---

## Random Seeds

- Training: seed=2718 (data), seed=42 (estimator)
- Evaluation: 50 samples per task, tasks t_ab, t_cd, t_m, t_z

---

## Commit Hash

```
TBD (owner to fill)
```

---

## Results on Development Benchmark

| Arm | Coverage | |G| | Oracle Regret (R_U) | Size Overhead (R_G) |
|-----|----------|-----|---------------------|---------------------|
| Oracle | 0.990 | 3.0 | 0.000 | 0.0 |
| **NICE-R v0.2** | **0.990** | **3.7** | **0.000** | **0.7** |
| Greedy | 0.985 | 6.0 | 0.005 | 3.0 |
| Beam | 0.985 | 6.0 | 0.005 | 3.0 |

---

## Graph Size Distribution (NICE-R v0.2)

| |G| | Count | % |
|-----|-------|---|
| 2 | ~5% | 5% |
| 3 | ~40% | 40% |
| 4 | ~40% | 40% |
| 5 | ~15% | 15% |
| 6 | ~0% | 0% |
| **Mean** | | **3.7** |
| **Median** | | **4.0** |
| **p95** | | **4.0** |
| **Max** | | **4** |

(Oracle distribution: 3 wings 100%)

---

## Next: Held-Out Test + Ablation

1. Run held-out task instances with multiple seeds
2. Ablation study removing one mechanism at a time
3. Report full distributions + regrets