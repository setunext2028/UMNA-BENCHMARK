# NICE-R v0.2 Plan — Marginal Utility, Complementarity, and Stop

**Status:** DRAFT
**Target:** Close the Oracle gap: (coverage, |G|) = (0.985, 6) → (0.990, 3)

---

## Current S5 Result

| Arm | t_ab | t_cd | t_m | t_z | Avg Coverage | Avg |G| | Admission |
|-----|------|------|-----|-----|--------------|-----|-----------|
| greedy | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 100% |
| topk | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 100% |
| gates | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 100% |
| pairs | 0.62 | 0.66 | 0.78 | 0.66 | 0.680 | 2.0 | 100% |
| beam | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 100% |
| **oracle** | 0.98 | 0.98 | 1.00 | 1.00 | **0.990** | **3.0** | 100% |

**Problem:** All search arms (greedy, topk, gates, beam) select 6 Wings. Oracle shows 3 Wings suffice for ~0.990 coverage.

---

## Diagnostic First: Hard-cap Beam at |G| ≤ 3

Run current beam with hard cap |G| ≤ 3.

* If coverage ≈ 0.99, |G|=3 → search/ranking is good, stopping criterion is the bug.
* If coverage = 0.80–0.90, |G|=3 → search isn't finding the right 3-Wing composition.

---

## Core Change: Marginal-Value Stopping Rule

Change planner question from:

> "Which graph has the largest predicted utility?"

to:

> "Does adding this Wing provide enough **incremental value** to justify its additional cost and interaction risk?"

### Formalization

For current graph S and candidate Wing w:

**Marginal utility:**
```
ΔÛ(w|S) = Û(S ∪ {w}, x) - Û(S, x)
```

**Incremental cost:**
```
ΔC(w|S) = C(S ∪ {w}) - C(S)
```

**Admission value:**
```
A(w|S) = ΔÛ(w|S) - λ_C ΔC(w|S) - λ_R R(w,S) + λ_K K(w,S)
```

* R(w,S) = redundancy/interference penalty
* K(w,S) = complementarity/synergy bonus

**Stop when:**
```
max_w A(w|S) ≤ τ
```

---

## Cost Model (Not Just Graph Size)

```
C(S) = α F(S) + β L(S) + γ M(S) + η |S|
```

* F = active FLOPs (host-measured)
* L = latency (ms)
* M = memory (MB)
* |S| = composition complexity (proxy)

For current harness, |S| can serve as proxy while debugging.

---

## Explicit Complementarity Modeling

Define pair complementarity:
```
κ(i,j) = U({i,j}) - U({i}) - U({j}) + U(∅)
```

* κ > 0: synergistic pair
* κ ≈ 0: additive
* κ < 0: redundant/interfering

Learn/estimate `κ̂(i,j|x)` to seed search from promising pairs.

### Pair-Seeded Beam Search Structure

```
INPUT: N Wings

Stage 1: Retrieve top-m plausible Wings (N → m)
Stage 2: Score singletons
Stage 3: Generate top-b complementary pairs
Stage 4: Beam over those pairs
Stage 5: Test best third-Wing additions
Stage 6: Marginal-value stopping
Stage 7: Pareto prune
Stage 8: Exact evaluation of small finalist set
```

Example (N=6):
```
m = 6, pair seeds = 4, beam width = 4, finalists = 3
```

Scales to N=1000:
```
1000 → 20 (Stage 1), then combinatorial search
```

---

## Pareto Pruning

Graph A dominates B if:
```
Q(A) ≥ Q(B) AND C(A) ≤ C(B) AND at least one strict
```

Delete dominated graphs from beam immediately.

Retains:
```
high quality / low cost
high quality / moderate cost
slightly lower quality / very low cost
```

---

## Redundancy Penalties

Define similarity/redundancy between Wings:
```
ρ(i,j)
R(S) = Σ_{i<j∈S} ρ(i,j)
```

Objective:
```
J(S,x) = Û(S,x) - λ_C C(S) - λ_R R(S) + λ_K K(S)
```

Prefers 3 complementary Wings over 6 redundant Wings.

---

## Confidence-Aware Stopping

Utility estimator outputs:
```
ΔÛ(w|S) and σ_ΔU(w|S)
```

Optimistic bound (UCB):
```
UCB(w|S) = ΔÛ(w|S) + κ σ(w|S)
```

**Stop if even optimistic bound can't pay for cost:**
```
UCB(w|S) - λ ΔC(w|S) ≤ 0
```

---

## Separate Search Failure from Estimation Failure

Run three systems on development benchmark:

| System | Utility | Search | Purpose |
|--------|---------|--------|---------|
| A. Restricted Oracle | true U | exhaustive | Upper bound |
| B. NICE-R + true U | true U | NICE-R search | Search error |
| C. NICE-R normal | estimated Û | NICE-R search | Full error |

**Decomposition:**
```
R_search = J(S_oracle) - J(S_search,true)      (search error)
R_estimation = J(S_search,true) - J(S_search,Û) (estimation error)
```

Only modify the component that's actually responsible.

---

## Experiment Sequence (C1–C9)

| Exp | Configuration | Expected Insight |
|-----|---------------|------------------|
| C1 | Current beam | Baseline: 0.985 / 6 |
| C2 | Beam hard cap = 3 | Diagnostic: is search good? |
| C3 | Beam cap 3 + true U | Isolate estimation error |
| C4 | Pair-seeded beam + true U | Does complementarity help? |
| C5 | Pair-seeded beam + Û | Full pipeline test |
| C6 | + marginal stopping | Does stop rule work? |
| C7 | + redundancy penalty | Does it reduce |G|? |
| C8 | + Pareto pruning | Does it clean beam? |
| C9 | + uncertainty-aware stop | Robustness to noise |

Record after each: coverage, |G|, regret, latency, FLOPs.

---

## Formal NICE-R v0.2 Objective

```
J(S,x) = Û(S,x) - λ_C C(S) - λ_R R(S) + λ_K K(S)
```

Subject to:
```
|S| ≤ K_max
L(S) ≤ B_L
M(S) ≤ B_M
S ∈ registered calibration class
```

**Expansion continues while:**
```
max_w [ UCB(ΔU(w|S)) - λ_C ΔC - λ_R ΔR + λ_K ΔK ] > 0
```

**Otherwise STOP.**

---

## Success Criteria

Not memorizing (0.990, 3) on this benchmark.

**Target:** A search principle that approaches Oracle's quality-cost Pareto frontier on held-out problems.

Example scientifically meaningful result:
```
Oracle          0.982 / 3.1
NICE-R          0.979 / 3.3
Greedy          0.978 / 5.8
Top-k           0.977 / 6.0
```

---

## Files to Modify

1. `planner/estimator.py` — add `_build_graph_with_wiring_patterns`, marginal utility, complementarity, Pareto pruning
2. `planner/arms/beam.py` — pair-seeded beam, hard-cap option, marginal stopping
3. `planner/arms/greedy.py` — marginal-value stopping
4. `planner/arms/pairs.py` — complementarity seeding
5. `planner/arms/oracle.py` — restricted search with same objective
6. `bench/harness.py` — diagnostic modes (C1–C9), decomposition reporting
7. `planner/interaction.py` — complementarity estimator κ̂(i,j|x)

---

## Gate for NICE-R v0.2

Per `docs/03_BUILD_ORDER.md` S5 gate:
> "the full comparison table, reproducible from one command, with per-arm tuning budget printed in the output"

Add decomposition:
- Search error vs estimation error
- Pareto frontier plots (coverage vs |G|)
- Ablation of each component (stop, redundancy, complementarity, Pareto)