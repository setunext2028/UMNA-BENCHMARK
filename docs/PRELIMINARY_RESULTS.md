# UMNA: Preliminary Experimental Results (Layer B v0.1)

**Version:** 0.1.0 | **Date:** 2026-09-24 | **Status:** GATED

---

## 1. Layer B v0.1 — Capability-Isolation Sanity Test

**Purpose:** Validate that the UMNA substrate (frozen Core + write firewall) isolates parameter drift, and that the benchmark can detect interference (FullFT shows it).

### Experimental Design

| Factor | Specification |
|--------|---------------|
| Core | Frozen 2-layer MLP (16→12→8), hash-pinned (SHA-256) |
| Wings | 6 tabular classifiers (α,β: t_ab; γ,δ: t_cd; ε: t_m; ζ: t_z) |
| Methods | UMNA (frozen Core), LoRA (rank=4), Adapter (bottleneck=4), FullFT |
| Seeds | 10 reproducibility runs, deterministic |
| Samples | 500 eval samples/run, 300 train steps |
| Metrics | Target accuracy gain (Δ), Mean \|Protected Δ\|, Worst regression |

### Results

| Method | Target Accuracy Gain | Mean \|Protected Δ\| | Worst Protected Regression |
|--------|---------------------|---------------------|----------------------------|
| **UMNA** | **+0.3240** | **0.0000** | **0.0000** |
| **LoRA** | **+0.3260** | **0.0000** | **0.0000** |
| **Adapter** | **+0.3240** | **0.0000** | **0.0000** |
| **FullFT** | **+0.3140** | **0.0084** | **−0.0160** |

**FullFT per-capability interference:**
| Wing | Δ (FullFT) |
|------|------------|
| α | −0.0160 |
| ε | −0.0140 |
| β | −0.0060 |
| δ | −0.0040 |
| ζ | +0.0020 |

**Multi-seed confidence intervals (10 runs, deterministic → zero variance):**
All CIs collapse to point estimates above.

### Interpretation

> **This experiment supports UMNA's capability-isolation mechanism under frozen-Core conditions. UMNA achieved target improvement comparable to LoRA and Adapter while preserving the evaluated protected capabilities, whereas full fine-tuning produced measurable cross-capability interference.**

**This is a SANITY TEST, not a hypothesis test.** It validates the substrate; it does not test NICE-R or CCE.

---

## 2. Layer B — NICE-R v0.2 Planner Comparison (S5)

### Experimental Design

| Factor | Specification |
|--------|---------------|
| Arms | Greedy, Top-K, Gates, Bounded-Pair, Beam, Oracle, **NICE-R v0.2** |
| Calibration classes | 234 (all combinations up to 6-wing across 4 tasks) |
| Tuning budget | 100 trials per arm (printed in output) |
| Budgets | Compute: 50k FLOPs, Latency: 50ms, Memory: 128MB |
| Tasks | t_ab, t_cd, t_m, t_z (50 samples each) |

### Results

| Arm | t_ab | t_cd | t_m | t_z | **Avg Cov** | **Avg \|G\|** | **R_U** | **R_G** |
|-----|------|------|-----|-----|-------------|---------------|---------|---------|
| Oracle | 0.98 | 0.98 | 1.00 | 1.00 | **0.990** | **3.0** | 0.000 | 0.0 |
| **NICE-R v0.2** | **0.98** | **0.98** | **1.00** | **1.00** | **0.990** | **3.7** | **0.000** | **0.7** |
| Greedy | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 0.005 | 3.0 |
| Top-K | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 0.005 | 3.0 |
| Gates | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 0.005 | 3.0 |
| Beam | 0.98 | 0.98 | 1.00 | 0.98 | 0.985 | 6.0 | 0.005 | 3.0 |
| Pairs | 0.62 | 0.66 | 0.78 | 0.66 | 0.680 | 2.0 | 0.310 | -1.0 |

**NICE-R v0.2 achieves Oracle-level coverage (0.990) with 3.7 wings vs Oracle's 3.0.**

### Graph Size Distribution (NICE-R v0.2)

| \|G\| | Frequency |
|------|-----------|
| 2 | 5% |
| 3 | 40% |
| 4 | 40% |
| 5 | 15% |
| 6 | 0% |
| **Mean** | **3.7** |
| **Median** | **4.0** |
| **p95** | **4.0** |
| **Max** | **4** |

*(Oracle: 3 wings, 100%)*

### Ablation Study (Single Packet, t_cd)

| Configuration | \|G\| | Key Finding |
|--------------|------|-------------|
| **Full v0.2** | **4** | Baseline |
| No marginal stop (τ=∞) | 2 | Stops at initial pair |
| **No redundancy (λ_R=0)** | **6** | **Explodes — critical** |
| No complementarity (λ_K=0) | 4 | No effect (all κ < 0) |
| No UCB (κ_ucb=0) | 3 | Slightly smaller |
| No cost | 4 | Cost not primary driver |
| No pair seeding | 4 | Seeding helps but not critical |

> **Key finding:** Redundancy penalty (λ_R) is the main mechanism preventing explosion to 6 wings.

---

## 3. Hypothesis Test — Route Displacement & Protected Drift

### Experimental Design

| Factor | Specification |
|--------|---------------|
| Target wing | γ (t_cd) |
| Methods | UMNA, LoRA, Adapter, FullFT |
| Seeds | 3 (42, 123, 456) |
| Eval samples | 300 per run |
| Metrics | Target Δ, Protected drift (under routing vs isolated), Route displacement (JS) |

### Results

| Method | Target Δ | Protected Drift (Under Routing) | Protected Drift (Isolated, FullFT) | Route JS |
|--------|----------|--------------------------------|-----------------------------------|----------|
| UMNA | +0.319 / +0.309 / +0.299 | 0.0000 | 0.0000 | 0.0000 |
| LoRA | +0.319 / +0.299 / +0.304 | 0.0000 | 0.0000 | 0.0000 |
| Adapter | +0.319 / +0.309 / +0.299 | 0.0000 | 0.0000 | 0.0000 |
| FullFT | +0.279 / +0.294 / +0.284 | 0.0000 | α:−0.040, β:−0.027, δ:+0.043, ε:+0.023, ζ:+0.003 | 0.0000 |

**Key finding:** Under production routing, all parameter-isolation methods (UMNA, LoRA, Adapter) show zero protected drift. FullFT shows parameter drift in isolation but zero drift under routing (route displacement JS=0). This validates the measurement methodology: protected wings evaluated under production routing, not in isolation.

---

## 4. Scaling Probe (S7)

| N \ M | M=1 | M=5 | M=10 | M=20 | M=50 |
|-------|-----|-----|------|------|------|
| 10 | 2.2 | 2.6 | 2.7 | N/A | N/A |
| 50 | 3.0 | 15.3 | 23.0 | 24.4 | 23.7 |
| 100 | 3.4 | 15.9 | 34.0 | 54.1 | 49.7 |
| 200 | 2.5 | 19.8 | 34.8 | 70.1 | 105.8 |
| 500 | 2.9 | 17.9 | 42.7 | 69.1 | 188.1 |
| 1000 | 2.9 | 16.6 | 37.5 | 74.6 | 198.1 |
| 2000 | 2.6 | 15.8 | 35.2 | 65.4 | 180.7 |
| 5000 | 2.8 | 18.3 | 37.0 | 74.0 | 183.0 |

**Recall@M = 1.000** for all N, M (proxy wings match query types perfectly).

**Latency scaling (μs/entry):**
- M=1: ~0.0000 (flat)
- M=5: 0.0014
- M=10: **0.0024** (linear)

---

## 5. Synergy Probe (S6 — F3 Verdict)

| Mode | Synergy Recall | vs Random |
|------|----------------|-----------|
| Greedy (Mode A) | 1.000 | — |
| Bounded-Pair (Mode B) | 0.000 | **0.0 advantage** |
| Random baseline | 0.060 | — |

> **Verdict:** Mode B REJECTED for this family (F3). Bounded-pair does not significantly outperform random pairing.

---

## 6. System Gates (All Passing)

| Gate | Status | Evidence |
|------|--------|----------|
| Unit tests | PASS | 197 passed |
| Lint (no emojis, forbidden vocab) | PASS | Clean |
| Audit (mechanical checks) | PASS | Clean |
| Scaling probe | PASS | Recall=1.0, linear |
| Synergy probe | PASS | Mode B rejected |
| S5 harness | PASS | Admission >0% all arms |
| Layer B v0.1 | PASS | 10 seeds, zero variance |

---

## 7. Reproducibility

```bash
# All benchmarks from single command
python3 -m bench.harness          # S5 planner comparison
python3 -m bench.interference_multi_seed  # Layer B (10 seeds)
python3 -m bench.hypothesis_test  # Route displacement + drift
python3 -m bench.scaling_probe    # S7
python3 -m bench.synergy_probe    # S6
python3 -m pytest tests/ -q       # 197 tests
python3 tools/lint_human.py       # Style check
```

**All results deterministic** — 10 seeds produce zero variance.

---

## 8. What These Results Do NOT Claim

| Not Claimed | Reason |
|-------------|--------|
| "UMNA beats LoRA/Adapter" | Layer B shows parity on parameter isolation |
| "NICE-R hypothesis confirmed" | Layer C (dynamic routing) not yet run |
| "CCE hypothesis confirmed" | Sequential promotion + drift not yet tested |
| "Superiority over FullFT" | FullFT is not a deployment baseline |

---

## 9. Next Experimental Targets

| Layer | Hypothesis | Blocker |
|-------|------------|---------|
| **Layer C v0.1 (H1)** | NICE-R approaches Oracle Pareto frontier on held-out tasks | Held-out planner test (multiple seeds, fresh tasks) |
| **H1 Final** | Frozen protocol for routing Pareto | Layer C v0.1 working |
| **H3 v0.1 (CCE)** | CCE promotion controls cumulative drift | Sequential promotion + drift tracking |
| **H3 Final** | Frozen CCE protocol | H3 v0.1 working |

---

## Appendix: Compute & Hardware

| Component | Specification |
|-----------|---------------|
| Hardware | CPU-only (Python reference) |
| Precision | float32 |
| FLOP budget | 50k per graph |
| Latency budget | 50ms per graph |
| Memory budget | 128MB per graph |
| Runtime | ~2 min full benchmark suite |