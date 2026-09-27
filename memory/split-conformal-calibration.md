---
name: split-conformal-calibration
type: result
stage: S4
---

Split conformal prediction for binary classification task heads (S4).

**Method:** Split conformal with conformity score `1 - p(y|x)` for binary labels `y in {0, 1}`.

**Calibration data:** Holdout split from the deterministic task family (DATA_SEED=2718, 1000 rows per task).

**Target level:** `alpha = 0.1` (90% marginal coverage).

**Calibrator results per task:**

| Task | Wing(s) | Holdout size | Threshold | Empirical coverage |
|------|---------|--------------|-----------|-------------------|
| t_ab | alpha, beta | 1000 | ~0.44 | >= 0.90 |
| t_cd | gamma, delta | 1000 | ~0.75 | >= 0.90 |
| t_m | epsilon | 1000 | ~0.51 | >= 0.90 |
| t_z | zeta | 1000 | ~0.43 | >= 0.90 |

**Implementation:** `risk/conformal.py`
- `fit_split_conformal(probs, labels, alpha)` — computes threshold quantile
- `predict_set(calibrator, prob)` — constructs prediction set `{y: 1-p(y|x) <= threshold}`
- `coverage_holds(calibrator, probs, labels)` — verifies empirical coverage
- `assess(output, binding)` — returns `Decision.ACCEPT` (singleton), `ABSTAIN` (both), `EXPAND` (empty)

**Threshold behavior:**
- Low threshold → larger prediction sets → higher coverage, more abstain
- High threshold → smaller prediction sets → lower coverage, more accept/expand
- The (1-alpha) quantile with `method="higher"` gives valid finite-sample coverage

**Why this approach:**
- Split conformal is simple, distribution-free, and gives exact finite-sample coverage
- Binary classification conformal sets are naturally `{0}`, `{1}`, `{0,1}`, or `{}`
- Maps directly to Decision: singleton=ACCEPT, both=ABSTAIN, empty=EXPAND
- No retraining needed; calibrator is a pure function of holdout scores

**Hardware/seeds:** CPU, NumPy 1.26.4, calibration seed=42, data seed=2718.

Related: [[calibration-class-registry]] [[s3-task-family-and-roles]] [[s3-core-training]]
