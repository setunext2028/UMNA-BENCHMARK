---
name: s3-task-family-and-roles
type: decision
stage: S3
---

The S3 tabular task family is defined in `core/tasks.py`. One modality (8-dimensional real vectors in [-1,1]^8), four honest binary tasks:

- `t_ab`: label = sign(x[0] + x[1]) — features 0,1
- `t_cd`: label = sign(x[2] + x[3]) — features 2,3
- `t_m`: label = sign(x[4]) — feature 4
- `t_z`: label = sign(x[7]) — feature 7

Each task has 5% label noise (deterministic via `DATA_SEED=2718`). Training/holdout split: 3000/1000 rows.

Six Wings trained through the frozen Core, each with a distinct role:

| Wing   | Role              | Task  | Features seen | Notes |
|--------|-------------------|-------|---------------|-------|
| alpha  | redundant         | t_ab  | (0, 1)        | identical task/features to beta, different seed |
| beta   | redundant         | t_ab  | (0, 1)        | identical task/features to alpha, different seed |
| gamma  | complementary     | t_cd  | (2,)          | only feature 2; alone weak, strong with delta |
| delta  | complementary     | t_cd  | (3,)          | only feature 3; alone weak, strong with gamma |
| epsilon| misleading        | t_m   | (4,)          | trained on permuted labels (corrupted signal) |
| zeta   | cheap-but-weak    | t_z   | (7,)          | trivial single-feature task, minimal capacity |

**Why:** This structure exercises the MVP requirements:
- Redundant pair (alpha/beta): tests planner's handling of duplicate capability
- Complementary pair (gamma/delta): sets up S6 synergy probe (weak alone, strong together)
- Misleading (epsilon): tests risk/calibration (looks capable on corrupted training, fails on holdout)
- Cheap-but-weak (zeta): tests cost-aware planning (low compute, low accuracy)

Feature masking for complementary wings is applied at training time (zeroing non-assigned features). All Wings share the same 8-dim input interface; feature selection is implicit in the learned transform weights.

Related: [[write-firewall-mechanism]] [[s3-core-training]] [[stage-pointer]]
