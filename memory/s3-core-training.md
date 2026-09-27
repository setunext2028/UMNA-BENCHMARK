---
name: s3-core-training
type: result
stage: S3
---

Core training (epoch 1.x) hyperparameters and results:

**Hyperparameters (fixed, from `core/train.py`):**
- `CORE_SEED = 1729`
- `DATA_SEED = 2718`
- `CORE_STEPS = 1200`
- `CORE_BATCH_SIZE = 128`
- `CORE_LEARNING_RATE = 0.04`
- `CORE_WEIGHT_DECAY = 1e-4`
- Architecture: 8 -> 16 (ReLU) -> 12 (ReLU) -> 8 (linear latent)
- Joint multi-task training with 4 probe heads (one per honest task)

**Training probe accuracy on holdout (seed 2718):**
- `t_ab`: 0.854
- `t_cd`: 0.761
- `t_m`: 0.789
- `t_z`: 0.676

**Core artifact:**
- Path: `core/artifacts/core-1.x-8d.npz`
- SHA256: `c7acf73fea428e087ebe91d1b3d2089d8c7be184326399ec849d7959414a96ab`
- Frozen parameters: w0(16,12), b0(12), w1(12,8), b1(8) — all mounted read-only

**Wing training through frozen Core (from `core/firewall.py`):**
- `WING_SEED`: per-wing (1-6)
- `WING_STEPS = 600`
- `WING_BATCH = 64`
- `WING_LR = 0.05`
- `WING_WEIGHT_DECAY = 1e-4`
- Architecture per wing: 8 -> 16 (ReLU, transform) -> Core trunk -> 8 -> 1 (head)

**Wing artifacts (SHA256):**
- alpha: `c8d79b0f9e2f6c59` (redundant, t_ab, features 0,1)
- beta: `6c84eeeb633d8b33` (redundant, t_ab, features 0,1)
- gamma: `4d929507be6e5ea6` (complementary, t_cd, feature 2)
- delta: `3c1d6d979c86d986` (complementary, t_cd, feature 3)
- epsilon: `5807f4de50a71c4c` (misleading, t_m corrupted, feature 4)
- zeta: `de200192ff3f89d6` (cheap-but-weak, t_z, feature 7)

All wing artifacts include transform weights, head weights, core_epoch, core_sha256, and wing_sha256. Core hash verified unchanged after each wing training.

Hardware: CPU, NumPy 1.26.4, single-threaded. Seeds fixed for exact reproducibility.

Related: [[write-firewall-mechanism]] [[s3-task-family-and-roles]] [[stage-pointer]]
