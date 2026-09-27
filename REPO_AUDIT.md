# REPO_AUDIT.md — UMNA-BENCHMARK Repository Audit

**Date:** 2026-09-27  
**Auditor:** opencode agent  
**Commit inspected:** 0c634edb17c89c984eba12d2a495e7d412867e54  
**Branch:** master (default)

---

## 1. Repository Structure Overview

```
UMNA-BENCHMARK/
├── .git/                          # Git repository
├── .pytest_cache/                 # pytest cache (should be ignored)
├── target/                        # Rust build artifacts (should be ignored)
├── Cargo.toml / Cargo.lock        # Rust workspace (in root)
├── layer-b-v0.1/                  # DUPLICATE of bench/ with results
├── runtime-abi/                   # Rust crate (parallel to runtime/)
├── runtime-core/
├── runtime-executor/
├── runtime-graph/
├── runtime-registry/
├── runtime-replay/
├── runtime-scheduler/
├── runtime/                       # Python MVP runtime (active)
├── planner/                       # Python planner (active)
├── bench/                         # Python benchmarks (active)
├── core/                          # Python core (active)
├── wings/                         # Python wing implementations (active)
├── risk/                          # Python risk/calibration (active)
├── tests/                         # Python tests (active)
├── tools/                         # Python audit/lint tools (active)
├── spec/                          # Specification docs
├── docs/                          # Documentation
├── memory/                        # Project memory
├── interference_*.json            # 13+ GENERATED result files (should not be tracked)
├── hypothesis_test_results.json   # GENERATED results (should not be tracked)
├── interference_results.json      # GENERATED results
├── ARCHITECTURE_BLUEPRINT.md      # Generated doc
├── BENCHMARKS.md                  # Generated doc
├── CLAUDE.md                      # Internal config
├── PUBLISH_GUIDE.txt              # Generated/upload artifact
├── UPLOAD_COMMANDS.txt            # Generated/upload artifact
├── TECHNICAL_REFERENCE.md         # Large doc
├── UMNA_v0.1.0.md                 # Spec document
└── REPO_AUDIT.md                  # This file
```

**Current git-tracked files (9):**
- bench/harness.py
- bench/hypothesis_test.py
- bench/interference_multi_seed.py
- hypothesis_test_results.json  ← GENERATED
- interference_multi_seed.json  ← GENERATED
- planner/arms/nice_r.py
- planner/estimator.py
- runtime/admission.py
- runtime/cce.py

---

## 2. Largest Tracked Files (by size)

| File | Size | Status |
|------|------|--------|
| bench/hypothesis_test.py | 27 KB | Source |
| planner/estimator.py | 24 KB | Source |
| bench/harness.py | 15 KB | Source |
| interference_multi_seed.json | 13 KB | **GENERATED - should not track** |
| runtime/cce.py | 13 KB | Source |
| hypothesis_test_results.json | 11 KB | **GENERATED - should not track** |
| bench/interference_multi_seed.py | 5 KB | Source |
| runtime/admission.py | 2.5 KB | Source |
| planner/arms/nice_r.py | 0.9 KB | Source |

---

## 3. Largest Files Modified in Problematic Commit (0c634ed)

| File | Lines Added | Type |
|------|-------------|------|
| bench/hypothesis_test.py | +698 | New source |
| planner/estimator.py | +754 | New source |
| bench/harness.py | +425 | New source |
| runtime/cce.py | +359 | New source |
| interference_multi_seed.json | +594 | **Generated artifact** |
| hypothesis_test_results.json | +458 | **Generated artifact** |
| planner/arms/nice_r.py | +29 | New source |
| bench/interference_multi_seed.py | +165 | New source |
| runtime/admission.py | +76 | New source |

**Total: 3,558 insertions across 9 files**

---

## 4. Generated Files Incorrectly Tracked

| Pattern | Count | Example |
|---------|-------|---------|
| `interference_seed*.json` | 13 | interference_seed42.json (2.2 KB each) |
| `interference_multi_seed.json` | 1 | 13 KB |
| `hypothesis_test_results.json` | 1 | 11 KB |
| `interference_results.json` | 1 | 2.2 KB |
| `__pycache__/` directories | 15+ | planner/__pycache__/, wings/*/__pycache__/ |
| `target/` | 1 | Rust build output (debug/, release/) |
| `Cargo.lock` | 1 | Should not be in root (workspace member) |
| `.pytest_cache/` | 1 | Test cache |

---

## 5. Duplicate/Redundant Directories

| Directory | Issue |
|-----------|-------|
| `layer-b-v0.1/` | Complete duplicate of `bench/` with copied results |
| `runtime-abi/`, `runtime-core/`, `runtime-executor/`, `runtime-graph/`, `runtime-registry/`, `runtime-replay/`, `runtime-scheduler/` | 7 Rust crates duplicating Python `runtime/` functionality; not integrated |
| `runtime/` (Python) | Active MVP implementation per docs/04_LANGUAGES_AND_LAYOUT.md |

---

## 6. Root Cause of GitHub Diff Rendering Problems

**Primary:** Commit 0c634ed adds **3,558 lines in a single commit** including **two large JSON result files** (interference_multi_seed.json: 13 KB, hypothesis_test_results.json: 11 KB). GitHub's diff renderer struggles with:
- Large JSON files (no meaningful diff)
- Single commits touching many unrelated components (benchmark + planner + runtime + results)

**Secondary:** No `.gitignore` exists, so generated artifacts, build output, and caches are mixed with source files in the working tree, making the repository appear disorganized.

---

## 7. Code Quality Problems

| Issue | Location | Severity |
|-------|----------|----------|
| No `.gitignore` | Root | Critical |
| AI-style verbose comments | planner/estimator.py, bench/hypothesis_test.py, runtime/cce.py | High |
| Duplicate docstrings explaining "what" not "why" | Multiple files | Medium |
| `bench/hypothesis_test.py` does too much (698 lines, mixes experiment logic, training, evaluation) | bench/hypothesis_test.py | High |
| `planner/estimator.py` 754 lines - combines estimator, search algorithms, complementarity, multiple planner modes | planner/estimator.py | High |
| Dynamic imports for audit compliance (`_get_execute_run()`) | bench/harness.py, bench/hypothesis_test.py | Medium |
| Redundant calibration class registration duplicated in harness and hypothesis_test | bench/harness.py:311, bench/hypothesis_test.py:231 | Medium |
| Rust crates (`runtime-*`) not built/tested in CI | runtime-*/ | Low |

---

## 8. Documentation Problems

| Issue | Files |
|-------|-------|
| 7+ generated/upload artifacts in root | PUBLISH_GUIDE.txt, UPLOAD_COMMANDS.txt, ARCHITECTURE_BLUEPRINT.md, BENCHMARKS.md, TECHNICAL_REFERENCE.md, UMNA_v0.1.0.md |
| No README.md | Missing |
| Spec docs in `spec/` and `docs/` overlap | spec/mathematics/, spec/algorithms/, docs/ |
| CLAUDE.md is internal tooling config, not project documentation | CLAUDE.md |

---

## 9. Test Status

```
199 tests collected
197 passed
2 failed (test_gates.py::TestGates::test_audit_exits_zero, test_lint_human_exits_zero)
  - Failures are due to lint/style violations in GENERATED files (PUBLISH_GUIDE.txt, UPLOAD_COMMANDS.txt, PRELIMINARY_RESULTS.md)
  - Not source code failures
```

**Test command:** `python3 -m pytest` or `python3 -m unittest discover -s tests`

---

## 10. Recommended Cleanup Actions (Priority Order)

### Immediate (Critical)
1. **Create `.gitignore`** - exclude `__pycache__/`, `target/`, `*.json` results, `Cargo.lock`, `.pytest_cache/`, `.venv/`, `layer-b-v0.1/`, upload artifacts
2. **Remove generated JSON from git tracking** - `git rm --cached hypothesis_test_results.json interference_multi_seed.json interference_seed*.json interference_results.json`
3. **Delete `layer-b-v0.1/`** - duplicate of bench/
4. **Delete upload artifacts** - `PUBLISH_GUIDE.txt`, `UPLOAD_COMMANDS.txt` (or move to docs/ if needed)
5. **Move generated docs** - `ARCHITECTURE_BLUEPRINT.md`, `BENCHMARKS.md` → `docs/` or delete

### Short-term (High)
6. **Refactor `planner/estimator.py`** - split into: estimator.py, search_algorithms.py, complementarity.py
7. **Refactor `bench/hypothesis_test.py`** - split into: experiment.py, evaluation.py, training.py, reporting.py
8. **Consolidate calibration class registration** - single source in bench/utils.py or risk/
9. **Create `README.md`** with: project description, reproduction instructions, status
10. **Separate Rust workspace** - move `runtime-*` to separate repo or `rust/` subdirectory with proper Cargo.toml

### Medium-term
11. **Clean AI-style comments** in source files (keep technical comments: invariants, equations, why)
12. **Add type hints** where missing
13. **Add missing tests** for critical components (contract validation, admission boundary, CCE rejection)
14. **Verify benchmark reproduction** with single command

---

## 11. Scientific Integrity Notes

**Must preserve:**
- All benchmark results in `hypothesis_test_results.json` and `interference_multi_seed.json` (regenerate if needed)
- Negative result: synergy experiment negative for tested family
- CCE functional test: promotion rejected (fail-closed behavior)
- Interference probe: 10 seeds, 0 variance, UMNA/LoRA/Adapter = 0 drift, FullFT = -0.016
- Layer B v0.1 GATED status

**Must not change:**
- NICE-R v0.2 hyperparameters (frozen per paper)
- CCE thresholds (α=0.05/0.10/0.20, ε, E per tier)
- Calibration α=0.1
- Core epoch 1.x hash

---

## 12. Branch Strategy

- Current: `master` (public, shared)
- Cleanup branch: `review-clean` (create from master)
- No force-push to master without explicit owner authorization