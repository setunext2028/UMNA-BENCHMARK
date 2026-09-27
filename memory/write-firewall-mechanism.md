---
name: write-firewall-mechanism
type: decision
stage: S3
---

The parameter write firewall is implemented by making Core parameter arrays read-only via `ndarray.setflags(write=False)`. During Wing training:

1. The frozen Core artifact is loaded and its parameters are mounted as read-only views.
2. The trainer computes gradients through the Core (backpropagation traverses the Core to reach Wing parameters).
3. The optimizer only receives and updates Wing-owned parameters (transform and head).
4. Any attempt to write to a Core parameter array raises `ValueError` (NumPy's read-only flag enforcement).

This implements Section 6 of the paper: "Gradients do flow through the frozen Core... What is forbidden is a *write*."

**Why:** Alternative approaches considered:
- `np.memmap` with read-only mode: adds file I/O complexity and platform differences.
- Wrapper classes with `__setattr__` override: more complex, harder to verify.
- Copy-on-write: would silently allow mutations, violating the write firewall guarantee.

NumPy's built-in `setflags(write=False)` is the simplest, most stable mechanism — it's enforced at the C level and cannot be bypassed accidentally.

**How to apply:** Any new training code that receives Core parameters must verify `assert not arr.flags.writeable` before use. The CI check in `tools/audit.py` (function `assert_wing_through_core_isolates_writes`) verifies this property.

Related: [[s3-core-training]] [[s3-task-family-and-roles]] [[docs/12_TRAPS.md]]
