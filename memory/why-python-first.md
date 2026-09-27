---
name: why-python-first
type: decision
stage: S0
---

The whole MVP is Python. Rust starts at S6 or later, and only for the control
plane.

**Why:** the MVP question is whether NICE-R picks better graphs than top-k or
greedy. That is a planner-quality question, not a runtime-engineering one.
Writing the registry, type checker, and scheduler in Rust before S5 has an
answer means carrying correctness for code that may be deleted, and slows
every iteration of the part that actually matters. The alternative considered
was Rust-first for the control plane with Python models over FFI; rejected
because the FFI boundary would have to be redesigned anyway once the planner
settles.

**How to apply:** design the S0 seams so the port is cheap — stable types,
no Python-specific structure leaking across a seam. Design for the port, do
not perform it early. Do not read this as permission to write throwaway code.

Related: [[stage-pointer]] [[admission-between-plan-and-execute]]
