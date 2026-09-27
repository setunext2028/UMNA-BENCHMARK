---
name: strict-type-unification-in-s1
type: decision
stage: S1
---

S1 type unification requires exact `TypeExpr.family` and exact ordered
arguments. There is no wildcard, conversion, covariance, or subtyping.

**Why:** S1 exists to separate control-plane failures from model failures.
Before S3 has real latent semantics, a permissive relation would invent
compatibility and could let an invalid edge through while making the type
checker appear successful. Exact matching is the smallest rule that can
fail honestly.

**How to apply:** keep `type_unifies()` strict through S2. If real Wings in
S3 require subtyping or conversion, make that a separate typed rule with
failure tests; do not loosen equality silently.

Related: [[stage-pointer]]
