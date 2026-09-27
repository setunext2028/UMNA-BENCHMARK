---
name: admission-between-plan-and-execute
type: invariant
stage: S4
---

`admit()` sits between planning and execution. No call path reaches
`execute.run` without passing it. Graph expansion returns to admission, not
to execution.

**Why:** calibration attaches to a binding, and the binding includes the
graph. A planner that builds graphs per input therefore produces uncalibrated
bindings by default, and no coverage guarantee survives to production. The
admission check is what keeps dynamic planning and calibrated risk control
from cancelling each other out.

**How to apply:** this looks like an extra hop to anyone tidying the call
path, and inlining it into `plan()` will seem like a simplification. It is
not — it removes the gate. If a refactor touches the plan/execute boundary,
check this invariant explicitly and add `T_admit` to the trace.

Related: [[why-python-first]]
