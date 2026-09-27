# runtime/admission.py

Developer: Koushik Das - Version 0.1.0

Calibration-class admission. The load-bearing invariant of the
architecture: no call path reaches execute.run without passing admit.

Imports abi and graph. Imported by the executor and by anyone who
assembles a serving path.

## Types

### Admission
- **Contract:** PRODUCTION or SHADOW_ONLY.
- **How it works:** the result of admitting a planned graph. PRODUCTION
  graphs may serve; SHADOW_ONLY graphs may run for data and logging, never
  serve.

## Functions (S4)

### admit(graph, binding)
- **Contract:** decide whether a planned graph may serve.
- **How it works (S4):** look up the calibration classes for the binding;
  if the graph belongs to a registered class with a currently valid
  calibrator, return PRODUCTION. Otherwise return SHADOW_ONLY and let the
  caller decide between falling back to the nearest admissible graph
  (allowed) and abstaining (default when no such graph exists).
- **Failure modes:**
  - the class registry is empty: every graph is shadow-only. This is the
    correct bootstrapping state, not an error to patch around.
  - a calibrator whose validity has lapsed: SHADOW_ONLY, and the lapse is
    recorded — never silently re-validated.
- **Boundary:** admit never plans and never executes. Expansion returns
  HERE (a grown graph is a different binding), and anyone tempted to
  inline it into plan() for tidiness is removing the gate: this is the
  invariant in memory/admission-between-plan-and-execute.md.