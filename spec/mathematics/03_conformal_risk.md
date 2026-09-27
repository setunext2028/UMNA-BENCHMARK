# Conformal risk control

Developer: Koushik Das - Version 0.1.0

## The mechanism

The uncertainty layer is calibrated selective prediction / conformal risk
control on the task head - never a confidence ceiling.

- Conformal prediction: finite-sample coverage guarantees under
  exchangeability, distribution-free.
- Risk-controlling prediction sets: control the expected risk of the
  prediction set, not just coverage.
- Selective classification: abstain below a calibrated threshold.

## The selection problem

Routing IS a selection procedure - the planner picks the binding per
input. Marginal coverage guarantees do not automatically survive selection.
This is why admission exists: the planner's production freedom is the union
of pre-registered calibration classes, so the served set of graphs is
exactly the set whose calibrators were validated under routing-induced
selection.

Let Phi be a calibration equivalence class:

Phi = { G : Calibrator(C_e, G, H, D, Q) is valid at level alpha }

Membership is established empirically at admission time, never asserted.

## Coverage and abstention

Coverage = 1 - AbstentionRate. Three independent mechanisms push toward
abstention (risk-test failure, exhausted candidates, fault fallback), so
coverage and abstention are first-class metrics. Each task family declares
a maximum acceptable abstention rate; exceeding it blocks promotion exactly
as a task-score regression would.

## Evaluation (Experiment G)

For each binding: expected calibration error, Brier score where
appropriate, conformal coverage, prediction-set size, selective risk,
abstention rate, coverage-risk curve.

Arms compared: raw model confidence; post-hoc calibrator; ARBITRARY
confidence clamp (negative control - clamping a well-calibrated score
downward manufactures underconfidence); conformal / risk-control methods.

Coverage is ALSO measured conditioned on the planner's graph choice,
because marginal coverage is not the quantity of interest when a router
selects the binding.