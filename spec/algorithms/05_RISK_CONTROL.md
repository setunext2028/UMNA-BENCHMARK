# Risk control

Developer: Koushik Das - Version 0.1.0

## The three risk mechanisms, deconflicted

| Mechanism | Stage | Question | Status |
|---|---|---|---|
| Calibration-class admission | pre-execution, hard gate | Is this graph allowed to serve at all? | construction-enforced |
| A_hat(S, x) predicted abstention | planning, soft term | Will this graph likely fail the risk test and waste compute? | research component |
| Conformal / selective risk | post-execution, authoritative | Does this output meet the risk target? | conditional on method assumptions |

The old planning-time penalty R(S) (estimated risk of regression or
calibration failure) is removed from the objective: it is not estimable at
plan time for an unvalidated graph, and it is already settled by class
admission for a validated one.

## Post-execution test

1. Output is produced by the fused graph under the binding.
2. The binding's selective-risk criterion is evaluated.
3. Criterion met: ACCEPT.
4. Not met, adaptive budget remains: EXPAND - one bounded expansion round,
   returning to ADMISSION (a grown graph is a different binding).
5. Not met, no budget: ABSTAIN.

## Abstain is the safe default

Every terminal path that is not a clean accept ends in abstain:
- risk criterion not met, no budget to expand;
- conflicting evidence above the action threshold (abstain AND escalate);
- no admissible graph;
- wing failure with no fallback;
- budget exhausted mid-plan.

Never resolve an unresolved state by picking the higher score. That is the
single most tempting shortcut and it silently converts a detected problem
into an undetected one.

## Coverage and abstention

Coverage = 1 - abstention rate. Abstention is counted, reported and capped
per task family. A system that abstains its way to good numbers has failed
differently, and exceeding the family's maximum acceptable abstention rate
blocks promotion exactly as a task-score regression would.

See flowchart `flowcharts/06_risk_expand.txt`.