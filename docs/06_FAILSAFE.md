# Failsafe

Developer: Koushik Das · Version 0.1.0

Failsafe here means: when something goes wrong, the system does something
*boring* rather than something confident.

## The default is abstain

Every terminal path that is not a clean accept ends in abstain:

- risk criterion not met, no budget to expand → abstain
- conflicting evidence above the action threshold → **abstain and escalate**
- no admissible graph → abstain
- Wing failure with no fallback → abstain
- budget exhausted mid-plan → abstain

Never resolve an unresolved state by picking the higher score. That is the
single most tempting shortcut in this codebase and it silently converts a
detected problem into an undetected one.

Abstention is not free. It is counted, reported, and capped per task family.
A system that abstains its way to good numbers has failed differently.

## Enforced at construction

These are cheap and absolute. Build them in S1, never relax them.

| Guard | Enforced where |
|---|---|
| Optimiser sees only owned parameters | training setup |
| Core and old-Wing hashes unchanged after training | CI assertion |
| Manifest cannot declare host-owned metrics | manifest validator, allow-list |
| Type-invalid graph cannot be built | `graph.build` |
| Core epoch mismatch refused | registry resolve |
| Version diamond refused (one Core epoch per deployment) | registry resolve |
| Inadmissible graph cannot serve | `admission.admit` |
| Budgets are constraints, not penalties | `planner.plan` |

## Fail loudly, in the right direction

- A Wing that returns NaN, a wrong schema, or a shape violation is **isolated
  and recorded**, and the graph continues without it if evidence allows,
  otherwise abstains.
- A Wing whose measured behaviour drifts from its admission profile is
  **demoted from routing eligibility** and flagged. Do not quietly keep using it.
- Never catch an exception to keep a run alive without recording it in the
  trace. A swallowed failure is worse than a crash, because it produces a
  number you will later trust.

## The honesty guards

These protect the *results*, which is the thing most easily corrupted:

- Every arm gets the same tuning budget, printed in the output.
- Every latency term is reported, including `T_plan`, `T_admit`, and
  amortised shadow cost. Reporting Wing compute alone is not a measurement.
- Oracle regret is stated against its declared restriction, never against an
  unreachable optimum.
- Replay claims name their level (L1–L4). L2 does not imply L3 — near-threshold
  float noise flips routes, which is exactly why hysteresis and decision
  margins exist.

## Near-threshold stability

Control flow branches on continuous values, so small numerical differences
change which Wing runs. Every control-path comparison uses hysteresis
(`tau_enter > tau_exit`), a decision margin (`delta_route`), or quantised
bins. Never compare a raw float against a threshold on the control path.
