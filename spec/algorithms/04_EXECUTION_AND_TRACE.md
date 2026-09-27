# Execution and trace

Developer: Koushik Das - Version 0.1.0

## Scheduling

A plan-first policy. The graph is planned, admitted, compiled and
scheduled before neural inference begins. Independent branches (wings with
no dependency between them, or parallel edges in the wiring) run in
parallel. Sequential chains run in order. A wing failure with no fallback
terminates in abstain.

## The trace

The trace is the measurement instrument. It is emitted whether or not the
run succeeds - a swallowed failure that produces a number you later trust
is worse than a crash.

Every latency term is recorded separately:

T_total = T_retrieve + T_plan + T_admit + T_load + T_transfer + T_wing +
T_fusion + T_risk

- T_retrieve: stage-1 candidate retrieval.
- T_plan: planner search.
- T_admit: calibration-class admission check.
- T_load: module cold load (warm loads are near zero and measured as such).
- T_transfer: host/device and wing-to-wing data movement.
- T_wing: wing compute.
- T_fusion: bridge/task-head fusion.
- T_risk: the post-execution risk test.

A new term of work must never be absorbed silently into T_wing: the
decomposition is the whole point.

## Replay

Every trace records a replay level, never assumed:

| Level | Guarantee |
|---|---|
| L1 bitwise | identical bits on identical hardware |
| L2 numerically close | bounded floating-point difference |
| L3 route-equivalent | same graph selected |
| L4 decision-equivalent | same downstream decision |

L2 does not imply L3. Near-threshold float noise flips routes, which is
why hysteresis and decision margins exist (see
`mathematics/06_numerical_stability.md`).

## Failures

- Wing returns NaN, wrong schema or shape violation: isolated and
  recorded; the graph continues without it if evidence allows, otherwise
  abstains.
- Exceptions are never caught to keep a run alive without recording them
  in the trace.

See flowchart `flowcharts/05_execute_trace.txt`.