# Compute physics

Developer: Koushik Das - Version 0.1.0

## The quantity

C(S): active FLOPs of the graph. Only wings actually scheduled for this
input count. Fused dense inference activates everything; sparse routing
activates a subset, which is the entire point of the architecture.

## Where compute is born

| Source | Term | Notes |
|---|---|---|
| Wing execution | T_wing contribution | measured per wing per input |
| Bridge / fusion | T_fusion contribution | measured when fusion runs |
| Planner search | T_plan | counts fully: search is compute |
| Shadow execution | s * C_shadow | amortised into every efficiency figure |
| Risk test | T_risk | counts: conformal scoring is compute |

## The budget

C(S) <= B_C is a constraint, not a penalty. The planner never trades a
FLOP against quality via a tuned coefficient in the default configuration
(lambda = 0).

## The trap

A system that appears efficient only by excluding the cost of the
mechanism that keeps its router trained is not measured honestly. If
efficiency numbers exclude shadow traffic, they exclude the cost of the
mechanism itself (paper 11.2).

## Measurement rule

active_flops is recorded per trace, per input, not averaged across the
batch. The report's active-FLOP column is the mean of per-input values
with the same decomposition discipline as latency.