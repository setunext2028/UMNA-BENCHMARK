# Data-flow physics

Developer: Koushik Das - Version 0.1.0

## The three flows

```
1. CONTROL FLOW  (host -> wings)
   input -> retrieval -> plan -> admit -> execute -> risk -> serve/abstain
   every hop is a typed value; every hop costs a latency term

2. GRADIENT FLOW (wings -> core, during training only)
   dL/d(theta_wing) traverses the frozen Core; the Core's own update is
   blocked. Gradients pass THROUGH; writes do not. This is the write
   firewall, and it is the only wall.

3. EVIDENCE FLOW (wings -> head -> decision)
   each wing returns WingOutput with evidence packets; contradictory
   packets coexist; the ConflictSet is passed to the decision layer.
```

## The transfer term

T_transfer exists between every pair of steps: host/device movement,
wing-to-wing movement of latent packets, and result collection. It is
measured per trace, not estimated from wing count. A graph with more wings
has more transfer surface; parallel branches transfer concurrently and
their transfer cost overlaps their compute cost - the scheduler records
both.

## Latency budget as throughput physics

The budget L(S) <= B_L is end-to-end. Planning (T_plan), admission
(T_admit) and cold load (T_load) are upstream of the first wing compute,
so they are paid by every input, sparse or not. This is the physics that
can falsify the speed claim (F2): planner, admission and orchestration
overhead can erase sparse-compute savings.

## Shadow cost

Amortised overhead, per input:

T_bar = T_live + s * T_shadow
C_bar = C_live + s * C_shadow

s is a declared deployment parameter. Live-only figures are not a
measurement of the system; they are a measurement of a system without its
router-learning mechanism.