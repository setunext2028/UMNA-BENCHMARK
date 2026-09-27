# Calibration classes and drift promotion

Developer: Koushik Das - Version 0.1.0

## The binding

B = (C_e, G, H, D, Q)

- C_e: Core epoch
- G: executed Wing/Bridge graph
- H: task head
- D: validation/deployment distribution
- Q: hardware/numerical execution profile

Calibrator = Calibrator(B). Changing any load-bearing component creates a
new binding version. Calibration belongs to the binding, never to a Wing:
a reusable Wing must not own calibration specific to a downstream head.

## The equivalence class

Phi = { G : Calibrator(C_e, G, H, D, Q) is valid at level alpha }

Two hard rules:

1. Production admission: the planner may emit a graph to production only if
   it belongs to some registered class with a currently valid calibrator.
2. Shadow-only novelty: a graph outside every registered class is
   executable in shadow mode only; its outputs are counterfactual logs
   that qualify a new class for registration through the promotion
   protocol.

## Step and origin drift

Origin performance P(0) is permanently pinned; it is the one field that is
never re-snapshotted by promotion.

Step drift:   Delta_step(i,t)  = P(t) - P(t-1)   >= -epsilon_i
Origin drift: Delta_origin(i,t) = P(t) - P(0)    >= -E_i

Promotion requires BOTH. Without the origin constraint, T sequential
promotions each passing at -epsilon i can degrade capability i by up to
T * epsilon i with every gate green.

## Power budget for Tier A

Under family-wise error control, per-test alpha shrinks roughly as 1/|A|.
At fixed effect size and sample size, power falls as Tier A grows:

|A|max = max { k : power(delta, n, alpha/k) >= 1 - beta }

Computed and published before the first promotion; recomputed whenever n or
delta changes. Designating more Tier A capabilities than |A|max is not
conservatism - it is running underpowered tests.

## Promotion interference

The interference matrix must not be evaluated where zeros are guaranteed
(pinned routes, immutable parameters). For promotion event j:

I_promo(i,j) = P_i(after(j)) - P_i(before(j))

measured on routes changed by the promotion, unprotected tasks, cross-wing
compositions, safety-interrupt behavior, calibration and coverage metrics,
latency and abstention. The matrix is meaningful precisely where behavior
is allowed to change.

## Migration cost

MigrationCost = w1 * N_retrain + w2 * N_revalidate + w3 * N_recalibrate +
w4 * T_engineering

This is a limitation, not a solved problem: a Core migration is a migration
event, and one deployment epoch = one Core major version (version diamonds
on a multi-billion-parameter Core are hard-unsatisfiable, not merely
expensive).