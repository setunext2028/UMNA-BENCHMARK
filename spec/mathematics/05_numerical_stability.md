# Numerical stability of routing decisions

Developer: Koushik Das - Version 0.1.0

## The problem

Adaptive graph systems amplify small numerical differences because
thresholds control discrete branches. A perturbation of 1e-7 near a
threshold can select an entirely different Wing and produce a
discontinuously different answer. Numerical closeness therefore does NOT
imply route equivalence - the reverse of the usual ordering.

## The four tools

1. **Hysteresis.** Separate enter and exit thresholds:
   tau_enter > tau_exit. A wing is taken when the score crosses tau_enter
   and dropped only when it falls below tau_exit. Prevents flicker.

2. **Decision margins.** A route change requires
   U_hat(S_new, x) - U_hat(S_old, x) > delta_route.
   A strict improvement, not a tie-break.

3. **Quantized control values.** Routing scores are mapped to stable bins
   before branching. Two scores in the same bin are the same decision.

4. **Replay levels.** Every replay claim names its level:

   | Level | Guarantee | Condition |
   |---|---|---|
   | L1 bitwise | identical bits on identical hardware | y1 == y2 |
   | L2 numerically close | bounded float difference | ||y1 - y2|| <= eps |
   | L3 route-equivalent | same graph selected | S*1 == S*2 |
   | L4 decision-equivalent | same downstream decision | Head(y1) == Head(y2) |

   L2 does not imply L3. Reporting a replay claim without naming its level
   is uninformative.

## Where it is enforced

Every comparison on the control path uses hysteresis, a decision margin,
or quantized bins. A bare float compared against a threshold on the
control path is a bug (audit check B10 in docs/07_SELF_AUDIT.md).