# Cost identification

Developer: Koushik Das - Version 0.1.0

## Why one coefficient and not six

An early formulation used six independent penalty coefficients
(lambda_C, lambda_L, lambda_M, lambda_R, lambda_A, lambda_K) plus three
budgets. A planner with six tuned coefficients compared against a top-k
baseline with one threshold wins partly from fitted degrees of freedom, and
the comparison does not isolate the mechanism under test. This almost
invalidated the whole benchmark.

## Strategy (a) - common-currency scalarization

Every cost is converted to one unit: expected utility forgone per unit of
the binding resource. Compute, memory and composition overhead become
latency at measured device throughput; latency becomes utility through a
deployment-specific service curve rho(...) elicited once per task family
and held fixed across all arms:

Gamma(S) = rho( L_eff(S) )
L_eff(S) = L(S) + kappa_C * C(S) + kappa_M * M(S) + kappa_K * K(S)

where kappa_x are MEASURED conversion constants, never tuned
hyperparameters.

## Strategy (b) - hard constraints only

lambda = 0. All cost terms are budgets. The planner maximises utility
subject to feasibility, with zero free coefficients.

Strategy (b) is the recommended default for the primary benchmark because
it removes the degrees-of-freedom objection. Strategy (a) is reported as a
secondary arm.

## Protocol 9.2 - equal tuning budget

Whichever strategy is used:

- every baseline receives an identical hyperparameter search budget;
- the search runs on the same held-out split;
- the budget is reported explicitly in the output;
- a result obtained by tuning NICE-R more than its baselines is not a
  NICE-R result.

## Where the constants are measured

| Constant | Measured by |
|---|---|
| kappa_C, kappa_M, kappa_K | device-throughput measurement on the target hardware profile |
| rho (service curve) | elicited once per task family, held fixed across arms |
| latency L(S) | decomposed trace, mean over the calibration split |
| compute C(S) | active-FLOP accounting in the trace |
| memory M(S) | peak-memory accounting in the trace |