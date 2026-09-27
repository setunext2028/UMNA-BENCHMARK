# planner/estimator.py

Developer: Koushik Das - Version 0.1.0

Host-measured utility estimator and the plan seam — the flagship module of
the project.

Imports abi, budgets, graph. Imported by bench, arms, and the serving
path. Never imported by runtime.

## Types

### Mode
- **Contract:** the three planner modes: GREEDY (A), BOUNDED_PAIR (B),
  BEAM (C).
- **How it works:** enum; the harness runs every arm under the same
  tuning budget, and each mode is reported separately. BOUNDED_PAIR and
  BEAM exist because greedy provably misses supermodular pairs.

### PredictedUtility
- **Contract:** host-predicted task utility of a graph for one input.
- **How it works:** the value plus the feature vector it came from, so
  the estimator can be audited as an estimator in its own right
  (Experiment A measures its calibration, not just its search results).

## Functions (S5)

### estimate_utility(graph, x, refs)
- **Contract:** host-side utility estimate of a candidate graph.
- **How it works (S5):** features from HostMetrics (measured latency,
  memory, reliability, utility features) plus input observables; a
  regression heads these into a scalar utility prediction. Trained on
  controlled validation, historical routes and shadow trials on the same
  held-out protocol as every arm.
- **Failure mode:** bad calibration of the estimator silently degrades
  search quality while looking fine in search-level metrics. The
  estimator must be evaluated as a predictor, not only as a search input.
- **Boundary:** no vendor-declared number may enter. The features are
  host-measured by construction — they come from HostMetrics, which the
  manifest cannot produce.

### plan(x, candidates, budgets, mode=GREEDY)
- **Contract:** choose the graph maximizing predicted utility subject to
  hard budgets. lambda = 0 by default: costs are constraints, not
  penalties.
- **How it works (S5):** search over the graph space over the candidates
  with the active mode's strategy; check every candidate against the
  three budgets; prune infeasible; return the best feasible graph. The
  returned graph has NOT been admitted — admission is the separate step
  between this function and execution.
- **Failure modes:** empty candidate set or no feasible graph -> the
  planner returns the empty graph (abstain territory), never a guessed
  graph; budget exhausted mid-plan terminates in abstain.
- **Boundary:** plan never executes wings and never reads a manifest. The
  only number sources are HostMetrics and the input itself.