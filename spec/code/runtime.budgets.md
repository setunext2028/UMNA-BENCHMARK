# runtime/budgets.py

Developer: Koushik Das - Version 0.1.0

Hard budget types. The smallest module in the codebase.

## Types

### ComputeBudget
- **Contract:** active compute ceiling, in FLOPs. C(S) <= B_C.
- **How it works:** frozen int. The planner never softens it; a graph that
  cannot meet it is infeasible, full stop.

### LatencyBudget
- **Contract:** end-to-end latency ceiling, in milliseconds. L(S) <= B_L.
- **How it works:** frozen float, end-to-end: the budget applies to the
  summed latency decomposition, including T_plan and T_admit. A graph
  whose wings fit but whose orchestration does not is infeasible.

### MemoryBudget
- **Contract:** peak memory ceiling, in megabytes. M(S) <= B_M.
- **How it works:** frozen float. Peak over the execution window, not the
  sum over wings — parallel branches share the accelerator.

### Budgets
- **Contract:** the triple handed to the planner.
- **How it works:** frozen composite of the three ceilings. This is the
  only budget type that crosses the planner seam.

## Design notes

- Budgets are constraints, not penalties (rule 5). The default planner
  carries lambda = 0 and these three values as its only cost terms.
- Frozen dataclasses: a budget that changes mid-plan is a bug, and the
  type prevents it.
- No coefficients live here. Any scalarised arm's coefficients belong to
  that arm and are measured constants, never shared state.