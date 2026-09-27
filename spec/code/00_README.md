# spec/code: how to read these files

Developer: Koushik Das - Version 0.1.0

One document per code file, mirroring the repo layout. Each document
covers, for every public name in the file:

- **Contract** - what the type or function promises, taken from
  docs/05_SEAMS_AND_API.md.
- **How it works** - the algorithm the implementation must follow when the
  stage lands.
- **Failure modes** - what can go wrong and what the safe default is.
- **Boundaries** - what may import it and what it may import.

S1 is gated. Registry/ABI/graph behavior and constant stub Wings are live;
S2 and later files remain typed stubs. These documents track that boundary.

| Code file | Spec document |
|---|---|
| runtime/abi.py | code/runtime.abi.md |
| runtime/registry.py | code/runtime.registry.md |
| runtime/budgets.py | code/runtime.budgets.md |
| runtime/graph.py | code/runtime.graph.md |
| runtime/admission.py | code/runtime.admission.md |
| runtime/execute.py | code/runtime.execute.md |
| planner/estimator.py | code/planner.estimator.md |
| planner/arms/*.py | code/planner.arms.md |
| planner/interaction.py | code/planner.interaction.md |
| risk/classes.py + risk/conformal.py | code/risk.md |
| wings/*/model.py | code/wings.stubs.md |
| tools/*.py | code/tools.md |
| tests/*.py | code/tests.md |
| package `__init__.py` files | code/packages.md |
| bench/*.py + core/ | code/bench_core_tools.md |
