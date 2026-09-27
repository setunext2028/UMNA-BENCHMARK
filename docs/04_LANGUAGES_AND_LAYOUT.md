# Languages and layout

Developer: Koushik Das · Version 0.1.0

## The boundary

**Python** owns the research plane: model definitions, training, the utility
estimator, calibration and conformal machinery, statistics, ablations, plots.

**Rust** owns the control plane: registry, manifest validation, type checking,
version resolution, budget accounting, scheduling, trace and audit records,
replay.

**Existing backends** own compilation and kernels. We do not write a tensor
library or a compiler. If lowering is needed later, it goes through MLIR /
IREE / ONNX Runtime / TVM — never a hand-rolled replacement.

## Which language for the MVP

**Python for the whole MVP.** Rust starts at S6 or later, and only for the
control plane, ported behind the seams already defined in S0.

This is deliberate. The MVP question is about planner quality, not runtime
engineering. Writing the control plane in Rust before S5 has an answer buys
correctness for code that may be deleted. The seams from S0 are what make the
port cheap when it happens — design for the port, do not perform it early.

Do not treat this as permission to write throwaway Python. The type
boundaries from S0 hold in either language.

## Layout

```
UMNA/
  CLAUDE.md
  docs/                 the files you are reading
  memory/               decisions, invariants, traps, results
  core/                 the frozen Core: definition, training, artifact + hash
  wings/
    <name>/
      manifest.yaml     declares types, deps, ceilings — never metrics
      model.py          the capability itself
      test_vectors/     fixed inputs and expected outputs
  runtime/
    registry.py         load, validate, resolve, pin
    abi.py              Wing protocol, LatentPacket, WingOutput
    graph.py            typed graph construction and validation
    admission.py        calibration-class check. Sits between plan and execute
    execute.py          scheduling, parallel branches, trace emission
    budgets.py          hard constraint types
  planner/
    estimator.py        host-measured utility
    arms/               topk.py · gates.py · greedy.py · pairs.py · beam.py · oracle.py
    interaction.py      pair proposal model (S6)
  risk/
    conformal.py
    classes.py          the calibration-class registry
  bench/
    harness.py          runs all arms under equal tuning budget
    tasks/
    report.py           emits the one table
  tools/
    audit.py            the self-audit checks from docs/07
    lint_human.py       the style checks from docs/09
```

## Rules

- `runtime/` never imports from `planner/` or `wings/`. Dependencies point
  inward toward the contracts, never outward toward strategies.
- A Wing never imports another Wing. Composition happens in the graph.
- `bench/` may import anything. Nothing imports `bench/`.
- Anything that touches a number affecting routing lives in `planner/` or
  `runtime/`, never in `wings/`.
