# runtime/execute.py

Developer: Koushik Das - Version 0.1.0

Execution and trace emission. The trace is the measurement instrument.

Imports abi, admission, graph. Imported by bench and by the serving path.
Never imported by planner or wings.

## Types

### LatencyBreakdown
- **Contract:** all eight terms of the latency decomposition, in
  milliseconds:
  T_total = retrieve + plan + admit + load + transfer + wing + fusion +
  risk.
- **How it works:** one frozen float per term. The S2 gate: timings must
  sum to wall clock within tolerance. A trace missing a term is a bug,
  not a gap.
- **Failure mode:** a new kind of work silently absorbed into wing_ms
  destroys the experiment; audit check B7 exists for exactly this.

### Trace
- **Contract:** the full execution record for one input.
- **How it works:** input hash, epoch, graph and binding ids, calibration
  class, wing versions, admission result, latency breakdown, active FLOPs,
  peak memory, evidence and conflicts, decision, replay level, failures,
  shadow flag. Emitted whether or not the run succeeds.
- **Failure mode:** a swallowed exception that produces a trace without a
  failure entry is worse than a crash — it yields a number you later
  trust.

## Functions (S2)

### run(graph, x)
- **Contract:** execute a graph over one input, returning the Output and
  the Trace.
- **How it works (S2):**
  1. The caller has already planned and admitted — the executor does not
     re-admit, but the trace records the admission result it was given.
  2. Schedule: independent branches (no wiring dependency between them)
     in parallel; sequential chains in order.
  3. Execute wings, recording load, transfer and wing time per branch.
  4. Fuse via the task head; record fusion time.
  5. hand the result to the binding's risk test; record risk time.
  6. Emit the trace; on failure, record the failure and still emit.
- **Failure modes:** a wing returning NaN or a wrong schema is isolated
  and the graph continues without it if evidence allows, otherwise the
  run abstains; no exception is ever caught to keep a run alive without
  recording it.
- **Boundary:** it is a hard invariant that no path reaches run() without
  passing admit(). The executor may assert the admission result it is
  given; it never performs the admission itself.