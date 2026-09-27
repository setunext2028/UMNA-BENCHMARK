# runtime/graph.py

Developer: Koushik Das - Version 0.1.0

Typed graph construction and validation. Imports ABI values plus the
registry's semantic validator and version-range function. Admission,
execution, planner, and risk consume the resulting immutable graph.

## Types

### Wiring
- Directed edges represented as `(source_name, source_output_index,
  target_name, target_input_index)`.
- Indices are zero-based and must be non-negative.
- `build` canonicalises edge order before assigning graph identity.

### Cost
- Estimated active FLOPs, latency, and memory.
- The type exists at S1; `cost()` remains deferred to S2 because accurate
  cost requires trace-compatible host measurements.

### Graph
- Exact graph id, sorted `WingRef` tuple, canonical wiring, and Core epoch.
- Input list order does not change graph identity; the Wing set and wiring
  do.

### Binding
- `(Core epoch, graph, head, distribution, hardware profile)`.
- Calibration attaches here, never to a Wing.
- `binding_id` includes all five dimensions. Changing distribution or
  hardware creates a different binding before S4 attaches a calibrator.

### BuildError
- Subclass of `TypeError`, matching the seam in `docs/05`.
- Covers invalid manifests, duplicate Wing names, mixed epochs, unknown
  wiring nodes, bad port indices, type mismatches, and missing dependencies.

### CostError
- Reserved for hard-budget failures when graph costing lands in S2.

## Functions

### _graph_id(refs, wiring)
- Serialises sorted exact Wing references and canonical edges.
- Hashes the serialisation with SHA-256 and keeps a 16-character id.
- Same graph, same id; changed wiring or version, changed id.

### build(wings, wiring)
1. Rejects an empty graph.
2. Reads each Wing manifest once and runs `validate_manifest`.
3. Rejects two versions or instances with the same Wing name.
4. Requires every Wing to share one Core epoch.
5. Checks that every edge names existing Wings and valid non-negative
   port indices.
6. Uses strict `type_unifies` on every source output / target input pair.
7. Checks every declared dependency against the other Wings and its
   version range.
8. Sorts refs and edges, derives graph id, and returns the immutable Graph.

Failure occurs at construction. No invalid graph reaches execution.

### cost(graph)
- Raises `NotImplementedError` at S1.
- S2 will calculate the trace-compatible cost estimate; pretending to
  know it before the measurement instrument exists would hide work.
