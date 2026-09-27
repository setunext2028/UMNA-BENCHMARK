# Seams and API

Developer: Koushik Das · Version 0.1.0

Every seam is listed here. If the thing you are about to write does not
belong to one of these, stop and raise it — do not invent a seam mid-task.

Shapes are shown in Python for the MVP. The same shapes hold in Rust later.

---

## 1 · Wing manifest (`wings/<name>/manifest.yaml`)

```yaml
wing:
  name: str
  version: str          # semver, exact
  core_epoch: str       # e.g. "1.x"
types:
  inputs:  [TypeExpr]
  outputs: [TypeExpr]
capabilities: [str]
dependencies:
  required: [{name: version_range}]
resource_ceilings:      # ceilings the host ENFORCES, not measurements
  max_memory_mb: int
  max_input_size: int
integrity:
  artifact_hash: str
```

**Validator must reject** any of: `reliability`, `accuracy`,
`expected_latency_ms`, `expected_evidence_gain`, `routing_priority`,
`quality`, `confidence`, or anything else the host measures. Reject on an
allow-list, not a deny-list — a deny-list is a game the vendor wins.

## 2 · Wing ABI

```python
class Wing(Protocol):
    def manifest(self) -> Manifest: ...
    def input_types(self) -> TypeSet: ...
    def output_types(self) -> TypeSet: ...
    def execute(self, ctx: CoreContext, x: LatentPacket) -> WingOutput: ...
```

`WingOutput` carries: `latent`, `schema`, `evidence: list[EvidencePacket]`,
`local_score`, `wing_version`, `provenance`. It carries **no** claim about its
own reliability.

`EvidencePacket` carries `polarity: SUPPORT | OPPOSE | UNKNOWN`. Conflict is
represented, never averaged away.

## 3 · Registry

```python
def typed_retrieval(x, registry, m: int) -> list[WingRef]
def resolve(refs, core_epoch) -> list[WingRef]     # raises on diamond
def measured(ref) -> HostMetrics                   # host-owned, never manifest
```

`HostMetrics` is the only source of latency, memory, reliability and utility
features. It is written by the measurement harness and read by the planner.

## 4 · Graph

```python
def build(wings, wiring) -> Graph      # raises TypeError on invalid composition
def cost(graph) -> Cost                # compute, latency, memory estimates
```

Type validity is checked at build time. An invalid graph must never reach
execution.

## 5 · Planner

```python
def plan(x, candidates, budgets: Budgets, mode: Mode) -> Graph
```

`Budgets` are **hard**. Default `lambda = 0`: the planner maximises predicted
utility subject to feasibility, with no free trade-off coefficients. If a
scalarised variant is used, every cost term converts through *measured*
constants, and that arm is reported separately.

## 6 · Admission — sits between plan and execute

```python
def admit(graph, binding) -> AdmissionRecord

AdmissionRecord:
  disposition: Production | ShadowOnly
  graph_id, binding_id, calibration_class
```

`ShadowOnly` means: execute if you want the data, log it, never serve it.
Falling back to the nearest admissible graph is allowed; serving an
inadmissible one is not.

## 7 · Execution

```python
def run(graph, x, context: ExecutionContext) -> tuple[Output, Trace]

ExecutionContext:
  registry, binding, admission_record, pipeline_started_ns
  upstream_latency: {retrieve, plan, admit}
```

Independent branches run in parallel. The trace is emitted whether or not the
run succeeds. The explicit context prevents execution from inventing binding,
admission or upstream timing values and lets tests use isolated registries.

```python
def replay(trace, registry) -> Graph
```

Replay rebuilds the exact graph from trace Wing versions and wiring, then
checks the rebuilt graph id against the recorded id.

## 8 · Risk

```python
def assess(output, binding) -> Decision   # Accept | Abstain | Expand
```

`Expand` returns to **admission**, not to execution — a grown graph is a
different binding and must be re-admitted.

## 9 · Trace (the measurement instrument)

```python
Trace:
  input_hash, core_epoch, graph_id, binding_id, calibration_class
  wing_versions: dict
  wiring
  admission: Production | ShadowOnly
  latency: {retrieve, plan, admit, load, transfer, wing, fusion, risk}
  wall_time_ms
  active_flops, peak_memory
  evidence: list[EvidencePacket]
  conflicts: list[ConflictSet]
  decision: Accept | Abstain | Expand
  replay_level: L1 | L2 | L3 | L4 | None
  failures: list
  shadow: bool
```

Every latency term is recorded separately. A trace missing a term is a bug,
not a gap — the decomposition is the whole point.

`None` means replay has not established any level. S2 reconstructs the route
before returning a successful trace and records L3 only after graph identity
matches; it never infers L1, L2 or L4 from route equality.
