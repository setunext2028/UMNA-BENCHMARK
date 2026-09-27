# runtime/abi.py

Developer: Koushik Das - Version 0.1.0

The typed capability ABI. The lowest module in the dependency graph:
everything runtime, planner and risk imports from here; abi.py imports
stdlib only.

## Types

### TypeExpr
- **Contract:** one declared neural type: a family plus arguments, e.g.
  TypeExpr("Image", ("XRay", "Chest")).
- **How it works:** immutable value type. The graph builder unifies a
  Wing's output TypeExpr against the next Wing's input TypeExpr.
- **Failure mode:** two families that never unify - a type-invalid graph
  must raise TypeError at build time, never at execution.

### type_unifies(given, required)
- **Contract:** tells graph construction whether one declared output may
  occupy a declared input port.
- **S1 rule:** exact family AND exact argument tuple. No implicit
  conversion, wildcard, covariance, or subtyping.
- **Why strict now:** S1 tests control-plane correctness with constants;
  real representation semantics do not exist until S3.

### TypeSet
- **Contract:** the ordered input or output contract of a Wing.
- **How it works:** tuple of TypeExpr with iteration support. Order
  matters: output index i of one wing feeds input index i of the next via
  the wiring. `__iter__`, `__len__`, and `__getitem__` give graph
  construction the minimum sequence operations it needs.

### LatentSchema
- **Contract:** the latent/type contract valid for one Core epoch:
  epoch tag, dimensionality, numeric format.
- **How it works:** frozen; bound to exactly one core_epoch string. The
  registry refuses a Wing declaring a different epoch.

### LatentPacket
- **Contract:** the typed value exchanged between Wings and the Core.
- **How it works:** a LatentSchema plus the data payload plus provenance.
  The payload is opaque at ABI level (typed as object); S3 pins it to the
  concrete tensor type.
- **Failure mode:** a Wing returning a packet whose schema differs from its
  declared output schema is isolated and the failure recorded in the
  trace.

### CoreContext
- **Contract:** fixed context handed to a Wing at execute time.
- **How it works:** carries the epoch and latent schema the Wing must
  declare compatibility with, plus the binding id when known.
- **Boundary:** created by the executor, consumed by wings. It is the only
  channel through which a Wing learns anything about the host.

### Polarity
- **Contract:** SUPPORT | OPPOSE | UNKNOWN. The three-way polarity of one
  claim.
- **How it works:** enum; the decision layer never averages away an OPPOSE
  against a SUPPORT.

### EvidencePacket
- **Contract:** one scored claim from one Wing version on one route.
- **How it works:** carries claim identity (type and value), polarity,
  local score, provenance, the source wing and version, the route and
  calibration binding, validity domain and timestamp. Nothing here is a
  reliability claim; local_score is a local signal the host measures, not
  a self-reported routing metric.
- **Failure mode:** a packet whose source wing does not match the executed
  graph is a provenance violation - flagged, never silently accepted.

### ConflictSet
- **Contract:** contradictory claims kept apart.
- **How it works:** groups opposing evidence for one claim type. Passed to
  the decision layer; above the action threshold it terminates in
  abstain-and-escalate unless a calibrated policy overrides.

### WingOutput
- **Contract:** what a Wing may return.
- **How it works:** latent packet, output schema, evidence list, local
  score, wing version, provenance. The type carries no claim about its own
  reliability - there is no field for it, and there never will be.
- **Failure mode:** shape or schema violations are isolated and recorded;
  the graph continues without the wing if evidence allows, otherwise
  abstains.

### Wing (Protocol)
- **Contract:** the capability module contract: manifest(), input_types(),
  output_types(), execute(ctx, x) -> WingOutput.
- **How it works:** structural typing. Any object implementing these four
  methods is a Wing; the registry accepts it as an installed capability.
- **Boundary:** a Wing never imports another Wing. Composition happens in
  the graph.

### Dependency, ResourceCeilings, ManifestIntegrity, ManifestProvenance, Manifest
- **Contract:** the manifest schema - the complete allow-list of what a
  Wing may declare.
- **How it works:** the validator in runtime.registry parses YAML strictly
  into these types; any key not representable is rejected with a clear
  error. There is deliberately no field for reliability, accuracy,
  latency, evidence gain, relevance or confidence - the allow-list is the
  enforcement mechanism.
- **Why ceiling, not measurement:** a ceiling (max_memory_mb, max_input)
  is a promise the host can enforce and falsify cheaply; a measurement is
  a claim the host must not take from a party competing for selection.

### Decision
- **Contract:** ACCEPT | ABSTAIN | EXPAND.
- **How it works:** the terminal outcome of risk control. EXPAND is not a
  decision to execute more - it is a request to re-enter ADMISSION with a
  changed graph.

### ReplayLevel
- **Contract:** L1 bitwise | L2 numerically close | L3 route-equivalent |
  L4 decision-equivalent.
- **How it works:** every replay claim records its level as an enum; L2
  does not imply L3.

### Output
- **Contract:** the final served (or abstained) result of one execution.
- **How it works:** the value, the decision, the evidence and conflicts
  that reached the decision. An abstain Output carries its conflicts, so
  the abstention is auditable.

### WingRef
- **Contract:** an exact, pinned reference to an installed Wing version.
- **How it works:** name@version identity. The registry resolves
  "name" references to WingRefs against the pinned Core epoch; the graph
  binds WingRefs only - unversioned wings do not exist in a graph.
- `__str__` renders the stable `name@version` form used in errors and
  resolution traces.
