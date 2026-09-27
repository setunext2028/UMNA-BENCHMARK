# tests/

Developer: Koushik Das - Version 0.1.0

The S1 suite uses standard-library `unittest`; pytest is not required.
There are 70 tests. The tests below are construction checks and stage-gate
evidence, not experimental findings.

## test_manifest_validation.py

### TestManifestParsing
- `test_valid_manifest_parses`: full YAML becomes the expected typed value.
- `test_reliability_declared_is_rejected`: the primary S1 gate with a clear
  host-owned-field error.
- `test_anything_host_owned_is_rejected`: every named host-owned field fails.
- `test_unknown_key_is_rejected_by_allow_list`: unknown top-level key fails.
- `test_nested_host_owned_key_is_rejected`: host metric hidden in a
  dependency entry still fails.
- `test_nested_unknown_key_is_rejected`: arbitrary nested metadata fails.
- `test_nested_key_at_top_level_is_rejected`: section placement is strict.
- `test_deny_list_cannot_hide_a_new_name`: a newly invented metric name is
  rejected even though it is not in `HOST_OWNED_FIELDS`.
- `test_bad_ceiling_rejected`: non-positive resource ceiling fails.
- `test_non_string_capability_rejected`: capability tags remain strings.
- `test_provenance_ids_must_be_a_list`: strings cannot masquerade as lists.
- `test_non_semver_rejected`: Wing versions are exact semver.
- `test_malformed_range_rejected`: dependency range grammar is enforced.
- `test_manifest_must_be_mapping`: YAML sequences are not manifests.

### TestTypeExpr
- `test_parses_family_with_args`, `test_parses_bare_family`: accepted forms.
- `test_rejects_garbage`: malformed neural type is rejected.

### TestVersionRanges
- `test_wildcard`, `test_exact`, `test_inequalities`: supported semantics.
- `test_malformed_raises`: malformed range never becomes a false match.

## test_registry.py

### manifest(...)
Builds a small typed manifest for registry failure tests.

### TestRegistryLoad
- Disk load, artifact attachment, seed-directory loading.
- Pinned-epoch mismatch, repinning, and pinning with incompatible installed
  Wings all fail.
- Reusing one exact WingRef with changed metadata fails.

### TestResolve
- Dependency closure for beta and gamma.
- Resolve-time epoch mismatch, no pinned epoch, unsatisfied dependency, and
  explicit version diamond all fail.
- Dependency selection remains inside the requested Core epoch even when a
  numerically newer version exists in another epoch.

### TestCeilingsAndMetrics
- Below-ceiling input passes; above-ceiling input is blocked; negative size
  fails.
- An unmeasured Wing is ineligible.
- Host metrics round-trip only for installed exact refs.
- Measured memory above the declared manifest ceiling is rejected.

### TestRetrieval
- Exact input type selects alpha/beta, not gamma.
- Candidate count obeys `m`; zero returns none; incompatible modality
  returns none.

## test_graph.py

### TestTypeUnification
Checks exact equality and both mismatch dimensions (family and arguments).

### TestGraphBuild
- Builds the valid alpha/beta/gamma graph.
- Exercises failures for type mismatch, missing dependency, unknown node,
  positive and negative bad indices, empty graph, duplicate Wing name, and
  mixed Core epochs.
- Checks deterministic graph identity, input-list-order independence, and
  identity change when wiring changes.
- Checks that distribution and hardware changes produce distinct binding
  ids.

## test_stub_wings.py

Executes every stub and checks constant output, exact version, manifest/type
agreement, and seed-directory discovery of all three Wings.

## test_gates.py

- `test_lint_human_exits_zero`: runs the style gate as a subprocess.
- `test_audit_exits_zero`: runs the audit gate. When audit itself invokes
  the suite, an environment marker skips this one recursive call.
