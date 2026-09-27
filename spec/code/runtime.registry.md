# runtime/registry.py

Developer: Koushik Das - Version 0.1.0

Registry seams: strict manifest loading, validation, installation, epoch
pinning, version resolution, ceiling checks, host measurements, and typed
retrieval. This is the only place a manifest becomes an installed Wing.

Imports `runtime.abi`. Planner code cannot import `Manifest`; it sees
`WingRef` and host-owned `HostMetrics` only.

## Errors

### ManifestError
- **Raised by:** parsing, semantic validation, installation, pinning,
  ceiling checks, artifact lookup, and metric lookup.
- **Meaning:** a manifest or registry operation violates the declared
  control-plane contract.
- **Safety direction:** reject the Wing or operation. Never repair an
  unknown key, epoch, or value silently.

### VersionError
- **Raised by:** dependency resolution.
- **Meaning:** a requested Wing is absent, belongs to another Core epoch,
  has an unsatisfied dependency, or creates a version diamond.
- **Safety direction:** resolution stops. The graph is not built.

## Constants

### HOST_OWNED_FIELDS
Names that a Wing may not declare: reliability, accuracy, latency,
evidence gain, routing priority, quality, and confidence. The strict
allow-list rejects ANY unknown key; this set exists only to make errors for
the known host-owned names explicit.

### ALLOWED_MANIFEST_KEYS
The union of every valid key across the schema. Parsing also uses
section-specific key sets, so a valid nested key placed at the top level is
still rejected.

### DEFAULT_REGISTRY
The process-wide in-memory registry used by the module-level seam functions.
Tests, benchmarks, and serving code may instantiate `InMemoryRegistry`
directly when isolated state is required.

## Version helpers

### _to_semver(value)
- Converts one-, two-, or three-part numeric versions to a comparable
  `(major, minor, patch)` tuple. Missing parts become zero, so the range
  bound `<2` means `<2.0.0`.
- Manifest Wing versions themselves remain exact three-part semver.

### version_in_range(version, range_spec)
- Accepts exact versions, `*`, or comma-separated inequalities such as
  `>=1.0.0,<2` and `>=2,<3`.
- Every clause must hold.
- Malformed ranges and malformed Wing versions raise `ManifestError`.
- Used by both registry dependency resolution and graph dependency checks.

## Manifest parsing

### parse_type_expr(text)
- Parses `Latent<Text>` into `TypeExpr("Latent", ("Text",))`.
- Bare families such as `Latent` are accepted.
- Nested type variables and subtyping are not part of S1.

### _required_string(section, key, path)
- Reads a required non-empty string and produces a field-qualified error.
- Used for Wing identity and integrity hash fields.

### _parse_type_list(section, key)
- Requires a YAML list of strings, parses each with `parse_type_expr`, and
  returns an ordered `TypeSet`.

### _parse_dependencies(section)
- Requires `dependencies` to be a mapping and `required` to be a list.
- Every entry has exactly `name` and `version_range`; extra keys are
  rejected recursively, including host-owned names.
- Returns immutable `Dependency` values.

### _validate_key_set(unknown)
- Rejects an unknown key list.
- Known host-owned names receive the specific "measured by the host"
  error; every other unknown name receives the allow-list error.

### parse_manifest(text)
- Uses `yaml.safe_load`, then validates keys at every schema level:
  top-level, Wing identity, types, dependency container, each dependency,
  ceilings, integrity, and provenance.
- Rejects wrong YAML shapes instead of coercing them.
- Builds the immutable `Manifest` value and calls `validate_manifest` for
  semantic checks.
- This function is the S1 gate for a manifest declaring `reliability:`.

### load_manifest(path)
- Reads UTF-8 YAML from disk and delegates to `parse_manifest`.
- Used by on-disk stub Wings and `InMemoryRegistry(seed_dir=...)`.

### validate_manifest(manifest)
- Checks already-typed manifests, including those built directly in Python.
- Requires non-empty identity fields, exact semver, at least one input and
  output type, string capability names, valid dependency ranges, positive
  integer ceilings, and a non-empty integrity hash when integrity metadata
  is present.
- Direct construction cannot bypass the rules enforced during YAML parsing.

## HostMetrics

The only source of routing-relevant numbers: measured latency, measured
memory, measured reliability, utility features, and failure history, keyed

## Registry protocol

The protocol fixes the implementation boundary:

- `load(manifest, artifact)` installs an exact version.
- `validate(manifest)` runs semantic checks.
- `resolve(refs, core_epoch)` closes dependencies inside one epoch.
- `pin(core_epoch)` freezes the deployment epoch.
- `measured(ref)` returns host-owned measurements.
- `retrieve(x, m)` performs bounded typed retrieval.
- `installed()` lists exact installed references.
- `artifact(ref)` returns the executable Wing object.
- `check_input(ref, size)` enforces the declared input ceiling.

## InMemoryRegistry

### __init__(core_epoch=None, seed_dir=None)
- Starts empty and optionally pre-pins an epoch.
- If `seed_dir` is supplied, loads every `<child>/manifest.yaml` in sorted
  order. This is the MVP's file-backed entry point.

### validate(manifest)
- Delegates to `validate_manifest`.

### load(manifest, artifact=None)
- Runs semantic validation.
- Rejects a manifest outside the pinned Core epoch.
- Installs exact `(name, version)` identity in deterministic order.
- Reloading the exact same manifest is idempotent and may attach a later
  artifact. Reusing the same identity with changed metadata is rejected.
- S1 validates integrity metadata but does not hash model bytes; real
  artifact hash verification begins with S3 artifacts.

### pin(core_epoch)
- A first pin freezes the deployment epoch.
- Repeating the same pin is harmless; repinning to another epoch fails.
- Pinning also fails when an already-installed Wing belongs to another
  epoch.

### installed()
- Returns installed `WingRef` values in deterministic installation order.

### artifact(ref)
- Returns the installed executable object.
- A manifest-only entry raises `ManifestError`; S2 execution cannot run a
  Wing whose bytes were never installed.

### check_input(ref, size)
- Rejects negative sizes.
- Returns `False` when `size` exceeds `max_input_size`; otherwise `True`.
- This is the S1 enforcement point consumed by S2 execution.

### set_measured(metrics)
- Writes host measurements only for an installed exact `WingRef`.
- Rejects negative latency or memory and reliability outside `[0, 1]`.
- Rejects measured memory above the manifest's declared memory ceiling;
  this is the S1 memory-ceiling enforcement point.
- The benchmark harness is the future producer of these records.

### measured(ref)
- Returns host measurements.
- An unmeasured Wing raises: unmeasured means ineligible for routing, not
  zero cost or perfect reliability.

### _choose(name, range_spec, core_epoch)
- Filters installed versions by BOTH dependency range and Core epoch.
- Selects the highest fitting exact semver.
- Prevents a newer version from another epoch displacing a compatible
  version during resolution.

### resolve(refs, core_epoch=None)
- Uses the supplied epoch or the pinned epoch; absence of both is an error.
- Walks the dependency closure breadth-first.
- Rejects missing Wings, epoch mismatches, unsatisfied ranges, and two
  versions of the same Wing in one closure (a version diamond).
- Returns requested references first, followed by dependencies in stable
  discovery order.

### retrieve(x, m)
- S1 retrieval is intentionally small: exact type unification in install
  order, capped at `m`, and filtered to the pinned epoch.
- `m <= 0` returns no candidates.
- Capability embeddings and host route statistics arrive in S7; adding
  them early would mix registry correctness with retrieval research.

## Module-level seams

### typed_retrieval(x, registry, m)
Delegates to the supplied registry's `retrieve` method.

### resolve(refs, core_epoch=None)
Delegates to `DEFAULT_REGISTRY.resolve`.

### measured(ref)
Delegates to `DEFAULT_REGISTRY.measured`.
