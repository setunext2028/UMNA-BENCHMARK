"""Registry seams: load, validate, resolve, pin.

The registry is the only place a manifest becomes an installed, measurable
Wing. HostMetrics is the single source of routing-relevant numbers; nothing
routing-relevant is ever read from a manifest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

import yaml

from .abi import (
    Dependency,
    Manifest,
    ManifestIntegrity,
    ManifestProvenance,
    ResourceCeilings,
    TypeExpr,
    TypeSet,
    Wing,
    WingRef,
    type_unifies,
)


class ManifestError(ValueError):
    """A manifest failed validation: unknown keys, host-owned fields,
    malformed values, or an epoch conflict at load time."""


class VersionError(ValueError):
    """Version resolution failed: unsatisfiable ranges or a version diamond
    inside one Core epoch."""


HOST_OWNED_FIELDS: frozenset[str] = frozenset(
    {
        "reliability",
        "accuracy",
        "expected_latency_ms",
        "expected_evidence_gain",
        "routing_priority",
        "quality",
        "confidence",
    }
)

ALLOWED_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "wing",
        "name",
        "version",
        "core_epoch",
        "author",
        "types",
        "inputs",
        "outputs",
        "capabilities",
        "dependencies",
        "required",
        "version_range",
        "resource_ceilings",
        "max_memory_mb",
        "max_input_size",
        "integrity",
        "artifact_hash",
        "provenance",
        "training_dataset_ids",
        "training_code_commit",
    }
)

_TOP_LEVEL_KEYS = {
    "wing",
    "types",
    "capabilities",
    "dependencies",
    "resource_ceilings",
    "integrity",
    "provenance",
}
_WING_KEYS = {"name", "version", "core_epoch", "author"}
_TYPE_KEYS = {"inputs", "outputs"}
_DEPENDENCY_KEYS = {"required"}
_DEPENDENCY_ENTRY_KEYS = {"name", "version_range"}
_CEILING_KEYS = {"max_memory_mb", "max_input_size"}
_INTEGRITY_KEYS = {"artifact_hash"}
_PROVENANCE_KEYS = {"training_dataset_ids", "training_code_commit"}

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_RANGE_OP_RE = re.compile(r"^([<>]=?)?(\d+(?:\.\d+){0,2})$")


def _to_semver(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    parts.extend("0" for _ in range(3 - len(parts)))
    return int(parts[0]), int(parts[1]), int(parts[2])


def version_in_range(version: str, range_spec: str) -> bool:
    """Does `version` satisfy a range like ">=2,<3" or ">=1.2.0,<2.0.0"?

    A bare version means exact. "*" means any version. Ranges are comma
    separated comparisons of the form (op)?semver.
    """

    if not isinstance(version, str) or not isinstance(range_spec, str):
        raise ManifestError("version and version range must be strings")
    if range_spec == "*":
        return True
    if _SEMVER_RE.match(range_spec):
        return version == range_spec
    for clause in range_spec.split(","):
        clause = clause.strip()
        match = _RANGE_OP_RE.match(clause)
        if not match:
            raise ManifestError(f"malformed version range: {range_spec!r}")
        op, bound = match.group(1), _to_semver(match.group(2))
        actual = _to_semver(version) if _SEMVER_RE.match(version) else None
        if actual is None:
            raise ManifestError(f"not a semver: {version!r}")
        if op == ">" and not actual > bound:
            return False
        if op == ">=" and not actual >= bound:
            return False
        if op == "<" and not actual < bound:
            return False
        if op == "<=" and not actual <= bound:
            return False
        if op is None and actual != bound:
            return False
    return True


_TYPEDEF_RE = re.compile(r"^([A-Za-z_]+)(?:<([A-Za-z0-9_, ]+)>)?$")


def parse_type_expr(text: str) -> TypeExpr:
    """Parse "Latent<Text>" or "Latent" into a TypeExpr."""

    match = _TYPEDEF_RE.match(text.strip())
    if not match:
        raise ManifestError(f"malformed type expression: {text!r}")
    family, args = match.group(1), match.group(2)
    if args is None:
        return TypeExpr(family)
    return TypeExpr(family, tuple(arg.strip() for arg in args.split(",")))


def _required_string(section, key, path="wing"):
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(
            f"manifest field {path}.{key} must be a non-empty string"
        )
    return value


def _parse_type_list(section, key):
    values = section.get(key)
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ManifestError(f"manifest field types.{key} must be a list of strings")
    return TypeSet(tuple(parse_type_expr(v) for v in values))


def _parse_dependencies(section):
    deps = section.get("dependencies")
    if deps is None:
        return ()
    if not isinstance(deps, dict):
        raise ManifestError("manifest field dependencies must be a mapping")
    _validate_key_set([key for key in deps if key not in _DEPENDENCY_KEYS])
    if "required" not in deps:
        return ()
    required = deps["required"]
    if not isinstance(required, list):
        raise ManifestError("manifest field dependencies.required must be a list")
    parsed = []
    for entry in required:
        if not isinstance(entry, dict):
            raise ManifestError("each required dependency must be a mapping")
        _validate_key_set(
            [key for key in entry if key not in _DEPENDENCY_ENTRY_KEYS]
        )
        if "name" not in entry or "version_range" not in entry:
            raise ManifestError(
                "each required dependency needs name and version_range"
            )
        name = entry["name"]
        version_range = entry["version_range"]
        if not isinstance(name, str) or not name:
            raise ManifestError("dependency name must be a non-empty string")
        if not isinstance(version_range, str) or not version_range:
            raise ManifestError(
                "dependency version_range must be a non-empty string"
            )
        parsed.append(Dependency(name, version_range))
    return tuple(parsed)


def _validate_key_set(unknown: list[str]) -> None:
    if not unknown:
        return
    for key in unknown:
        if key in HOST_OWNED_FIELDS:
            raise ManifestError(
                f"host-owned field {key!r} is not allowed in a manifest: "
                "that quantity is measured by the host, never declared"
            )
    raise ManifestError(
        f"unknown manifest field(s): {', '.join(unknown)}; the allow-list "
        "does not recognise them"
    )


def parse_manifest(text: str) -> Manifest:
    """Strictly parse manifest YAML into an abi.Manifest.

    Enforces the allow-list: any key that abi.Manifest cannot represent is
    rejected, with host-owned fields named specifically. A deny-list would
    lose the moment a vendor picks a new field name; the allow-list cannot.
    """

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a YAML mapping")
    _validate_key_set([key for key in raw if key not in _TOP_LEVEL_KEYS])

    wing_section = raw.get("wing")
    if not isinstance(wing_section, dict):
        raise ManifestError("manifest needs a wing section")
    _validate_key_set([key for key in wing_section if key not in _WING_KEYS])

    types_section = raw.get("types") or {}
    if not isinstance(types_section, dict):
        raise ManifestError("manifest field types must be a mapping")
    _validate_key_set([key for key in types_section if key not in _TYPE_KEYS])

    raw_capabilities = raw.get("capabilities", [])
    if not isinstance(raw_capabilities, list) or not all(
        isinstance(capability, str) and capability
        for capability in raw_capabilities
    ):
        raise ManifestError("manifest field capabilities must be a list of strings")

    ceilings = None
    raw_ceilings = raw.get("resource_ceilings")
    if raw_ceilings is not None:
        if not isinstance(raw_ceilings, dict):
            raise ManifestError("manifest field resource_ceilings must be a mapping")
        _validate_key_set(
            [key for key in raw_ceilings if key not in _CEILING_KEYS]
        )
        memory = raw_ceilings.get("max_memory_mb")
        size = raw_ceilings.get("max_input_size")
        if not isinstance(memory, int) or memory <= 0:
            raise ManifestError("resource_ceilings.max_memory_mb must be a positive integer")
        if not isinstance(size, int) or size <= 0:
            raise ManifestError("resource_ceilings.max_input_size must be a positive integer")
        ceilings = ResourceCeilings(max_memory_mb=memory, max_input_size=size)

    integrity = None
    raw_integrity = raw.get("integrity")
    if raw_integrity is not None:
        if not isinstance(raw_integrity, dict):
            raise ManifestError("manifest field integrity must be a mapping")
        _validate_key_set(
            [key for key in raw_integrity if key not in _INTEGRITY_KEYS]
        )
        integrity = ManifestIntegrity(
            artifact_hash=_required_string(
                raw_integrity, "artifact_hash", path="integrity"
            )
        )

    provenance = None
    raw_provenance = raw.get("provenance")
    if raw_provenance is not None:
        if not isinstance(raw_provenance, dict):
            raise ManifestError("manifest field provenance must be a mapping")
        _validate_key_set(
            [key for key in raw_provenance if key not in _PROVENANCE_KEYS]
        )
        ids = raw_provenance.get("training_dataset_ids", ())
        if not isinstance(ids, (list, tuple)) or not all(
            isinstance(dataset_id, str) and dataset_id for dataset_id in ids
        ):
            raise ManifestError("provenance.training_dataset_ids must be a list of strings")
        training_code_commit = raw_provenance.get("training_code_commit", "")
        if not isinstance(training_code_commit, str):
            raise ManifestError(
                "provenance.training_code_commit must be a string"
            )
        provenance = ManifestProvenance(
            training_dataset_ids=tuple(ids),
            training_code_commit=training_code_commit,
        )

    manifest = Manifest(
        name=_required_string(wing_section, "name"),
        version=_required_string(wing_section, "version"),
        core_epoch=_required_string(wing_section, "core_epoch"),
        author=_required_string(wing_section, "author"),
        input_types=_parse_type_list(types_section, "inputs"),
        output_types=_parse_type_list(types_section, "outputs"),
        capabilities=tuple(raw_capabilities),
        dependencies=_parse_dependencies(raw),
        resource_ceilings=ceilings,
        integrity=integrity,
        provenance=provenance,
    )
    validate_manifest(manifest)
    return manifest


def load_manifest(path: str | Path) -> Manifest:
    """Read and strictly parse one manifest file."""

    return parse_manifest(Path(path).read_text(encoding="utf-8"))


def validate_manifest(manifest: Manifest) -> None:
    """Semantic validation of an already-typed manifest."""

    if not isinstance(manifest.name, str) or not manifest.name:
        raise ManifestError("wing name must be a non-empty string")
    if not isinstance(manifest.version, str) or not _SEMVER_RE.match(
        manifest.version
    ):
        raise ManifestError(f"wing version must be semver, got {manifest.version!r}")
    if not isinstance(manifest.core_epoch, str) or not manifest.core_epoch:
        raise ManifestError("wing core_epoch must not be empty")
    if not isinstance(manifest.author, str) or not manifest.author:
        raise ManifestError("wing author must be a non-empty string")
    if len(manifest.input_types) == 0 or len(manifest.output_types) == 0:
        raise ManifestError("a wing needs at least one input and one output type")
    if not all(
        isinstance(capability, str) and capability
        for capability in manifest.capabilities
    ):
        raise ManifestError("capabilities must be non-empty strings")
    for dep in manifest.dependencies:
        if not dep.name or not dep.version_range:
            raise ManifestError(
                "dependencies need non-empty name and version_range"
            )
        version_in_range(manifest.version, dep.version_range)
    ceilings = manifest.resource_ceilings
    if ceilings is not None:
        if type(ceilings.max_memory_mb) is not int or ceilings.max_memory_mb <= 0:
            raise ManifestError("max_memory_mb must be a positive integer")
        if type(ceilings.max_input_size) is not int or ceilings.max_input_size <= 0:
            raise ManifestError("max_input_size must be a positive integer")
    if manifest.integrity is not None and not manifest.integrity.artifact_hash:
        raise ManifestError("artifact_hash must be a non-empty string")
    provenance = manifest.provenance
    if provenance is not None:
        if not all(
            isinstance(dataset_id, str) and dataset_id
            for dataset_id in provenance.training_dataset_ids
        ):
            raise ManifestError("training_dataset_ids must be non-empty strings")
        if not isinstance(provenance.training_code_commit, str):
            raise ManifestError("training_code_commit must be a string")


@dataclass(frozen=True)
class HostMetrics:
    """Host-measured routing attributes. Written by the harness, read by the
    planner. Never declared by a Wing."""

    ref: WingRef
    measured_latency_ms: float
    measured_memory_mb: float
    measured_reliability: float
    measured_utility_features: tuple[float, ...]
    measured_active_flops: float = 0.0
    failure_history: tuple[str, ...] = ()


class Registry(Protocol):
    """The installed Wing set, epoch-pinned and hash-checked."""

    def load(self, manifest: Manifest, artifact: Wing | None = None) -> WingRef: ...

    def validate(self, manifest: Manifest) -> None: ...

    def resolve(self, refs: list[WingRef], core_epoch: str | None = None) -> list[WingRef]: ...

    def pin(self, core_epoch: str) -> None: ...

    def measured(self, ref: WingRef) -> HostMetrics: ...

    def retrieve(self, x: TypeExpr, m: int) -> list[WingRef]: ...

    def installed(self) -> tuple[WingRef, ...]: ...

    def artifact(self, ref: WingRef) -> Wing: ...

    def check_input(self, ref: WingRef, size: int) -> bool: ...


class InMemoryRegistry:
    """The in-process, file-backed registry of the MVP.

    Installed manifests live in memory; manifests that came from disk can
    be re-materialised from their source path. The optional `seed_dir`
    loads every wings/<name>/manifest.yaml under it at construction, which
    is how the on-disk stub Wings become installed.
    """

    def __init__(self, core_epoch: str | None = None, seed_dir: str | Path | None = None):
        self._epoch = core_epoch
        self._installed: dict[str, dict[str, Manifest]] = {}
        self._artifacts: dict[WingRef, Wing] = {}
        self._metrics: dict[WingRef, HostMetrics] = {}
        self._order: list[WingRef] = []
        if seed_dir is not None:
            for path in sorted(Path(seed_dir).glob("*/manifest.yaml")):
                self.load(load_manifest(path))

    # -- installation ----------------------------------------------------

    def validate(self, manifest: Manifest) -> None:
        validate_manifest(manifest)

    def load(self, manifest: Manifest, artifact: Wing | None = None) -> WingRef:
        self.validate(manifest)
        if self._epoch is not None and manifest.core_epoch != self._epoch:
            raise ManifestError(
                f"wing {manifest.name} declares core_epoch {manifest.core_epoch}, "
                f"pinned epoch is {self._epoch}: incompatible Core epoch"
            )
        ref = WingRef(manifest.name, manifest.version)
        existing = self._installed.get(manifest.name, {}).get(ref.version)
        if existing is not None:
            if existing != manifest:
                raise ManifestError(
                    f"{ref} is already installed with a different manifest"
                )
            if artifact is not None:
                self._artifacts[ref] = artifact
            return ref
        self._installed.setdefault(manifest.name, {})[manifest.version] = manifest
        if artifact is not None:
            self._artifacts[ref] = artifact
        self._order.append(ref)
        return ref

    def pin(self, core_epoch: str) -> None:
        if not isinstance(core_epoch, str) or not core_epoch:
            raise ManifestError("Core epoch pin must be a non-empty string")
        if self._epoch is not None and self._epoch != core_epoch:
            raise ManifestError(
                f"Core epoch is already pinned to {self._epoch}; "
                f"cannot repin to {core_epoch}"
            )
        incompatible = [
            ref
            for ref in self._order
            if self._installed[ref.name][ref.version].core_epoch != core_epoch
        ]
        if incompatible:
            names = ", ".join(str(ref) for ref in incompatible)
            raise ManifestError(
                f"cannot pin Core epoch {core_epoch}; incompatible installed "
                f"wing(s): {names}"
            )
        self._epoch = core_epoch

    def installed(self) -> tuple[WingRef, ...]:
        return tuple(self._order)

    def artifact(self, ref: WingRef) -> Wing:
        if ref not in self._artifacts:
            raise ManifestError(f"no artifact installed for {ref}")
        return self._artifacts[ref]

    # -- ceilings --------------------------------------------------------

    def check_input(self, ref: WingRef, size: int) -> bool:
        manifest = self._installed.get(ref.name, {}).get(ref.version)
        if manifest is None:
            raise ManifestError(f"not installed: {ref}")
        if size < 0:
            raise ManifestError("input size cannot be negative")
        ceilings = manifest.resource_ceilings
        if ceilings is None:
            return True
        return size <= ceilings.max_input_size

    # -- measurement -----------------------------------------------------

    def set_measured(self, metrics: HostMetrics) -> None:
        if metrics.ref not in self._order:
            raise ManifestError(
                f"cannot attach host measurements to uninstalled {metrics.ref}"
            )
        measured_costs = (
            metrics.measured_latency_ms,
            metrics.measured_memory_mb,
            metrics.measured_active_flops,
        )
        if any(
            not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0
            for value in measured_costs
        ):
            raise ManifestError(
                "host-measured latency, memory and active FLOPs must be "
                "finite and non-negative"
            )
        if (
            not isinstance(metrics.measured_reliability, Real)
            or not math.isfinite(float(metrics.measured_reliability))
            or not 0.0 <= metrics.measured_reliability <= 1.0
        ):
            raise ManifestError("host-measured reliability must be in [0, 1]")
        if not isinstance(metrics.measured_utility_features, tuple) or any(
            not isinstance(value, Real) or not math.isfinite(float(value))
            for value in metrics.measured_utility_features
        ):
            raise ManifestError("host-measured utility features must be finite")
        manifest = self._installed[metrics.ref.name][metrics.ref.version]
        ceilings = manifest.resource_ceilings
        if (
            ceilings is not None
            and metrics.measured_memory_mb > ceilings.max_memory_mb
        ):
            raise ManifestError(
                f"{metrics.ref} measured memory {metrics.measured_memory_mb} MB "
                f"exceeds declared ceiling {ceilings.max_memory_mb} MB"
            )
        self._metrics[metrics.ref] = metrics

    def measured(self, ref: WingRef) -> HostMetrics:
        if ref not in self._metrics:
            raise ManifestError(
                f"no host measurements for {ref}; the harness must measure a "
                "wing before it is eligible for routing"
            )
        return self._metrics[ref]

    # -- version resolution ----------------------------------------------

    def _choose(
        self, name: str, range_spec: str, core_epoch: str
    ) -> WingRef | None:
        versions = self._installed.get(name)
        if versions is None:
            return None
        fitting = [
            version
            for version, manifest in versions.items()
            if manifest.core_epoch == core_epoch
            and version_in_range(version, range_spec)
        ]
        if not fitting:
            return None
        fitting.sort(key=_to_semver)
        return WingRef(name, fitting[-1])

    def resolve(self, refs: list[WingRef], core_epoch: str | None = None) -> list[WingRef]:
        epoch = self._epoch if core_epoch is None else core_epoch
        if not isinstance(epoch, str) or not epoch:
            raise ManifestError("no Core epoch pinned: call pin() first")
        resolved: dict[str, str] = {}
        pending = list(refs)
        while pending:
            ref = pending.pop(0)
            manifest = self._installed.get(ref.name, {}).get(ref.version)
            if manifest is None:
                raise VersionError(f"not installed: {ref}")
            if manifest.core_epoch != epoch:
                raise VersionError(
                    f"{ref} belongs to Core epoch {manifest.core_epoch}, "
                    f"resolution requires {epoch}: incompatible Core epoch"
                )
            if manifest.name in resolved:
                if resolved[manifest.name] != manifest.version:
                    raise VersionError(
                        f"version diamond for {manifest.name}: "
                        f"{resolved[manifest.name]} and {manifest.version} "
                        "cannot coexist in one Core epoch"
                    )
                continue
            resolved[manifest.name] = manifest.version
            for dep in manifest.dependencies:
                chosen = self._choose(dep.name, dep.version_range, epoch)
                if chosen is None:
                    raise VersionError(
                        f"{ref} requires {dep.name} {dep.version_range}, "
                        "no installed version satisfies it"
                    )
                pending.append(chosen)
        ordered = []
        seen = set()
        for ref in refs:
            version = resolved[ref.name]
            picked = WingRef(ref.name, version)
            if picked not in seen:
                seen.add(picked)
                ordered.append(picked)
        for name, version in resolved.items():
            picked = WingRef(name, version)
            if picked not in seen:
                seen.add(picked)
                ordered.append(picked)
        return ordered

    # -- stage-1 retrieval ------------------------------------------------

    def retrieve(self, x: TypeExpr, m: int) -> list[WingRef]:
        """Typed retrieval over the installed set, in install order.

        Semantic/embedding retrieval over capability tags arrives in S7;
        stage 1 here is the type-valid subset, which is the contract the
        planner depends on.
        """

        if m <= 0:
            return []
        hits = []
        for ref in self._order:
            manifest = self._installed[ref.name][ref.version]
            if self._epoch is not None and manifest.core_epoch != self._epoch:
                continue
            if any(type_unifies(x, inp) for inp in manifest.input_types):
                hits.append(ref)
            if len(hits) >= m:
                break
        return hits


def typed_retrieval(x: TypeExpr, registry: Registry, m: int) -> list[WingRef]:
    """Stage-1 retrieval: N installed Wings down to m candidates (seam)."""

    return registry.retrieve(x, m)


# The process-wide registry of the MVP. Bench and serving paths may build
# their own instances; these module-level seams operate on this one.
DEFAULT_REGISTRY = InMemoryRegistry()


def resolve(refs: list[WingRef], core_epoch: str | None = None) -> list[WingRef]:
    """Version resolution against one Core epoch. Raises on diamond (seam)."""

    return DEFAULT_REGISTRY.resolve(refs, core_epoch)


def measured(ref: WingRef) -> HostMetrics:
    """Host-owned metrics for one Wing (seam)."""

    return DEFAULT_REGISTRY.measured(ref)
