"""Typed capability ABI: the contract every Wing, planner and runtime term obeys.

S0 contract: types only, no behaviour. Everything the host measures is
deliberately absent from the manifest types; the host-measured side lives
in runtime.registry.HostMetrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass(frozen=True)
class TypeExpr:
    """One declared neural type, e.g. TypeExpr("Image", ("XRay", "Chest"))."""

    family: str
    args: tuple[str, ...] = ()


def type_unifies(given: TypeExpr, required: TypeExpr) -> bool:
    """Can a value of type `given` be accepted where `required` is declared?

    MVP strictness: the families must match and the argument sets must be
    equal. A more permissive subtyping relation is an S3 decision, not this
    one.
    """

    return given.family == required.family and given.args == required.args


@dataclass(frozen=True)
class TypeSet:
    """The ordered input or output contract of a Wing."""

    entries: tuple[TypeExpr, ...] = ()

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int) -> TypeExpr:
        return self.entries[index]


@dataclass(frozen=True)
class LatentSchema:
    """The latent/type contract valid for one Core epoch."""

    core_epoch: str
    dimensions: int
    dtype: str


@dataclass(frozen=True)
class LatentPacket:
    """The typed value exchanged between Wings and the Core."""

    schema: LatentSchema
    data: object
    provenance: str = ""


@dataclass(frozen=True)
class CoreContext:
    """Fixed context handed to a Wing at execute time."""

    core_epoch: str
    latent_schema: LatentSchema
    binding_id: str | None = None


class Polarity(Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidencePacket:
    """One scored claim from one Wing version on one route."""

    claim_type: str
    claim_value: str
    polarity: Polarity
    source_wing: str
    source_version: str
    route_id: str
    provenance: str
    local_score: float
    calibration_binding: str
    validity_domain: str
    timestamp: str


@dataclass(frozen=True)
class ConflictSet:
    """Contradictory claims kept apart; never averaged by the runtime."""

    claim_type: str
    claims: tuple[EvidencePacket, ...]


@dataclass(frozen=True)
class WingOutput:
    """What a Wing may return. It carries no claim about its own reliability."""

    latent: LatentPacket
    schema: LatentSchema
    evidence: tuple[EvidencePacket, ...] = ()
    local_score: float = 0.0
    wing_version: str = ""
    provenance: str = ""


class Wing(Protocol):
    """The capability module contract."""

    def manifest(self) -> Manifest: ...

    def input_types(self) -> TypeSet: ...

    def output_types(self) -> TypeSet: ...

    def execute(self, ctx: CoreContext, x: LatentPacket) -> WingOutput: ...


@dataclass(frozen=True)
class Dependency:
    name: str
    version_range: str


@dataclass(frozen=True)
class ResourceCeilings:
    """Ceilings the host enforces, never measurements."""

    max_memory_mb: int
    max_input_size: int


@dataclass(frozen=True)
class ManifestIntegrity:
    artifact_hash: str


@dataclass(frozen=True)
class ManifestProvenance:
    training_dataset_ids: tuple[str, ...] = ()
    training_code_commit: str = ""


@dataclass(frozen=True)
class Manifest:
    """The complete allow-list a Wing may declare.

    There is deliberately no reliability, accuracy, latency, evidence-gain,
    relevance or confidence field here: those are host-owned. The validator
    in runtime.registry rejects any YAML key not representable by this type.
    """

    name: str
    version: str
    core_epoch: str
    author: str
    input_types: TypeSet
    output_types: TypeSet
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    resource_ceilings: ResourceCeilings | None = None
    integrity: ManifestIntegrity | None = None
    provenance: ManifestProvenance | None = None


class Decision(Enum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"
    EXPAND = "expand"


class ReplayLevel(Enum):
    L1 = "bitwise"
    L2 = "numerically_close"
    L3 = "route_equivalent"
    L4 = "decision_equivalent"


@dataclass(frozen=True)
class Output:
    """The final served (or abstained) result of one execution."""

    value: object
    decision: Decision
    evidence: tuple[EvidencePacket, ...] = ()
    conflicts: tuple[ConflictSet, ...] = ()


@dataclass(frozen=True)
class WingRef:
    """An exact, pinned reference to an installed Wing version."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


def input_observables(packet: LatentPacket) -> tuple[float, float, float, float]:
    """Extract simple observable features from input packet.

    Returns tuple of (mean, std, max_abs, length).
    """
    import numpy as np
    data = packet.data
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32, copy=False)
        if arr.ndim > 1:
            arr = arr[0]
        return (
            float(arr.mean()),
            float(arr.std()),
            float(np.max(np.abs(arr))),
            float(arr.shape[0]) if arr.size > 0 else 0.0,
        )
    else:
        return (0.0, 0.0, 0.0, 0.0)
