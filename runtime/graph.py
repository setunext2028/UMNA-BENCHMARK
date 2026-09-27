"""Typed graph construction and validation.

A graph is a set of Wings plus a wiring between their typed outputs and
inputs. Type validity is decided at build time: an invalid composition
raises TypeError and must never reach execution.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .abi import Wing, WingRef, type_unifies
from .registry import ManifestError, Registry, validate_manifest, version_in_range


@dataclass(frozen=True)
class Wiring:
    """Directed edges between Wing outputs and Wing inputs.

    Each edge is (from_wing_name, from_output_index, to_wing_name,
    to_input_index).
    """

    edges: tuple[tuple[str, int, str, int], ...] = ()


@dataclass(frozen=True)
class Cost:
    """Estimates of what this graph costs to run. Estimates, not promises."""

    compute_flops: float
    latency_ms: float
    memory_mb: float


@dataclass(frozen=True)
class Graph:
    """A typed execution graph over pinned Wing versions."""

    graph_id: str
    wings: tuple[WingRef, ...]
    wiring: Wiring
    core_epoch: str


@dataclass(frozen=True)
class Binding:
    """Where calibration attaches: (Core epoch, graph, head, distribution,
    hardware profile). Never to a Wing alone.

    A change to any load-bearing component is a new binding and needs a new
    calibrator.
    """

    core_epoch: str
    graph: Graph
    head: str
    distribution: str
    hardware_profile: str

    @property
    def binding_id(self) -> str:
        return "|".join(
            (
                self.core_epoch,
                self.graph.graph_id,
                self.head,
                self.distribution,
                self.hardware_profile,
            )
        )


class BuildError(TypeError):
    """Raised when a graph cannot be built from the given wings and wiring."""


class CostError(Exception):
    """Raised when host measurements cannot produce a graph cost."""


def _graph_id(refs: tuple[WingRef, ...], wiring: Wiring) -> str:
    identity = "|".join(str(ref) for ref in refs)
    identity += ";" + ";".join(
        f"{src}:{so}->{dst}:{di}" for src, so, dst, di in wiring.edges
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def build(wings: list[Wing], wiring: Wiring) -> Graph:
    """Construct a graph from concrete Wing objects, checking types.

    Every edge output type must unify with the target input type, every
    declared dependency must be satisfied by another wing in the graph,
    and all wings must share one Core epoch. Any violation raises
    BuildError (itself a TypeError, per the seam).
    """

    if not wings:
        raise BuildError("a graph needs at least one wing")
    manifests = []
    for wing in wings:
        manifest = wing.manifest()
        try:
            validate_manifest(manifest)
        except ManifestError as error:
            raise BuildError(
                f"invalid manifest for {manifest.name}: {error}"
            ) from error
        manifests.append(manifest)
    names = [manifest.name for manifest in manifests]
    if len(names) != len(set(names)):
        raise BuildError("a graph cannot contain two versions of one wing")
    by_name = {manifest.name: wing for manifest, wing in zip(manifests, wings)}
    epochs = {manifest.core_epoch for manifest in manifests}
    if len(epochs) > 1:
        raise BuildError(
            f"mixed Core epochs in one graph: {sorted(epochs)}; one "
            "deployment, one epoch"
        )
    epoch = epochs.pop()
    incoming = {name: 0 for name in by_name}
    outgoing = {name: [] for name in by_name}
    for src, so, dst, di in wiring.edges:
        if src not in by_name:
            raise BuildError(f"wiring references unknown wing {src!r}")
        if dst not in by_name:
            raise BuildError(f"wiring references unknown wing {dst!r}")
        outputs = by_name[src].output_types()
        inputs = by_name[dst].input_types()
        if type(so) is not int or so < 0 or so >= len(outputs):
            raise BuildError(f"{src} has no output at index {so}")
        if type(di) is not int or di < 0 or di >= len(inputs):
            raise BuildError(f"{dst} has no input at index {di}")
        if not type_unifies(outputs[so], inputs[di]):
            raise BuildError(
                f"type mismatch on edge {src}:{so} -> {dst}:{di}: "
                f"{outputs[so]} does not unify with {inputs[di]}"
            )
        incoming[dst] += 1
        outgoing[src].append(dst)
        if incoming[dst] > 1:
            raise BuildError(
                f"{dst} has more than one incoming data edge; the S2 Wing "
                "ABI accepts one LatentPacket"
            )
    remaining = dict(incoming)
    ready = [name for name, count in remaining.items() if count == 0]
    visited = 0
    while ready:
        name = ready.pop()
        visited += 1
        for target in outgoing[name]:
            remaining[target] -= 1
            if remaining[target] == 0:
                ready.append(target)
    if visited != len(by_name):
        raise BuildError("graph wiring contains a cycle")
    for manifest in manifests:
        for dep in manifest.dependencies:
            satisfied = any(
                candidate.name == dep.name
                and version_in_range(candidate.version, dep.version_range)
                for candidate in manifests
                if candidate.name != manifest.name
            )
            if not satisfied:
                raise BuildError(
                    f"{manifest.name} requires {dep.name} "
                    f"{dep.version_range}, no wing in the graph satisfies it"
                )
    refs = tuple(
        sorted(
            (WingRef(manifest.name, manifest.version) for manifest in manifests),
            key=lambda ref: (ref.name, ref.version),
        )
    )
    canonical_wiring = Wiring(tuple(sorted(wiring.edges)))
    return Graph(
        graph_id=_graph_id(refs, canonical_wiring),
        wings=refs,
        wiring=canonical_wiring,
        core_epoch=epoch,
    )


def cost(graph: Graph, registry: Registry) -> Cost:
    """Estimate scheduled graph cost from host-owned Wing measurements.

    Compute sums over all scheduled Wings. Latency is the sum of each
    topological wave's slowest Wing; peak memory is the largest concurrent
    wave sum. These match the level-synchronous S2 executor.
    """

    try:
        metrics = {ref.name: registry.measured(ref) for ref in graph.wings}
    except ManifestError as error:
        raise CostError(str(error)) from error

    names = {ref.name for ref in graph.wings}
    indegree = {name: 0 for name in names}
    outgoing = {name: [] for name in names}
    for source, _, target, _ in graph.wiring.edges:
        if source not in names or target not in names:
            raise CostError("graph wiring references a Wing outside the graph")
        indegree[target] += 1
        outgoing[source].append(target)

    compute_flops = sum(
        measurement.measured_active_flops for measurement in metrics.values()
    )
    latency_ms = 0.0
    memory_mb = 0.0
    processed = set()
    ready = sorted(name for name, count in indegree.items() if count == 0)
    while ready:
        latency_ms += max(metrics[name].measured_latency_ms for name in ready)
        memory_mb = max(
            memory_mb,
            sum(metrics[name].measured_memory_mb for name in ready),
        )
        processed.update(ready)
        for name in ready:
            for target in outgoing[name]:
                indegree[target] -= 1
        ready = sorted(
            name
            for name, count in indegree.items()
            if count == 0 and name not in processed
        )
    if processed != names:
        raise CostError("graph wiring contains a cycle or unresolved edge")
    return Cost(compute_flops, latency_ms, memory_mb)
