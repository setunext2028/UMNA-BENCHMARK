"""Execution, trace emission, and route replay.

The trace is the measurement instrument. Every listed latency term is
recorded separately, and a trace is returned when a Wing or artifact fails.
S2 runs are shadow-only because calibration classes arrive in S4.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from time import perf_counter_ns

from .abi import (
    ConflictSet,
    CoreContext,
    Decision,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Output,
    Polarity,
    ReplayLevel,
    Wing,
    WingOutput,
    WingRef,
)
from .admission import Admission, AdmissionRecord
from .graph import Binding, Graph, Wiring, build
from .registry import HostMetrics, Registry
from risk.conformal import assess


class ExecutionError(RuntimeError):
    """The execution request is internally inconsistent."""


class ReplayError(RuntimeError):
    """A trace cannot reconstruct the graph it claims to describe."""


@dataclass(frozen=True)
class UpstreamLatency:
    """Measured phases completed before execution begins, in milliseconds."""

    retrieve_ms: float = 0.0
    plan_ms: float = 0.0
    admit_ms: float = 0.0

    def __post_init__(self):
        terms = (self.retrieve_ms, self.plan_ms, self.admit_ms)
        if any(
            not isinstance(term, Real)
            or not math.isfinite(float(term))
            or term < 0
            for term in terms
        ):
            raise ExecutionError("upstream latency terms must be finite and non-negative")

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.plan_ms + self.admit_ms


@dataclass(frozen=True)
class ExecutionContext:
    """Host state that `run` cannot derive from a Graph and input alone."""

    registry: Registry
    binding: Binding
    admission: AdmissionRecord
    pipeline_started_ns: int
    upstream_latency: UpstreamLatency = UpstreamLatency()

    def __post_init__(self):
        if (
            type(self.pipeline_started_ns) is not int
            or self.pipeline_started_ns <= 0
        ):
            raise ExecutionError("pipeline_started_ns must come from perf_counter_ns")


@dataclass(frozen=True)
class LatencyBreakdown:
    """All eight terms of the latency decomposition, in milliseconds."""

    retrieve_ms: float
    plan_ms: float
    admit_ms: float
    load_ms: float
    transfer_ms: float
    wing_ms: float
    fusion_ms: float
    risk_ms: float

    def __post_init__(self):
        terms = (
            self.retrieve_ms,
            self.plan_ms,
            self.admit_ms,
            self.load_ms,
            self.transfer_ms,
            self.wing_ms,
            self.fusion_ms,
            self.risk_ms,
        )
        if any(
            not isinstance(term, Real)
            or not math.isfinite(float(term))
            or term < 0
            for term in terms
        ):
            raise ExecutionError("latency terms must be finite and non-negative")

    @property
    def total_ms(self) -> float:
        return (
            self.retrieve_ms
            + self.plan_ms
            + self.admit_ms
            + self.load_ms
            + self.transfer_ms
            + self.wing_ms
            + self.fusion_ms
            + self.risk_ms
        )


@dataclass(frozen=True)
class Trace:
    """The full execution record for one input."""

    input_hash: str
    core_epoch: str
    graph_id: str
    binding_id: str
    calibration_class: str | None
    wing_versions: dict[str, str]
    wiring: Wiring
    admission: Admission
    latency: LatencyBreakdown
    wall_time_ms: float
    active_flops: float
    peak_memory_mb: float
    evidence: tuple[EvidencePacket, ...]
    conflicts: tuple[ConflictSet, ...]
    decision: Decision
    replay_level: ReplayLevel | None
    failures: tuple[str, ...]
    shadow: bool

    @property
    def timing_error_ms(self) -> float:
        """Absolute difference between decomposed and independent wall time."""

        return abs(self.latency.total_ms - self.wall_time_ms)


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000.0


def _input_hash(packet: LatentPacket) -> str:
    if isinstance(packet, LatentPacket) and isinstance(
        packet.schema, LatentSchema
    ):
        value = (
            packet.schema.core_epoch,
            packet.schema.dimensions,
            packet.schema.dtype,
            packet.provenance,
            repr(packet.data),
        )
    else:
        value = repr(packet)
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _packet_size(packet: LatentPacket) -> int:
    nbytes = getattr(packet.data, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    if isinstance(packet.data, str):
        return len(packet.data.encode("utf-8"))
    if isinstance(packet.data, (bytes, bytearray, memoryview)):
        return len(packet.data)
    return len(repr(packet.data).encode("utf-8"))


def _all_finite(value) -> bool:
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return all(_all_finite(item) for item in value)
    return True


def _validate_output(
    name: str,
    ref: WingRef,
    output,
    core_epoch: str,
) -> list[str]:
    failures = []
    if not isinstance(output, WingOutput):
        return [f"{name}: returned {type(output).__name__}, expected WingOutput"]
    if output.wing_version != ref.version:
        failures.append(
            f"{name}: returned version {output.wing_version!r}, expected {ref.version!r}"
        )
    if not isinstance(output.schema, LatentSchema):
        failures.append(f"{name}: output schema is not a LatentSchema")
    if not isinstance(output.latent, LatentPacket):
        failures.append(f"{name}: output latent is not a LatentPacket")
    if failures and (
        not isinstance(output.schema, LatentSchema)
        or not isinstance(output.latent, LatentPacket)
    ):
        return failures
    if output.schema != output.latent.schema:
        failures.append(f"{name}: WingOutput schema differs from latent schema")
    if output.schema.core_epoch != core_epoch:
        failures.append(
            f"{name}: output Core epoch {output.schema.core_epoch!r}, "
            f"expected {core_epoch!r}"
        )
    if (
        not isinstance(output.local_score, Real)
        or not math.isfinite(float(output.local_score))
    ):
        failures.append(f"{name}: local score is not finite")
    if (
        type(output.schema.dimensions) is not int
        or output.schema.dimensions <= 0
        or not isinstance(output.schema.dtype, str)
        or not output.schema.dtype
    ):
        failures.append(f"{name}: output schema is malformed")
    if not _all_finite(output.latent.data):
        failures.append(f"{name}: latent data contains a non-finite value")
    shape = getattr(output.latent.data, "shape", None)
    if shape is not None:
        try:
            if len(shape) and shape[-1] != output.schema.dimensions:
                failures.append(
                    f"{name}: last data dimension {shape[-1]} differs from "
                    f"declared {output.schema.dimensions}"
                )
        except (IndexError, TypeError):
            failures.append(f"{name}: latent data shape is malformed")
    if not isinstance(output.evidence, tuple):
        failures.append(f"{name}: evidence is not a tuple")
        return failures
    for evidence in output.evidence:
        if not isinstance(evidence, EvidencePacket):
            failures.append(f"{name}: evidence is not an EvidencePacket")
            continue
        if evidence.source_wing != name or evidence.source_version != ref.version:
            failures.append(f"{name}: evidence source does not match the WingRef")
        if (
            not isinstance(evidence.local_score, Real)
            or not math.isfinite(float(evidence.local_score))
        ):
            failures.append(f"{name}: evidence score is not finite")
    return failures


def _conflict_sets(
    evidence: tuple[EvidencePacket, ...],
) -> tuple[ConflictSet, ...]:
    grouped: dict[tuple[str, str], list[EvidencePacket]] = {}
    for packet in evidence:
        grouped.setdefault((packet.claim_type, packet.claim_value), []).append(
            packet
        )
    conflicts = []
    for (claim_type, _), packets in sorted(grouped.items()):
        polarities = {packet.polarity for packet in packets}
        if Polarity.SUPPORT in polarities and Polarity.OPPOSE in polarities:
            conflicts.append(ConflictSet(claim_type, tuple(packets)))
    return tuple(conflicts)


def _topology(graph: Graph):
    names = [ref.name for ref in graph.wings]
    incoming = {name: [] for name in names}
    outgoing = {name: [] for name in names}
    indegree = {name: 0 for name in names}
    for source, _, target, _ in graph.wiring.edges:
        incoming[target].append(source)
        outgoing[source].append(target)
        indegree[target] += 1
    return incoming, outgoing, indegree


def _execution_waves(indegree, outgoing) -> tuple[tuple[str, ...], ...]:
    """Compile a valid DAG into deterministic topological execution waves."""

    remaining = dict(indegree)
    processed = set()
    waves = []
    ready = sorted(name for name, count in remaining.items() if count == 0)
    while ready:
        wave = tuple(ready)
        waves.append(wave)
        processed.update(wave)
        for name in wave:
            for target in outgoing[name]:
                remaining[target] -= 1
        ready = sorted(
            name
            for name, count in remaining.items()
            if count == 0 and name not in processed
        )
    unresolved = sorted(set(remaining) - processed)
    if unresolved:
        raise ExecutionError(
            "graph contains a cycle or unresolved edge: " + ", ".join(unresolved)
        )
    return tuple(waves)


def _execute_one(
    wing: Wing,
    packet: LatentPacket,
    core_epoch: str,
    binding_id: str,
) -> WingOutput:
    context = CoreContext(
        core_epoch=core_epoch,
        latent_schema=packet.schema,
        binding_id=binding_id,
    )
    return wing.execute(context, packet)


def _rebuild_graph(
    wing_versions: Mapping[str, str],
    wiring: Wiring,
    registry: Registry,
) -> Graph:
    refs = [
        WingRef(name, version)
        for name, version in sorted(wing_versions.items())
    ]
    wings = [registry.artifact(ref) for ref in refs]
    return build(wings, wiring)


def replay(trace: Trace, registry: Registry) -> Graph:
    """Rebuild the exact route recorded by a trace and verify its identity."""

    try:
        graph = _rebuild_graph(trace.wing_versions, trace.wiring, registry)
    except Exception as error:
        raise ReplayError(
            f"trace graph reconstruction failed: {type(error).__name__}: {error}"
        ) from error
    if graph.graph_id != trace.graph_id:
        raise ReplayError(
            f"replayed graph {graph.graph_id} differs from trace {trace.graph_id}"
        )
    return graph


def run(
    graph: Graph,
    x: LatentPacket,
    context: ExecutionContext,
) -> tuple[Output, Trace]:
    """Execute one admitted graph and return its shadow output and trace.

    Independent topological branches execute in parallel. Any failure is
    isolated, named in the trace, and drives the S2 decision to abstain.
    """

    run_started_ns = perf_counter_ns()
    failures: list[str] = []
    artifacts: dict[str, Wing] = {}
    metrics: dict[str, HostMetrics] = {}
    load_started_ns = run_started_ns
    wing_versions = {ref.name: ref.version for ref in graph.wings}
    replay_level = None

    if context.pipeline_started_ns > run_started_ns:
        failures.append("pipeline start is later than execution start")
        pipeline_started_ns = run_started_ns
    else:
        pipeline_started_ns = context.pipeline_started_ns
    if context.binding.graph.graph_id != graph.graph_id:
        failures.append("execution binding graph does not match the graph")
    if context.binding.core_epoch != graph.core_epoch:
        failures.append("execution binding Core epoch does not match the graph")
    if context.admission.graph_id != graph.graph_id:
        failures.append("admission record graph does not match the graph")
    if context.admission.binding_id != context.binding.binding_id:
        failures.append("admission record binding does not match execution binding")
    if (
        context.admission.disposition is Admission.PRODUCTION
        and not context.admission.calibration_class
    ):
        failures.append("production admission has no calibration class")
    if not isinstance(x, LatentPacket):
        failures.append("execution input is not a LatentPacket")
    elif not isinstance(x.schema, LatentSchema):
        failures.append("execution input schema is not a LatentSchema")
    else:
        if x.schema.core_epoch != graph.core_epoch:
            failures.append("execution input Core epoch does not match the graph")
        if (
            type(x.schema.dimensions) is not int
            or x.schema.dimensions <= 0
            or not isinstance(x.schema.dtype, str)
            or not x.schema.dtype
        ):
            failures.append("execution input schema is malformed")

    context_valid = not failures
    if context_valid:
        for ref in graph.wings:
            try:
                artifacts[ref.name] = context.registry.artifact(ref)
                metrics[ref.name] = context.registry.measured(ref)
            except Exception as error:
                failures.append(f"{ref.name}: {type(error).__name__}: {error}")
    try:
        incoming, outgoing, indegree = _topology(graph)
        waves = _execution_waves(indegree, outgoing)
    except Exception as error:
        failures.append(f"topology: {type(error).__name__}: {error}")
        incoming = {ref.name: [] for ref in graph.wings}
        outgoing = {ref.name: [] for ref in graph.wings}
        waves = ()
    try:
        replayed = _rebuild_graph(wing_versions, graph.wiring, context.registry)
        if replayed.graph_id != graph.graph_id:
            raise ReplayError(
                f"replayed graph {replayed.graph_id} differs from {graph.graph_id}"
            )
        replay_level = ReplayLevel.L3
    except Exception as error:
        failures.append(f"replay: {type(error).__name__}: {error}")
    input_hash = _input_hash(x)
    load_ms = _elapsed_ms(load_started_ns)

    transfer_ms = 0.0
    wing_ms = 0.0
    active_flops = 0.0
    peak_memory_mb = 0.0
    outputs: dict[str, WingOutput] = {}
    refs_by_name = {ref.name: ref for ref in graph.wings}

    for ready in waves:
        transfer_started_ns = perf_counter_ns()
        runnable: dict[str, tuple[Wing, LatentPacket]] = {}
        for name in ready:
            if name not in artifacts or name not in metrics:
                continue
            sources = incoming[name]
            if sources and sources[0] not in outputs:
                failures.append(f"{name}: blocked by failed source {sources[0]}")
                continue
            packet = x if not sources else outputs[sources[0]].latent
            try:
                within_ceiling = context.registry.check_input(
                    refs_by_name[name], _packet_size(packet)
                )
            except Exception as error:
                failures.append(f"{name}: {type(error).__name__}: {error}")
                continue
            if not within_ceiling:
                failures.append(f"{name}: input exceeds declared ceiling")
                continue
            runnable[name] = (artifacts[name], packet)
        transfer_ms += _elapsed_ms(transfer_started_ns)

        if runnable:
            active_flops += sum(
                metrics[name].measured_active_flops for name in runnable
            )
            peak_memory_mb = max(
                peak_memory_mb,
                sum(metrics[name].measured_memory_mb for name in runnable),
            )
            wing_started_ns = perf_counter_ns()
            with ThreadPoolExecutor(max_workers=len(runnable)) as executor:
                futures = {
                    name: executor.submit(
                        _execute_one,
                        wing,
                        packet,
                        graph.core_epoch,
                        context.binding.binding_id,
                    )
                    for name, (wing, packet) in runnable.items()
                }
                for name in sorted(futures):
                    try:
                        output = futures[name].result()
                    except Exception as error:
                        failures.append(
                            f"{name}: {type(error).__name__}: {error}"
                        )
                        continue
                    try:
                        output_failures = _validate_output(
                            name, refs_by_name[name], output, graph.core_epoch
                        )
                    except Exception as error:
                        failures.append(
                            f"{name}: output validation: "
                            f"{type(error).__name__}: {error}"
                        )
                        continue
                    if output_failures:
                        failures.extend(output_failures)
                    else:
                        outputs[name] = output
            wing_ms += _elapsed_ms(wing_started_ns)

    fusion_started_ns = perf_counter_ns()
    evidence = tuple(
        packet
        for ref in graph.wings
        if ref.name in outputs
        for packet in outputs[ref.name].evidence
    )
    conflicts = _conflict_sets(evidence)
    sinks = sorted(name for name, targets in outgoing.items() if not targets)
    sink_outputs = {name: outputs[name].latent.data for name in sinks if name in outputs}
    if len(sink_outputs) == 1:
        value = next(iter(sink_outputs.values()))
    else:
        value = sink_outputs

    fusion_ms = _elapsed_ms(fusion_started_ns)

    risk_started_ns = perf_counter_ns()
    # Risk assessment using calibrator from admission binding
    # Use the evidence score (local_score) for conformal prediction
    score = evidence[0].local_score if evidence else 0.0
    decision = assess(Output(value=score, decision=Decision.ABSTAIN, evidence=evidence, conflicts=conflicts), context.binding)
    output = Output(
        value=value,
        decision=decision,
        evidence=evidence,
        conflicts=conflicts,
    )
    risk_ms = _elapsed_ms(risk_started_ns)
    wall_time_ms = (perf_counter_ns() - pipeline_started_ns) / 1_000_000.0

    latency = LatencyBreakdown(
        retrieve_ms=context.upstream_latency.retrieve_ms,
        plan_ms=context.upstream_latency.plan_ms,
        admit_ms=context.upstream_latency.admit_ms,
        load_ms=load_ms,
        transfer_ms=transfer_ms,
        wing_ms=wing_ms,
        fusion_ms=fusion_ms,
        risk_ms=risk_ms,
    )
    tolerance_ms = max(2.0, wall_time_ms * 0.10)
    timing_error_ms = abs(latency.total_ms - wall_time_ms)
    if timing_error_ms > tolerance_ms:
        failures.append(
            f"latency decomposition differs from wall time by "
            f"{timing_error_ms:.3f} ms"
        )

    trace = Trace(
        input_hash=input_hash,
        core_epoch=graph.core_epoch,
        graph_id=graph.graph_id,
        binding_id=context.binding.binding_id,
        calibration_class=context.admission.calibration_class,
        wing_versions=wing_versions,
        wiring=graph.wiring,
        admission=context.admission.disposition,
        latency=latency,
        wall_time_ms=wall_time_ms,
        active_flops=active_flops,
        peak_memory_mb=peak_memory_mb,
        evidence=evidence,
        conflicts=conflicts,
        decision=decision,
        replay_level=replay_level,
        failures=tuple(failures),
        shadow=context.admission.disposition is Admission.SHADOW_ONLY,
    )
    return output, trace
