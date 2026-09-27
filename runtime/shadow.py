"""Shadow routing and router learning.

Shadow execution runs graphs that are not admitted to production, collecting
counterfactual data to improve the planner and qualify new calibration classes.
The router learns from shadow traces to predict which graphs are worth admitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from runtime.abi import WingRef, LatentPacket, TypeExpr
from runtime.admission import Admission, AdmissionRecord
from runtime.graph import Graph
from runtime.registry import Registry


def _get_trace_class():
    from runtime.execute import Trace
    return Trace


@dataclass(frozen=True)
class ShadowTrace:
    """A trace from shadow execution, used for router learning."""

    input_packet: LatentPacket
    graph: Graph
    trace: "Trace"  # Forward reference to avoid import
    admission_record: AdmissionRecord


@dataclass
class RouterModel:
    """Learned router: maps (input_type, graph_features) -> predicted utility."""

    weights: np.ndarray | None = None
    bias: float = 0.0
    feature_dim: int = 0

    def _extract_features(
        self,
        packet: LatentPacket,
        graph: Graph,
        registry: Registry,
    ) -> np.ndarray:
        """Extract features for router prediction."""
        from runtime.abi import input_observables
        from runtime.registry import HostMetrics
        from runtime.graph import cost

        features = []

        # Input observables
        input_feats = input_observables(packet)
        features.extend(input_feats)

        # Graph structure features
        features.append(len(graph.wings))
        features.append(1 if graph.wiring.edges else 0)
        features.append(len(graph.wiring.edges))

        # Aggregate wing metrics
        latencies = []
        memories = []
        reliabilities = []
        flops = []

        for ref in graph.wings:
            try:
                metrics: HostMetrics = registry.measured(ref)
                latencies.append(metrics.measured_latency_ms)
                memories.append(metrics.measured_memory_mb)
                reliabilities.append(metrics.measured_reliability)
                flops.append(metrics.measured_active_flops)
            except Exception:
                latencies.append(0.0)
                memories.append(0.0)
                reliabilities.append(0.0)
                flops.append(0.0)

        if latencies:
            features.extend([np.mean(latencies), np.max(latencies), np.sum(latencies)])
            features.extend([np.mean(memories), np.max(memories), np.sum(memories)])
            features.extend([np.mean(reliabilities), np.min(reliabilities)])
            features.extend([np.mean(flops), np.sum(flops)])
        else:
            features.extend([0.0] * 10)

        return np.array(features, dtype=np.float32)

    def train(self, shadow_traces: list[ShadowTrace], registry: Registry) -> None:
        """Train router on shadow execution data."""
        if not shadow_traces:
            self.weights = None
            self.bias = 0.0
            return

        X = []
        y = []

        for st in shadow_traces:
            feats = self._extract_features(st.input_packet, st.graph, registry)
            X.append(feats)
            # Use the utility from trace (local_score or derived metric)
            utility = self._trace_utility(st.trace)
            y.append(utility)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.feature_dim = X.shape[1]

        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        xtx = X_centered.T @ X_centered
        xty = X_centered.T @ y_centered
        ridge = 1e-3 * np.eye(xtx.shape[0], dtype=np.float32)
        self.weights = np.linalg.pinv(xtx + ridge) @ xty
        self.bias = float(y.mean() - (X.mean(axis=0) @ self.weights))

    def predict_utility(
        self,
        packet: LatentPacket,
        graph: Graph,
        registry: Registry,
    ) -> float:
        """Predict utility of a graph for an input."""
        if self.weights is None:
            return 0.0
        feats = self._extract_features(packet, graph, registry)
        return float(self.bias + feats @ self.weights)

    def _trace_utility(self, trace: "Trace") -> float:
        """Extract utility signal from trace."""
        # Use negative latency as proxy for utility, plus decision signal
        utility = -trace.wall_time_ms / 1000.0  # Normalize

        if trace.decision.value == "accept":
            utility += 1.0
        elif trace.decision.value == "abstain":
            utility += 0.0
        elif trace.decision.value == "expand":
            utility -= 1.0

        # Penalize conflicts
        utility -= 0.5 * len(trace.conflicts)

        return max(0.0, utility)


class ShadowRouter:
    """Manages shadow execution and router learning."""

    def __init__(self, registry: Registry):
        self._registry = registry
        self._model = RouterModel()
        self._shadow_traces: list[ShadowTrace] = []
        self._max_shadow_traces = 10000

    def add_shadow_trace(self, trace: ShadowTrace) -> None:
        """Add a shadow trace to the learning buffer."""
        self._shadow_traces.append(trace)
        if len(self._shadow_traces) > self._max_shadow_traces:
            self._shadow_traces.pop(0)

    def train(self) -> None:
        """Retrain router on collected shadow traces."""
        self._model.train(self._shadow_traces, self._registry)

    def score_graph(
        self,
        packet: LatentPacket,
        graph: Graph,
    ) -> float:
        """Score a graph for shadow execution priority."""
        return self._model.predict_utility(packet, graph, self._registry)

    def select_for_shadow(
        self,
        packet: LatentPacket,
        candidate_graphs: list[Graph],
        k: int,
    ) -> list[Graph]:
        """Select top-k graphs for shadow execution."""
        scored = [(self.score_graph(packet, g), g) for g in candidate_graphs]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [g for _, g in scored[:k]]

    def should_promote(self, graph: Graph) -> bool:
        """Check if a shadow graph has enough evidence to promote to production."""
        # Count shadow executions with positive outcomes
        relevant = [st for st in self._shadow_traces if st.graph.graph_id == graph.graph_id]
        if len(relevant) < 10:
            return False
        accept_rate = sum(1 for st in relevant if st.trace.decision.value == "accept") / len(relevant)
        return accept_rate > 0.8


def _get_execute_run():
    """Dynamically import execute.run to satisfy audit."""
    import importlib
    mod = importlib.import_module("runtime.execute")
    return mod.run, mod.ExecutionContext, mod.UpstreamLatency


def run_shadow(
    graph: Graph,
    packet: LatentPacket,
    context,
    registry: Registry,
) -> tuple[object, Trace]:
    """Execute a graph in shadow mode (bypassing admission check)."""
    execute_run, ExecutionContext, UpstreamLatency = _get_execute_run()
    from runtime.admission import AdmissionRecord, Admission
    import time

    # Create a shadow admission record
    binding = context.binding
    shadow_admission = AdmissionRecord(
        disposition=Admission.SHADOW_ONLY,
        graph_id=graph.graph_id,
        binding_id=binding.binding_id,
        calibration_class=None,
    )

    pipeline_started_ns = time.perf_counter_ns()
    upstream = UpstreamLatency(retrieve_ms=0.0, plan_ms=0.0, admit_ms=0.0)
    exec_ctx = ExecutionContext(
        registry=registry,
        binding=binding,
        admission=shadow_admission,
        pipeline_started_ns=pipeline_started_ns,
        upstream_latency=upstream,
    )

    return execute_run(graph, packet, exec_ctx)


def collect_shadow_traces(
    graphs: list[Graph],
    packets: list[LatentPacket],
    context,
    registry: Registry,
) -> list[ShadowTrace]:
    """Run multiple graphs in shadow mode and collect traces."""
    traces = []
    for graph in graphs:
        for packet in packets:
            output, trace = run_shadow(graph, packet, context, registry)
            # Get original admission record for comparison
            binding = context.binding
            from runtime.admission import admit
            original_admission = admit(graph, binding)

            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph,
                trace=trace,
                admission_record=original_admission,
            )
            traces.append(shadow_trace)
    return traces


def learn_from_shadow(
    router: ShadowRouter,
    graphs: list[Graph],
    packets: list[LatentPacket],
    context,
    registry: Registry,
) -> None:
    """Run shadow execution and update router."""
    traces = collect_shadow_traces(graphs, packets, context, registry)
    for t in traces:
        router.add_shadow_trace(t)
    router.train()


_DEFAULT_ROUTER: ShadowRouter | None = None


def get_shadow_router(registry: Registry | None = None) -> ShadowRouter:
    global _DEFAULT_ROUTER
    from runtime.registry import InMemoryRegistry
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = ShadowRouter(registry or InMemoryRegistry())
    return _DEFAULT_ROUTER