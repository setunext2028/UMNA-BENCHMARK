"""Tests for shadow routing and router learning."""

import unittest

import numpy as np

from runtime.abi import (
    CoreContext,
    Decision,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Manifest,
    Polarity,
    TypeExpr,
    TypeSet,
    WingOutput,
)
from runtime.admission import Admission, AdmissionRecord, admit
from runtime.graph import Binding, Wiring, build
from runtime.registry import HostMetrics, InMemoryRegistry, WingRef
from runtime.shadow import ShadowRouter, run_shadow, ShadowTrace, get_shadow_router, learn_from_shadow


def make_manifest(name, input_arg="Token", output_arg="Token", max_input_size=100000):
    return Manifest(
        name=name,
        version="1.0.0",
        core_epoch="1.x",
        author="test",
        input_types=TypeSet((TypeExpr("Latent", (input_arg,)),)),
        output_types=TypeSet((TypeExpr("Latent", (output_arg,)),)),
        resource_ceilings=None,
    )


class ConstantWing:
    def __init__(self, name, *, data=0.0, error=None, evidence=()):
        self._manifest = make_manifest(name, max_input_size=100000)
        self.data = data
        self.error = error
        self.evidence = evidence
        self.calls = 0

    def manifest(self):
        return self._manifest

    def input_types(self):
        return self._manifest.input_types

    def output_types(self):
        return self._manifest.output_types

    def execute(self, ctx: CoreContext, x: LatentPacket):
        self.calls += 1
        if self.error is not None:
            raise self.error
        schema = LatentSchema(ctx.core_epoch, 1, "float32")
        latent = LatentPacket(schema, self.data, f"test/{self._manifest.name}")
        return WingOutput(
            latent=latent,
            schema=schema,
            evidence=self.evidence,
            local_score=self.data,
            wing_version=self._manifest.version,
            provenance=f"test/{self._manifest.name}",
        )


class TestShadowRouting(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        wing_a = ConstantWing("alpha", data=0.9, evidence=(
            EvidencePacket(
                claim_type="class", claim_value="1", polarity=Polarity.SUPPORT,
                source_wing="alpha", source_version="1.0.0", route_id="test",
                provenance="test", local_score=0.9, calibration_binding="test",
                validity_domain="test", timestamp="now",
            ),
        ))
        wing_b = ConstantWing("beta", data=0.8, evidence=(
            EvidencePacket(
                claim_type="class", claim_value="1", polarity=Polarity.SUPPORT,
                source_wing="beta", source_version="1.0.0", route_id="test",
                provenance="test", local_score=0.8, calibration_binding="test",
                validity_domain="test", timestamp="now",
            ),
        ))
        self.registry.load(wing_a.manifest(), wing_a)
        self.registry.load(wing_b.manifest(), wing_b)
        self.registry.set_measured(HostMetrics(
            ref=WingRef("alpha", "1.0.0"),
            measured_latency_ms=1.0,
            measured_memory_mb=8.0,
            measured_reliability=0.95,
            measured_utility_features=(0.9, 0.8),
            measured_active_flops=1000.0,
        ))
        self.registry.set_measured(HostMetrics(
            ref=WingRef("beta", "1.0.0"),
            measured_latency_ms=1.5,
            measured_memory_mb=8.0,
            measured_reliability=0.93,
            measured_utility_features=(0.8, 0.7),
            measured_active_flops=1100.0,
        ))

    def _make_context(self, graph, admission):
        from runtime.execute import ExecutionContext, UpstreamLatency
        import time
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        return ExecutionContext(
            registry=self.registry,
            binding=binding,
            admission=admission,
            pipeline_started_ns=time.perf_counter_ns(),
            upstream_latency=UpstreamLatency(),
        )

    def test_run_shadow_executes_graph(self):
        wing_a = ConstantWing("alpha", data=0.9)
        wing_b = ConstantWing("beta", data=0.8)
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(wing_a.manifest(), wing_a)
        registry.load(wing_b.manifest(), wing_b)
        registry.set_measured(HostMetrics(
            ref=WingRef("alpha", "1.0.0"), measured_latency_ms=1.0,
            measured_memory_mb=8.0, measured_reliability=0.95,
            measured_utility_features=(0.9, 0.8), measured_active_flops=1000.0,
        ))
        registry.set_measured(HostMetrics(
            ref=WingRef("beta", "1.0.0"), measured_latency_ms=1.5,
            measured_memory_mb=8.0, measured_reliability=0.93,
            measured_utility_features=(0.8, 0.7), measured_active_flops=1100.0,
        ))

        graph = build([wing_a, wing_b], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        output, trace = run_shadow(graph, packet, context, registry)

        self.assertIsNotNone(trace)
        self.assertEqual(trace.shadow, True)
        self.assertEqual(trace.admission, Admission.SHADOW_ONLY)

    def test_shadow_trace_captures_decision(self):
        wing = ConstantWing("alpha", data=0.9)
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(wing.manifest(), wing)
        registry.set_measured(HostMetrics(
            ref=WingRef("alpha", "1.0.0"), measured_latency_ms=1.0,
            measured_memory_mb=8.0, measured_reliability=0.95,
            measured_utility_features=(0.9, 0.8), measured_active_flops=1000.0,
        ))

        graph = build([wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        output, trace = run_shadow(graph, packet, context, registry)

        self.assertIn(trace.decision.value, ("accept", "abstain", "expand"))

    def test_shadow_router_initialization(self):
        router = ShadowRouter(self.registry)
        self.assertIsNotNone(router)
        self.assertEqual(len(router._shadow_traces), 0)

    def test_shadow_router_add_trace(self):
        router = ShadowRouter(self.registry)
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        output, trace = run_shadow(graph, packet, context, self.registry)

        shadow_trace = ShadowTrace(
            input_packet=packet,
            graph=graph,
            trace=trace,
            admission_record=admission,
        )
        router.add_shadow_trace(shadow_trace)

        self.assertEqual(len(router._shadow_traces), 1)

    def test_shadow_router_train(self):
        router = ShadowRouter(self.registry)
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        for _ in range(5):
            output, trace = run_shadow(graph, packet, context, self.registry)
            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph,
                trace=trace,
                admission_record=admission,
            )
            router.add_shadow_trace(shadow_trace)

        router.train()

        self.assertIsNotNone(router._model.weights)
        self.assertEqual(router._model.feature_dim, router._model.weights.shape[0])

    def test_shadow_router_score_graph(self):
        router = ShadowRouter(self.registry)
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        for _ in range(5):
            output, trace = run_shadow(graph, packet, context, self.registry)
            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph,
                trace=trace,
                admission_record=admission,
            )
            router.add_shadow_trace(shadow_trace)

        router.train()

        score = router.score_graph(packet, graph)
        self.assertIsInstance(score, float)

    def test_shadow_router_select_for_shadow(self):
        router = ShadowRouter(self.registry)
        wing_a = ConstantWing("alpha", data=0.9)
        wing_b = ConstantWing("beta", data=0.8)
        graph_a = build([wing_a], Wiring())
        graph_b = build([wing_b], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph_a, "head", "test", "python-cpu")
        admission = admit(graph_a, binding)
        context = self._make_context(graph_a, admission)

        for _ in range(5):
            output, trace = run_shadow(graph_a, packet, context, self.registry)
            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph_a,
                trace=trace,
                admission_record=admission,
            )
            router.add_shadow_trace(shadow_trace)

        router.train()

        selected = router.select_for_shadow(packet, [graph_a, graph_b], 1)
        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0], [graph_a, graph_b])

    def test_should_promote_requires_evidence(self):
        router = ShadowRouter(self.registry)
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())

        self.assertFalse(router.should_promote(graph))

        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        # With default ABSTAIN decisions, promotion should not happen
        for _ in range(15):
            output, trace = run_shadow(graph, packet, context, self.registry)
            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph,
                trace=trace,
                admission_record=admission,
            )
            router.add_shadow_trace(shadow_trace)

        self.assertFalse(router.should_promote(graph))

    def test_should_promote_with_accept_decisions(self):
        """Test promotion works when traces have accept decisions."""
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        from runtime.abi import ConflictSet

        router = ShadowRouter(self.registry)
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())

        self.assertFalse(router.should_promote(graph))

        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)

        # Create mock traces with accept decisions
        for _ in range(15):
            trace = Trace(
                input_hash="test",
                core_epoch="1.x",
                graph_id=graph.graph_id,
                binding_id=binding.binding_id,
                calibration_class=None,
                wing_versions={"alpha": "1.0.0"},
                wiring=Wiring(),
                admission=Admission.SHADOW_ONLY,
                latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
                wall_time_ms=1.0,
                active_flops=1000.0,
                peak_memory_mb=8.0,
                evidence=(),
                conflicts=(),
                decision=Decision.ACCEPT,
                replay_level=ReplayLevel.L3,
                failures=(),
                shadow=True,
            )
            shadow_trace = ShadowTrace(
                input_packet=packet,
                graph=graph,
                trace=trace,
                admission_record=admission,
            )
            router.add_shadow_trace(shadow_trace)

        self.assertTrue(router.should_promote(graph))

    def test_learn_from_shadow(self):
        wing = ConstantWing("alpha", data=0.9)
        graph = build([wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([1.0], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)

        router = get_shadow_router(self.registry)
        learn_from_shadow(router, [graph], [packet], context, self.registry)

        self.assertGreater(len(router._shadow_traces), 0)


if __name__ == "__main__":
    unittest.main()