"""Tests for safety interrupts."""

import unittest

import numpy as np

from runtime.abi import (
    CoreContext,
    Decision,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Manifest,
    Output,
    Polarity,
    TypeExpr,
    TypeSet,
    WingOutput,
    WingRef,
)
from runtime.admission import Admission, AdmissionRecord, admit
from runtime.graph import Binding, Wiring, build
from runtime.registry import HostMetrics, InMemoryRegistry
from runtime.interrupts import (
    InterruptType,
    InterruptAction,
    InterruptPacket,
    InterruptPolicy,
    InterruptBinding,
    InterruptWing,
    InterruptRouter,
    evaluate_interrupts,
    get_interrupt_router,
)
from risk.conformal import fit_split_conformal


def make_manifest(name, input_arg="Token", output_arg="Token"):
    return Manifest(
        name=name,
        version="1.0.0",
        core_epoch="1.x",
        author="test",
        input_types=TypeSet((TypeExpr("Latent", (input_arg,)),)),
        output_types=TypeSet((TypeExpr("Latent", (output_arg,)),)),
    )


class TestInterruptWing(InterruptWing):
    """Test interrupt wing that can emit interrupts."""
    def __init__(self, name, interrupt_on=False, interrupt_confidence=0.95):
        self._manifest = make_manifest(name)
        self._interrupt_on = interrupt_on
        self._interrupt_confidence = interrupt_confidence

    def manifest(self):
        return self._manifest

    def input_types(self):
        return self._manifest.input_types

    def output_types(self):
        return self._manifest.output_types

    def execute(self, ctx: CoreContext, x: LatentPacket):
        schema = LatentSchema(ctx.core_epoch, 1, "float32")
        latent = LatentPacket(schema, 0.5, f"test/{self._manifest.name}")
        
        evidence = []
        if self._interrupt_on:
            evidence.append(EvidencePacket(
                claim_type="interrupt",
                claim_value="emergency",
                polarity=Polarity.SUPPORT,
                source_wing=self._manifest.name,
                source_version=self._manifest.version,
                route_id="interrupt",
                provenance=f"interrupt/{self._manifest.name}",
                local_score=self._interrupt_confidence,
                calibration_binding="interrupt",
                validity_domain="interrupt:emergency",
                timestamp="now",
            ))
        
        return WingOutput(
            latent=latent,
            schema=schema,
            evidence=tuple(evidence),
            local_score=0.5,
            wing_version=self._manifest.version,
            provenance=f"test/{self._manifest.name}",
        )


class TestInterrupts(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        
        # Create interrupt wing
        self.interrupt_wing = TestInterruptWing("safety_wing", interrupt_on=True, interrupt_confidence=0.95)
        self.normal_wing = TestInterruptWing("normal_wing", interrupt_on=False)
        
        self.registry.load(self.interrupt_wing.manifest(), self.interrupt_wing)
        self.registry.load(self.normal_wing.manifest(), self.normal_wing)
        
        self.registry.set_measured(HostMetrics(
            ref=WingRef("safety_wing", "1.0.0"),
            measured_latency_ms=1.0,
            measured_memory_mb=8.0,
            measured_reliability=0.95,
            measured_utility_features=(0.9, 0.8),
            measured_active_flops=1000.0,
        ))
        self.registry.set_measured(HostMetrics(
            ref=WingRef("normal_wing", "1.0.0"),
            measured_latency_ms=1.0,
            measured_memory_mb=8.0,
            measured_reliability=0.95,
            measured_utility_features=(0.9, 0.8),
            measured_active_flops=1000.0,
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

    def _make_context_for_registry(self, registry, graph, admission):
        from runtime.execute import ExecutionContext, UpstreamLatency
        import time
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        return ExecutionContext(
            registry=registry,
            binding=binding,
            admission=admission,
            pipeline_started_ns=time.perf_counter_ns(),
            upstream_latency=UpstreamLatency(),
        )

    def test_interrupt_types_exist(self):
        self.assertEqual(InterruptType.EMERGENCY.value, "emergency")
        self.assertEqual(InterruptType.CONTRAINDICATION.value, "contraindication")
        self.assertEqual(InterruptType.HARD_CONFLICT.value, "hard_conflict")
        self.assertEqual(InterruptType.SECURITY.value, "security")
        self.assertEqual(InterruptType.LEGAL_POLICY.value, "legal_policy")

    def test_interrupt_actions_exist(self):
        self.assertEqual(InterruptAction.APPEND_GUARD.value, "append_guard")
        self.assertEqual(InterruptAction.FORCE_ESCALATION.value, "force_escalation")
        self.assertEqual(InterruptAction.FORCE_ABSTAIN.value, "force_abstain")
        self.assertEqual(InterruptAction.REQUEST_SECONDARY.value, "request_secondary")
        self.assertEqual(InterruptAction.BLOCK_ACTION.value, "block_action")

    def test_interrupt_packet_creation(self):
        wing = TestInterruptWing("test", interrupt_on=True)
        interrupt = wing.emit_interrupt(
            InterruptType.EMERGENCY,
            InterruptAction.FORCE_ABSTAIN,
            0.95,
            {"reason": "test"},
        )
        
        self.assertEqual(interrupt.interrupt_type, InterruptType.EMERGENCY)
        self.assertEqual(interrupt.action, InterruptAction.FORCE_ABSTAIN)
        self.assertEqual(interrupt.source_wing, "test")
        self.assertEqual(interrupt.confidence, 0.95)
        self.assertEqual(interrupt.payload, {"reason": "test"})

    def test_interrupt_policy_registration(self):
        router = InterruptRouter(self.registry)
        
        policy = InterruptPolicy(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            authorized_wings=(WingRef("safety_wing", "1.0.0"),),
            min_confidence=0.9,
        )
        router.register_policy(policy)
        
        self.assertIn(InterruptType.EMERGENCY, router._policies)
        self.assertEqual(router._policies[InterruptType.EMERGENCY].action, InterruptAction.FORCE_ABSTAIN)

    def test_interrupt_binding_registration(self):
        router = InterruptRouter(self.registry)
        
        # Create a simple graph for the interrupt binding
        wing = TestInterruptWing("interrupt_wing")
        self.registry.load(wing.manifest(), wing)
        graph = build([wing], Wiring())
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        
        # Create calibrator
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.7, 0.95, 100)
        labels = (rng.uniform(0, 1, 100) < probs).astype(int)
        calibrator = fit_split_conformal(probs, labels, alpha=0.1, rng=rng)
        
        interrupt_binding = InterruptBinding(
            interrupt_type=InterruptType.EMERGENCY,
            graph=graph,
            binding=binding,
            calibrator=calibrator,
            false_interrupt_rate=0.05,
            recall=0.90,
        )
        router.register_binding(interrupt_binding)
        
        self.assertIn(InterruptType.EMERGENCY, router._bindings)
        self.assertEqual(router._bindings[InterruptType.EMERGENCY].calibrator, calibrator)

    def test_check_interrupt_fires(self):
        router = InterruptRouter(self.registry)
        
        # Create calibrator that accepts high confidence
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.8, 0.95, 100)
        labels = (rng.uniform(0, 1, 100) < probs).astype(int)
        calibrator = fit_split_conformal(probs, labels, alpha=0.1, rng=rng)
        
        policy = InterruptPolicy(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            authorized_wings=(WingRef("safety_wing", "1.0.0"),),
            calibrator=calibrator,
            min_confidence=0.9,
        )
        router.register_policy(policy)
        
        # Create graph with interrupt wing
        graph = build([self.interrupt_wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([0.5], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)
        
        interrupt = router.check_interrupt(InterruptType.EMERGENCY, packet, context)
        
        self.assertIsNotNone(interrupt)
        self.assertEqual(interrupt.interrupt_type, InterruptType.EMERGENCY)
        self.assertEqual(interrupt.action, InterruptAction.FORCE_ABSTAIN)
        self.assertEqual(interrupt.source_wing, "safety_wing")
        self.assertGreaterEqual(interrupt.confidence, 0.9)

    def test_check_interrupt_no_fire_below_confidence(self):
        # Fresh registry for this test
        registry = InMemoryRegistry(core_epoch="1.x")
        
        router = InterruptRouter(registry)
        
        # Calibrator with high threshold
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.5, 0.7, 100)  # Low confidence
        labels = (rng.uniform(0, 1, 100) < probs).astype(int)
        calibrator = fit_split_conformal(probs, labels, alpha=0.1, rng=rng)
        
        policy = InterruptPolicy(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            authorized_wings=(WingRef("safety_wing", "1.0.0"),),
            calibrator=calibrator,
            min_confidence=0.9,
        )
        router.register_policy(policy)
        
        # Wing with low confidence interrupt
        low_conf_wing = TestInterruptWing("low_conf", interrupt_on=True, interrupt_confidence=0.5)
        registry.load(low_conf_wing.manifest(), low_conf_wing)
        
        graph = build([low_conf_wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([0.5], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context_for_registry(registry, graph, admission)
        
        interrupt = router.check_interrupt(InterruptType.EMERGENCY, packet, context)
        
        self.assertIsNone(interrupt)

    def test_check_interrupt_no_fire_unauthorized_wing(self):
        # Fresh registry for this test
        registry = InMemoryRegistry(core_epoch="1.x")
        
        router = InterruptRouter(registry)
        
        policy = InterruptPolicy(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            authorized_wings=(WingRef("safety_wing", "1.0.0"),),  # Only safety_wing authorized
            min_confidence=0.9,
        )
        router.register_policy(policy)
        
        # Normal wing tries to emit interrupt but is not authorized
        normal_wing = TestInterruptWing("normal_wing", interrupt_on=False)
        registry.load(normal_wing.manifest(), normal_wing)
        
        graph = build([normal_wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([0.5], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context_for_registry(registry, graph, admission)
        
        interrupt = router.check_interrupt(InterruptType.EMERGENCY, packet, context)
        
        self.assertIsNone(interrupt)

    def test_apply_interrupt_append_guard(self):
        router = InterruptRouter(self.registry)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        # Mock trace
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.APPEND_GUARD,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Decision should remain ACCEPT
        self.assertEqual(new_output.decision, Decision.ACCEPT)
        # Evidence should have interrupt guard added
        self.assertEqual(len(new_output.evidence), 1)
        self.assertEqual(new_output.evidence[0].claim_type, "interrupt")
        self.assertEqual(new_output.evidence[0].claim_value, "emergency")

    def test_apply_interrupt_force_abstain(self):
        router = InterruptRouter(self.registry)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.CONTRAINDICATION,
            action=InterruptAction.FORCE_ABSTAIN,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Decision should be forced to ABSTAIN
        self.assertEqual(new_output.decision, Decision.ABSTAIN)
        # Evidence should have interrupt guard
        self.assertEqual(len(new_output.evidence), 1)

    def test_apply_interrupt_force_escalation(self):
        router = InterruptRouter(self.registry)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.HARD_CONFLICT,
            action=InterruptAction.FORCE_ESCALATION,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Decision should remain ACCEPT but conflict added
        self.assertEqual(new_output.decision, Decision.ACCEPT)
        self.assertEqual(len(new_output.conflicts), 1)
        self.assertEqual(new_output.conflicts[0].claim_type, "escalation")

    def test_apply_interrupt_request_secondary(self):
        router = InterruptRouter(self.registry)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.SECURITY,
            action=InterruptAction.REQUEST_SECONDARY,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Decision should be forced to ABSTAIN with secondary request
        self.assertEqual(new_output.decision, Decision.ABSTAIN)
        self.assertEqual(len(new_output.conflicts), 1)
        self.assertEqual(new_output.conflicts[0].claim_type, "secondary_route_request")

    def test_apply_interrupt_block_action(self):
        router = InterruptRouter(self.registry)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.LEGAL_POLICY,
            action=InterruptAction.BLOCK_ACTION,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Decision should be forced to ABSTAIN with action blocked
        self.assertEqual(new_output.decision, Decision.ABSTAIN)
        self.assertEqual(len(new_output.conflicts), 1)
        self.assertEqual(new_output.conflicts[0].claim_type, "action_blocked")

    def test_evaluate_interrupts_priority_order(self):
        router = InterruptRouter(self.registry)
        
        # Register multiple interrupt policies
        for int_type, action in [
            (InterruptType.EMERGENCY, InterruptAction.FORCE_ABSTAIN),
            (InterruptType.CONTRAINDICATION, InterruptAction.FORCE_ABSTAIN),
            (InterruptType.HARD_CONFLICT, InterruptAction.FORCE_ESCALATION),
            (InterruptType.SECURITY, InterruptAction.REQUEST_SECONDARY),
            (InterruptType.LEGAL_POLICY, InterruptAction.BLOCK_ACTION),
        ]:
            policy = InterruptPolicy(
                interrupt_type=int_type,
                action=action,
                authorized_wings=(WingRef("safety_wing", "1.0.0"),),
                min_confidence=0.5,
            )
            router.register_policy(policy)
        
        # Enable all interrupts on the wing
        self.interrupt_wing._interrupt_on = True
        self.interrupt_wing._interrupt_confidence = 0.95
        
        # Create wing that emits multiple interrupt types
        class MultiInterruptWing(TestInterruptWing):
            def execute(self, ctx, x):
                schema = LatentSchema(ctx.core_epoch, 1, "float32")
                latent = LatentPacket(schema, 0.5, f"test/{self._manifest.name}")
                evidence = []
                for int_type in InterruptType:
                    evidence.append(EvidencePacket(
                        claim_type="interrupt",
                        claim_value=int_type.value,
                        polarity=Polarity.SUPPORT,
                        source_wing=self._manifest.name,
                        source_version=self._manifest.version,
                        route_id="interrupt",
                        provenance=f"interrupt/{self._manifest.name}",
                        local_score=0.95,
                        calibration_binding="interrupt",
                        validity_domain=f"interrupt:{int_type.value}",
                        timestamp="now",
                    ))
                return WingOutput(
                    latent=latent,
                    schema=schema,
                    evidence=tuple(evidence),
                    local_score=0.5,
                    wing_version=self._manifest.version,
                    provenance=f"test/{self._manifest.name}",
                )
        
        multi_wing = MultiInterruptWing("multi")
        self.registry.load(multi_wing.manifest(), multi_wing)
        
        graph = build([multi_wing], Wiring())
        packet = LatentPacket(LatentSchema("1.x", 1, "float32"), np.array([0.5], dtype=np.float32))
        binding = Binding("1.x", graph, "head", "test", "python-cpu")
        admission = admit(graph, binding)
        context = self._make_context(graph, admission)
        
        protected_output = Output(
            value=0.5,
            decision=Decision.ACCEPT,
            evidence=(),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=(),
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        new_output, new_trace = evaluate_interrupts(
            protected_output, protected_trace, router, packet, context
        )
        
        # EMERGENCY should fire first and stop further processing (force_abstain)
        self.assertEqual(new_output.decision, Decision.ABSTAIN)

    def test_interrupt_never_replaces_protected_route(self):
        """Test that interrupts never silently replace the protected route."""
        router = InterruptRouter(self.registry)
        
        policy = InterruptPolicy(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            authorized_wings=(WingRef("safety_wing", "1.0.0"),),
            min_confidence=0.5,
        )
        router.register_policy(policy)
        
        protected_output = Output(
            value={"result": "protected_value"},
            decision=Decision.ACCEPT,
            evidence=(
                EvidencePacket(
                    claim_type="class", claim_value="1", polarity=Polarity.SUPPORT,
                    source_wing="protected", source_version="1.0.0", route_id="test",
                    provenance="protected", local_score=0.9, calibration_binding="test",
                    validity_domain="test", timestamp="now",
                ),
            ),
            conflicts=(),
        )
        
        from runtime.execute import Trace, LatencyBreakdown, ReplayLevel
        protected_trace = Trace(
            input_hash="test",
            core_epoch="1.x",
            graph_id="test",
            binding_id="test",
            calibration_class=None,
            wing_versions={},
            wiring=Wiring(),
            admission=Admission.PRODUCTION,
            latency=LatencyBreakdown(0, 0, 0, 0, 0, 0, 0, 0),
            wall_time_ms=1.0,
            active_flops=1000.0,
            peak_memory_mb=8.0,
            evidence=protected_output.evidence,
            conflicts=(),
            decision=Decision.ACCEPT,
            replay_level=ReplayLevel.L3,
            failures=(),
            shadow=False,
        )
        
        interrupt = InterruptPacket(
            interrupt_type=InterruptType.EMERGENCY,
            action=InterruptAction.FORCE_ABSTAIN,
            source_wing="safety_wing",
            source_version="1.0.0",
            route_id="test",
            provenance="interrupt/safety_wing",
            confidence=0.95,
        )
        
        new_output, new_trace = router.apply_interrupt(interrupt, protected_output, protected_trace)
        
        # Original value should be preserved
        self.assertEqual(new_output.value, {"result": "protected_value"})
        # Original evidence should be preserved
        self.assertEqual(len(new_output.evidence), 2)  # Original + interrupt guard
        # Original evidence first
        self.assertEqual(new_output.evidence[0].source_wing, "protected")
        # Interrupt evidence second
        self.assertEqual(new_output.evidence[1].claim_type, "interrupt")

    def test_global_interrupt_router(self):
        router1 = get_interrupt_router(self.registry)
        router2 = get_interrupt_router(self.registry)
        self.assertIs(router1, router2)


if __name__ == "__main__":
    unittest.main()