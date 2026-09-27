import math
import threading
import unittest
from dataclasses import replace
from time import perf_counter_ns, sleep

from runtime.abi import (
    CoreContext,
    Decision,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Manifest,
    Polarity,
    ResourceCeilings,
    TypeExpr,
    TypeSet,
    WingOutput,
)
from runtime.admission import (
    Admission,
    AdmissionError,
    AdmissionRecord,
    admit,
)
from runtime.execute import (
    ExecutionContext,
    ExecutionError,
    LatencyBreakdown,
    ReplayError,
    UpstreamLatency,
    replay,
    run,
)
from runtime.graph import Binding, Wiring, build
from runtime.registry import HostMetrics, InMemoryRegistry


def make_manifest(
    name,
    input_arg="Token",
    output_arg="Token",
    max_input_size=100000,
):
    return Manifest(
        name=name,
        version="1.0.0",
        core_epoch="1.x",
        author="test",
        input_types=TypeSet((TypeExpr("Latent", (input_arg,)),)),
        output_types=TypeSet((TypeExpr("Latent", (output_arg,)),)),
        resource_ceilings=ResourceCeilings(128, max_input_size),
    )


class ConstantWing:
    def __init__(
        self,
        name,
        *,
        data=0.0,
        barrier=None,
        error=None,
        evidence=(),
        max_input_size=100000,
        delay_s=0.0,
    ):
        self._manifest = make_manifest(name, max_input_size=max_input_size)
        self.data = data
        self.barrier = barrier
        self.error = error
        self.evidence = evidence
        self.delay_s = delay_s
        self.calls = 0
        self.inputs = []

    def manifest(self):
        return self._manifest

    def input_types(self):
        return self._manifest.input_types

    def output_types(self):
        return self._manifest.output_types

    def execute(self, ctx: CoreContext, x: LatentPacket):
        self.calls += 1
        self.inputs.append(x.data)
        if self.barrier is not None:
            self.barrier.wait(timeout=1.0)
        if self.delay_s:
            sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        schema = LatentSchema(ctx.core_epoch, 1, "float32")
        latent = LatentPacket(schema, self.data, f"test/{self._manifest.name}")
        return WingOutput(
            latent=latent,
            schema=schema,
            evidence=self.evidence,
            local_score=0.0,
            wing_version=self._manifest.version,
            provenance=f"test/{self._manifest.name}",
        )


def install(registry, wings, memory=None, flops=None):
    memory = memory or {}
    flops = flops or {}
    for wing in wings:
        ref = registry.load(wing.manifest(), wing)
        registry.set_measured(
            HostMetrics(
                ref=ref,
                measured_latency_ms=0.0,
                measured_memory_mb=memory.get(ref.name, 1.0),
                measured_reliability=1.0,
                measured_utility_features=(),
                measured_active_flops=flops.get(ref.name, 1.0),
            )
        )


def input_packet(data="hello"):
    return LatentPacket(
        LatentSchema("1.x", 1, "float32"),
        data,
        "test/input",
    )


def execution_context(registry, graph):
    pipeline_started_ns = perf_counter_ns()
    binding = Binding("1.x", graph, "stub-head", "test", "python-cpu")
    admit_started_ns = perf_counter_ns()
    admission = admit(graph, binding)
    admit_ms = (perf_counter_ns() - admit_started_ns) / 1_000_000.0
    return ExecutionContext(
        registry=registry,
        binding=binding,
        admission=admission,
        pipeline_started_ns=pipeline_started_ns,
        upstream_latency=UpstreamLatency(admit_ms=admit_ms),
    )


class TestAdmission(unittest.TestCase):
    def test_s2_admission_is_bound_and_shadow_only(self):
        wing = ConstantWing("one")
        graph = build([wing], Wiring())
        binding = Binding("1.x", graph, "head", "test", "cpu")
        record = admit(graph, binding)
        self.assertEqual(record.disposition, Admission.SHADOW_ONLY)
        self.assertEqual(record.graph_id, graph.graph_id)
        self.assertEqual(record.binding_id, binding.binding_id)

    def test_admission_rejects_wrong_graph_binding(self):
        first = build([ConstantWing("first")], Wiring())
        second = build([ConstantWing("second")], Wiring())
        binding = Binding("1.x", second, "head", "test", "cpu")
        with self.assertRaises(AdmissionError):
            admit(first, binding)

    def test_admission_rejects_wrong_epoch(self):
        graph = build([ConstantWing("one")], Wiring())
        binding = Binding("9.x", graph, "head", "test", "cpu")
        with self.assertRaises(AdmissionError):
            admit(graph, binding)


class TestExecution(unittest.TestCase):
    def test_chain_emits_complete_trace_and_replays(self):
        wings = [
            ConstantWing("alpha", data=1.0),
            ConstantWing("beta", data=2.0),
            ConstantWing("gamma", data=3.0),
        ]
        registry = InMemoryRegistry(core_epoch="1.x")
        install(
            registry,
            wings,
            memory={"alpha": 8.0, "beta": 16.0, "gamma": 32.0},
            flops={"alpha": 10.0, "beta": 20.0, "gamma": 30.0},
        )
        graph = build(
            wings,
            Wiring((("alpha", 0, "beta", 0), ("beta", 0, "gamma", 0))),
        )
        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(output.decision, Decision.ABSTAIN)
        self.assertEqual(trace.decision, Decision.ABSTAIN)
        self.assertTrue(trace.shadow)
        self.assertEqual(trace.admission, Admission.SHADOW_ONLY)
        self.assertEqual(trace.failures, ())
        self.assertEqual(trace.active_flops, 60.0)
        self.assertEqual(trace.peak_memory_mb, 32.0)
        self.assertEqual(trace.replay_level.value, "route_equivalent")
        self.assertEqual(replay(trace, registry), graph)
        self.assertLessEqual(
            trace.timing_error_ms,
            max(2.0, trace.wall_time_ms * 0.10),
        )
        self.assertEqual(trace.latency.retrieve_ms, 0.0)
        self.assertEqual(trace.latency.plan_ms, 0.0)
        self.assertGreaterEqual(trace.latency.admit_ms, 0.0)

    def test_independent_branches_reach_barrier_in_parallel(self):
        barrier = threading.Barrier(2)
        wings = [
            ConstantWing("left", data="left", barrier=barrier, delay_s=0.03),
            ConstantWing("right", data="right", barrier=barrier, delay_s=0.03),
        ]
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, wings)
        graph = build(wings, Wiring())

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(trace.failures, ())
        self.assertEqual(output.value, {"left": "left", "right": "right"})
        self.assertEqual([wing.calls for wing in wings], [1, 1])
        self.assertLessEqual(
            trace.timing_error_ms,
            max(2.0, trace.wall_time_ms * 0.10),
        )

    def test_sequential_edge_passes_predecessor_output(self):
        first = ConstantWing("first", data="from-first")
        second = ConstantWing("second", data="from-second")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [first, second])
        graph = build(
            [first, second],
            Wiring((("first", 0, "second", 0),)),
        )

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(trace.failures, ())
        self.assertEqual(second.inputs, ["from-first"])
        self.assertEqual(output.value, "from-second")

    def test_wing_exception_is_recorded_and_other_branch_survives(self):
        good = ConstantWing("good", data="kept")
        bad = ConstantWing("bad", error=RuntimeError("boom"))
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [good, bad])
        graph = build([good, bad], Wiring())

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(output.value, "kept")
        self.assertTrue(any("RuntimeError: boom" in item for item in trace.failures))
        self.assertEqual(trace.decision, Decision.ABSTAIN)

    def test_nan_output_is_isolated_and_recorded(self):
        wing = ConstantWing("nan", data=math.nan)
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(output.value, {})
        self.assertTrue(any("non-finite" in item for item in trace.failures))

    def test_wrong_output_schema_is_isolated_and_recorded(self):
        class WrongSchemaWing(ConstantWing):
            def execute(self, ctx, x):
                output = super().execute(ctx, x)
                return replace(
                    output,
                    schema=LatentSchema(ctx.core_epoch, 2, "float32"),
                )

        wing = WrongSchemaWing("wrong-schema")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(output.value, {})
        self.assertTrue(any("schema differs" in item for item in trace.failures))

    def test_wrong_input_epoch_blocks_execution(self):
        wing = ConstantWing("epoch-guarded")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())
        packet = LatentPacket(
            LatentSchema("9.x", 1, "float32"),
            "hello",
            "test/input",
        )

        _, trace = run(graph, packet, execution_context(registry, graph))

        self.assertEqual(wing.calls, 0)
        self.assertTrue(any("input Core epoch" in item for item in trace.failures))

    def test_input_ceiling_blocks_execution(self):
        wing = ConstantWing("tiny", max_input_size=1)
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())

        _, trace = run(graph, input_packet("too large"), execution_context(registry, graph))

        self.assertEqual(wing.calls, 0)
        self.assertTrue(any("input exceeds" in item for item in trace.failures))
        self.assertEqual(trace.active_flops, 0.0)

    def test_missing_artifact_still_returns_trace(self):
        wing = ConstantWing("manifest-only")
        registry = InMemoryRegistry(core_epoch="1.x")
        ref = registry.load(wing.manifest())
        registry.set_measured(
            HostMetrics(ref, 0.0, 1.0, 1.0, (), measured_active_flops=1.0)
        )
        graph = build([wing], Wiring())

        _, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertTrue(any("no artifact installed" in item for item in trace.failures))
        self.assertIsNone(trace.replay_level)

    def test_missing_metrics_still_returns_trace(self):
        wing = ConstantWing("unmeasured")
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(wing.manifest(), wing)
        graph = build([wing], Wiring())

        _, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertTrue(any("no host measurements" in item for item in trace.failures))

    def test_mismatched_admission_record_prevents_execution(self):
        wing = ConstantWing("guarded")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())
        context = execution_context(registry, graph)
        context = replace(
            context,
            admission=replace(context.admission, graph_id="wrong"),
        )

        _, trace = run(graph, input_packet(), context)

        self.assertEqual(wing.calls, 0)
        self.assertTrue(any("admission record graph" in item for item in trace.failures))

    def test_opposing_evidence_remains_a_conflict(self):
        support = EvidencePacket(
            "claim",
            "value",
            Polarity.SUPPORT,
            "supporter",
            "1.0.0",
            "route",
            "test",
            0.9,
            "binding",
            "test",
            "now",
        )
        oppose = replace(
            support,
            polarity=Polarity.OPPOSE,
            source_wing="opposer",
        )
        wings = [
            ConstantWing("supporter", evidence=(support,)),
            ConstantWing("opposer", evidence=(oppose,)),
        ]
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, wings)
        graph = build(wings, Wiring())

        output, trace = run(graph, input_packet(), execution_context(registry, graph))

        self.assertEqual(len(output.conflicts), 1)
        self.assertEqual(trace.conflicts, output.conflicts)
        self.assertEqual(trace.decision, Decision.ABSTAIN)


class TestReplay(unittest.TestCase):
    def test_tampered_graph_id_fails_replay(self):
        wing = ConstantWing("one")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())
        _, trace = run(graph, input_packet(), execution_context(registry, graph))

        with self.assertRaises(ReplayError):
            replay(replace(trace, graph_id="tampered"), registry)

    def test_negative_upstream_latency_is_rejected(self):
        with self.assertRaises(ExecutionError):
            UpstreamLatency(plan_ms=-1.0)

    def test_non_finite_latency_is_rejected(self):
        with self.assertRaises(ExecutionError):
            UpstreamLatency(plan_ms=math.nan)
        with self.assertRaises(ExecutionError):
            LatencyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, math.inf, 0.0, 0.0)

    def test_zero_pipeline_clock_is_rejected(self):
        wing = ConstantWing("one")
        registry = InMemoryRegistry(core_epoch="1.x")
        install(registry, [wing])
        graph = build([wing], Wiring())
        binding = Binding("1.x", graph, "head", "test", "cpu")
        record = AdmissionRecord(
            Admission.SHADOW_ONLY,
            graph.graph_id,
            binding.binding_id,
        )
        with self.assertRaises(ExecutionError):
            ExecutionContext(registry, binding, record, 0)


if __name__ == "__main__":
    unittest.main()
