import unittest

from runtime.abi import Manifest, TypeExpr, TypeSet, type_unifies
from runtime.graph import Binding, BuildError, CostError, Wiring, build, cost
from runtime.registry import HostMetrics, InMemoryRegistry


class TypedWing:
    def __init__(self, name, input_arg, output_arg):
        self._manifest = Manifest(
            name=name,
            version="1.0.0",
            core_epoch="1.x",
            author="test",
            input_types=TypeSet((TypeExpr("Latent", (input_arg,)),)),
            output_types=TypeSet((TypeExpr("Latent", (output_arg,)),)),
        )

    def manifest(self):
        return self._manifest

    def input_types(self):
        return self._manifest.input_types

    def output_types(self):
        return self._manifest.output_types


def make_wing(name, input_arg, output_arg):
    return TypedWing(name, input_arg, output_arg)


class TestTypeUnification(unittest.TestCase):
    def test_equal_types_unify(self):
        self.assertTrue(type_unifies(TypeExpr("Latent", ("Claim",)),
                                     TypeExpr("Latent", ("Claim",))))

    def test_different_families_do_not_unify(self):
        self.assertFalse(type_unifies(TypeExpr("Latent", ("Claim",)),
                                      TypeExpr("Image", ("Claim",))))

    def test_different_args_do_not_unify(self):
        self.assertFalse(type_unifies(TypeExpr("Latent", ("Summary",)),
                                      TypeExpr("Latent", ("Claim",))))


class TestGraphBuild(unittest.TestCase):
    def setUp(self):
        self.alpha = make_wing("alpha", "Text", "Summary")
        self.beta = make_wing("beta", "Text", "Summary")
        self.gamma = make_wing("gamma", "Summary", "Claim")

    def test_full_chain_builds(self):
        graph = build(
            [self.alpha, self.beta, self.gamma],
            Wiring((("beta", 0, "gamma", 0),)),
        )
        self.assertEqual(graph.core_epoch, "1.x")
        self.assertEqual(len(graph.wings), 3)

    def test_type_mismatch_fails(self):
        bad_wing = make_wing("bad", "Image", "Summary")
        with self.assertRaises(BuildError):
            build(
                [bad_wing, self.beta],
                Wiring((("bad", 0, "beta", 0),)),
            )

    def test_cycle_fails_at_build_time(self):
        with self.assertRaises(BuildError):
            build(
                [self.alpha, self.beta],
                Wiring((("alpha", 0, "beta", 0), ("beta", 0, "alpha", 0))),
            )

    def test_empty_graph_fails(self):
        with self.assertRaises(BuildError):
            build([], Wiring())

    def test_duplicate_wing_name_fails(self):
        dup = make_wing("alpha", "Text", "Summary")
        with self.assertRaises(BuildError):
            build([self.alpha, dup], Wiring())

    def test_unknown_wing_in_wiring_fails(self):
        with self.assertRaises(BuildError):
            build(
                [self.alpha],
                Wiring((("unknown", 0, "alpha", 0),)),
            )

    def test_negative_index_fails(self):
        with self.assertRaises(BuildError):
            build(
                [self.alpha, self.beta],
                Wiring((("alpha", -1, "beta", 0),)),
            )

    def test_out_of_range_index_fails(self):
        with self.assertRaises(BuildError):
            build(
                [self.alpha, self.beta],
                Wiring((("alpha", 5, "beta", 0),)),
            )

    def test_multiple_data_edges_to_one_input_fail(self):
        with self.assertRaises(BuildError):
            build(
                [self.alpha, self.beta, self.gamma],
                Wiring((("alpha", 0, "gamma", 0), ("beta", 0, "gamma", 0))),
            )

    def test_mixed_core_epochs_fail(self):
        other_epoch = make_wing("other", "Text", "Summary")
        other_epoch._manifest = Manifest(
            name="other",
            version="1.0.0",
            core_epoch="9.x",
            author="test",
            input_types=TypeSet((TypeExpr("Latent", ("Text",)),)),
            output_types=TypeSet((TypeExpr("Latent", ("Summary",)),)),
        )
        with self.assertRaises(BuildError):
            build([self.alpha, other_epoch], Wiring())

    def test_graph_id_changes_with_wiring(self):
        # Use wings with compatible types for wiring
        w1 = make_wing("w1", "Text", "Summary")
        w2 = make_wing("w2", "Summary", "Claim")
        g1 = build([w1, w2], Wiring())
        g2 = build([w1, w2], Wiring((("w1", 0, "w2", 0),)))
        self.assertNotEqual(g1.graph_id, g2.graph_id)

    def test_graph_id_deterministic(self):
        w1 = make_wing("w1", "Text", "Summary")
        w2 = make_wing("w2", "Summary", "Claim")
        g1 = build([w1, w2], Wiring((("w1", 0, "w2", 0),)))
        g2 = build([w1, w2], Wiring((("w1", 0, "w2", 0),)))
        self.assertEqual(g1.graph_id, g2.graph_id)

    def test_graph_id_ignores_input_list_order(self):
        g1 = build([self.alpha, self.beta], Wiring())
        g2 = build([self.beta, self.alpha], Wiring())
        self.assertEqual(g1.graph_id, g2.graph_id)

    def test_unconnected_inputs_allowed(self):
        graph = build([self.alpha, self.gamma], Wiring())
        self.assertEqual(len(graph.wings), 2)

    def test_binding_id_changes_with_distribution_and_hardware(self):
        graph = build([self.alpha], Wiring())
        b1 = Binding("1.x", graph, "head", "dist1", "cpu")
        b2 = Binding("1.x", graph, "head", "dist2", "cpu")
        b3 = Binding("1.x", graph, "head", "dist1", "gpu")
        self.assertNotEqual(b1.binding_id, b2.binding_id)
        self.assertNotEqual(b1.binding_id, b3.binding_id)


class TestGraphCost(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        self.w1 = make_wing("w1", "Text", "Summary")
        self.w2 = make_wing("w2", "Summary", "Claim")
        for w in (self.w1, self.w2):
            ref = self.registry.load(w.manifest(), w)
            self.registry.set_measured(
                HostMetrics(ref, 1.0, 16.0, 0.99, (0.5,), measured_active_flops=100.0)
            )

    def test_cost_matches_level_synchronous_execution(self):
        graph = build([self.w1, self.w2], Wiring((("w1", 0, "w2", 0),)))
        c = cost(graph, self.registry)
        # Two wings in sequence: sum of both (100 + 100 = 200)
        self.assertEqual(c.compute_flops, 200.0)
        self.assertEqual(c.latency_ms, 2.0)
        self.assertEqual(c.memory_mb, 16.0)

    def test_cost_rejects_an_unmeasured_wing(self):
        w3 = make_wing("w3", "Text", "Summary")
        graph = build([self.w1, w3], Wiring())
        with self.assertRaises(CostError):
            cost(graph, self.registry)


if __name__ == "__main__":
    unittest.main()
