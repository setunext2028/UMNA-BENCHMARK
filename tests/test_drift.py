"""Tests for cumulative drift gating."""

import unittest

from runtime.abi import WingRef
from runtime.registry import InMemoryRegistry
from runtime.drift import (
    Tier,
    DriftStatus,
    DriftConfig,
    TierConfig,
    CapabilityState,
    DriftCheckResult,
    InterferenceEntry,
    PromotionRecord,
    DriftGate,
    get_drift_gate,
    compute_power,
)


class TestDriftConfig(unittest.TestCase):
    def test_default_config(self):
        config = DriftConfig()
        self.assertEqual(config.epsilon, 0.01)
        self.assertEqual(config.E, 0.05)

    def test_custom_config(self):
        config = DriftConfig(epsilon=0.02, E=0.10)
        self.assertEqual(config.epsilon, 0.02)
        self.assertEqual(config.E, 0.10)


class TestTierConfig(unittest.TestCase):
    def test_tier_a_config(self):
        config = TierConfig(Tier.A)
        self.assertEqual(config.tier, Tier.A)
        self.assertEqual(config.alpha, 0.05)
        self.assertEqual(config.beta, 0.20)
        self.assertEqual(config.delta, 0.02)
        self.assertEqual(config.min_samples, 100)

    def test_tier_b_config(self):
        config = TierConfig(Tier.B)
        self.assertEqual(config.tier, Tier.B)
        self.assertEqual(config.alpha, 0.10)
        self.assertEqual(config.beta, 0.20)
        self.assertEqual(config.delta, 0.03)
        self.assertEqual(config.min_samples, 50)

    def test_tier_c_config(self):
        config = TierConfig(Tier.C)
        self.assertEqual(config.tier, Tier.C)
        self.assertEqual(config.alpha, 0.20)
        self.assertEqual(config.beta, 0.20)
        self.assertEqual(config.delta, 0.05)
        self.assertEqual(config.min_samples, 30)


class TestDriftGate(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry()
        self.gate = DriftGate(self.registry)
        self.ref_alpha = WingRef("alpha", "1.0.0")
        self.ref_gamma = WingRef("gamma", "1.0.0")

    def test_register_capability(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.assertIn("alpha", self.gate._capabilities)
        self.assertEqual(self.gate._capabilities["alpha"].tier, Tier.A)

    def test_duplicate_registration_fails(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        with self.assertRaises(ValueError):
            self.gate.register_capability(self.ref_alpha, Tier.B)

    def test_set_origin_performance(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        state = self.gate._capabilities["alpha"]
        self.assertEqual(state.origin_performance, 0.95)
        self.assertEqual(state.current_performance, 0.95)
        self.assertEqual(state.history, [0.95])

    def test_set_origin_twice_fails(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        with self.assertRaises(ValueError):
            self.gate.set_origin_performance(self.ref_alpha, 0.94)

    def test_update_performance(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.945)

        state = self.gate._capabilities["alpha"]
        self.assertEqual(state.previous_performance, 0.95)
        self.assertEqual(state.current_performance, 0.945)
        self.assertAlmostEqual(state.step_drift, -0.005, places=10)
        self.assertAlmostEqual(state.origin_drift, -0.005, places=10)
        self.assertEqual(state.samples, 1)
        self.assertEqual(len(state.history), 2)

    def test_check_drift_pass(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.945)  # Within epsilon

        result = self.gate.check_drift(self.ref_alpha)
        self.assertEqual(result.status, DriftStatus.PASS)
        self.assertAlmostEqual(result.step_drift, -0.005, places=10)
        self.assertAlmostEqual(result.origin_drift, -0.005, places=10)

    def test_check_drift_step_fail(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.93)  # Step drift -0.02 < -0.01

        result = self.gate.check_drift(self.ref_alpha)
        self.assertEqual(result.status, DriftStatus.STEP_DRIFT)

    def test_check_drift_origin_fail(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        # Multiple small steps that accumulate origin drift > E
        # epsilon=0.01, E=0.05
        # Each step: -0.01 (at boundary), after 6 steps = -0.06 > -0.05
        self.gate.update_performance(self.ref_alpha, 0.94)   # step=-0.01, origin=-0.01
        self.gate.update_performance(self.ref_alpha, 0.93)   # step=-0.01, origin=-0.02
        self.gate.update_performance(self.ref_alpha, 0.92)   # step=-0.01, origin=-0.03
        self.gate.update_performance(self.ref_alpha, 0.91)   # step=-0.01, origin=-0.04
        self.gate.update_performance(self.ref_alpha, 0.90)   # step=-0.01, origin=-0.05
        self.gate.update_performance(self.ref_alpha, 0.89)   # step=-0.01, origin=-0.06 > -0.05

        result = self.gate.check_drift(self.ref_alpha)
        self.assertEqual(result.status, DriftStatus.ORIGIN_DRIFT)

    def test_check_drift_both_fail(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        # Large step drift that exceeds both epsilon and E
        self.gate.update_performance(self.ref_alpha, 0.89)  # -0.06 step, -0.06 origin

        result = self.gate.check_drift(self.ref_alpha)
        self.assertEqual(result.status, DriftStatus.BOTH)

    def test_can_promote(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.945)

        can, result = self.gate.can_promote(self.ref_alpha)
        self.assertTrue(can)
        self.assertEqual(result.status, DriftStatus.PASS)

    def test_can_promote_fails(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.93)

        can, result = self.gate.can_promote(self.ref_alpha)
        self.assertFalse(can)
        self.assertEqual(result.status, DriftStatus.STEP_DRIFT)

    def test_demote(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)

        self.assertTrue(self.gate.is_eligible(self.ref_alpha))
        self.gate.demote(self.ref_alpha, "step drift alarm")
        self.assertFalse(self.gate.is_eligible(self.ref_alpha))

        state = self.gate._capabilities["alpha"]
        self.assertTrue(state.demoted)
        self.assertEqual(state.demotion_reason, "step drift alarm")

    def test_restore_eligibility(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.demote(self.ref_alpha, "step drift alarm")
        self.gate.restore_eligibility(self.ref_alpha)

        self.assertTrue(self.gate.is_eligible(self.ref_alpha))
        state = self.gate._capabilities["alpha"]
        self.assertFalse(state.demoted)
        self.assertEqual(state.demotion_reason, "")

    def test_is_eligible_unknown_capability(self):
        # Unknown capabilities default to eligible
        unknown = WingRef("unknown", "1.0.0")
        self.assertTrue(self.gate.is_eligible(unknown))

    def test_record_promotion(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.945)

        record = self.gate.record_promotion(
            self.ref_alpha, "1.0.0", "1.1.0",
            step_drift=-0.005, origin_drift=-0.005,
            admitted=True, reason="both gates pass"
        )

        self.assertEqual(record.event_id, 1)
        self.assertEqual(record.promoted_wing, self.ref_alpha)
        self.assertEqual(record.old_version, "1.0.0")
        self.assertEqual(record.new_version, "1.1.0")
        self.assertTrue(record.admitted)
        self.assertEqual(len(self.gate._promotion_events), 1)

    def test_interference_matrix(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.register_capability(self.ref_gamma, Tier.B)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.set_origin_performance(self.ref_gamma, 0.90)
        self.gate.update_performance(self.ref_alpha, 0.945)
        self.gate.update_performance(self.ref_gamma, 0.895)

        self.gate.record_promotion(
            self.ref_alpha, "1.0.0", "1.1.0",
            step_drift=-0.005, origin_drift=-0.005,
            admitted=True, reason="test"
        )

        matrix = self.gate.get_interference_matrix()
        self.assertEqual(len(matrix), 1)  # One entry for gamma
        self.assertEqual(matrix[0].affected_capability, self.ref_gamma)
        self.assertEqual(matrix[0].tier, Tier.B)

    def test_get_all_status(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.register_capability(self.ref_gamma, Tier.B)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.set_origin_performance(self.ref_gamma, 0.90)
        self.gate.update_performance(self.ref_alpha, 0.945)
        self.gate.update_performance(self.ref_gamma, 0.895)

        status = self.gate.get_all_status()
        self.assertIn("alpha", status)
        self.assertIn("gamma", status)
        self.assertEqual(status["alpha"].status, DriftStatus.PASS)
        self.assertEqual(status["gamma"].status, DriftStatus.PASS)

    def test_get_demoted(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.register_capability(self.ref_gamma, Tier.B)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.demote(self.ref_alpha, "drift")

        demoted = self.gate.get_demoted()
        self.assertEqual(len(demoted), 1)
        self.assertEqual(demoted[0], self.ref_alpha)

    def test_get_promotion_history(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.945)

        self.gate.record_promotion(
            self.ref_alpha, "1.0.0", "1.1.0",
            step_drift=-0.005, origin_drift=-0.005,
            admitted=True, reason="test"
        )

        history = self.gate.get_promotion_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].promoted_wing, self.ref_alpha)

    def test_tier_a_max_computation(self):
        self.gate._total_samples = 20000
        max_a = self.gate.compute_tier_a_max()
        self.assertGreater(max_a, 0)
        self.assertEqual(self.gate._tier_a_max, max_a)

        # Second call should return cached value
        self.assertEqual(self.gate.compute_tier_a_max(), max_a)

    def test_get_tier_a_count(self):
        self.gate.register_capability(self.ref_alpha, Tier.A)
        self.gate.register_capability(self.ref_gamma, Tier.B)
        self.assertEqual(self.gate.get_tier_a_count(), 1)

    def test_custom_drift_config(self):
        custom = DriftConfig(epsilon=0.05, E=0.10)
        self.gate.register_capability(self.ref_alpha, Tier.A, custom)
        self.gate.set_origin_performance(self.ref_alpha, 0.95)
        self.gate.update_performance(self.ref_alpha, 0.92)  # -0.03 step, within custom epsilon

        result = self.gate.check_drift(self.ref_alpha)
        self.assertEqual(result.status, DriftStatus.PASS)
        self.assertEqual(result.epsilon, 0.05)
        self.assertEqual(result.E, 0.10)

    def test_unregistered_check_fails(self):
        with self.assertRaises(ValueError):
            self.gate.check_drift(self.ref_alpha)


class TestComputePower(unittest.TestCase):
    def test_compute_power(self):
        # Basic sanity check - power should be between 0 and 1
        power = compute_power(n=100, alpha=0.05, beta=0.20, delta=0.02)
        self.assertGreaterEqual(power, 0.0)
        self.assertLessEqual(power, 1.0)

    def test_power_increases_with_n(self):
        power_100 = compute_power(n=100, alpha=0.05, beta=0.20, delta=0.02)
        power_500 = compute_power(n=500, alpha=0.05, beta=0.20, delta=0.02)
        self.assertGreaterEqual(power_500, power_100)

    def test_power_decreases_with_alpha(self):
        power_05 = compute_power(n=100, alpha=0.05, beta=0.20, delta=0.02)
        power_10 = compute_power(n=100, alpha=0.10, beta=0.20, delta=0.02)
        self.assertGreaterEqual(power_10, power_05)


if __name__ == "__main__":
    unittest.main()