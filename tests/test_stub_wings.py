"""Tests for real Wings (S3)."""

import numpy as np
import unittest

from runtime.abi import CoreContext, LatentPacket, LatentSchema
from runtime.registry import InMemoryRegistry, load_manifest
from wings.alpha.model import AlphaWing
from wings.beta.model import BetaWing
from wings.gamma.model import GammaWing
from wings.delta.model import DeltaWing
from wings.epsilon.model import EpsilonWing
from wings.zeta.model import ZetaWing


class TestRealWings(unittest.TestCase):
    def setUp(self):
        self.ctx = CoreContext(
            core_epoch="1.x",
            latent_schema=LatentSchema(core_epoch="1.x", dimensions=8, dtype="float32"),
        )

    def test_alpha_executes_and_returns_score(self):
        wing = AlphaWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)
        self.assertEqual(output.wing_version, "1.0.0")

    def test_beta_executes_and_returns_score(self):
        wing = BetaWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)

    def test_gamma_executes_and_returns_score(self):
        wing = GammaWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)

    def test_delta_executes_and_returns_score(self):
        wing = DeltaWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)

    def test_epsilon_executes_and_returns_score(self):
        wing = EpsilonWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)

    def test_zeta_executes_and_returns_score(self):
        wing = ZetaWing()
        packet = LatentPacket(
            schema=self.ctx.latent_schema,
            data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
        )
        output = wing.execute(self.ctx, packet)
        self.assertIsInstance(output.latent.data, (float, np.floating, np.ndarray))
        self.assertGreaterEqual(output.local_score, 0.0)
        self.assertLessEqual(output.local_score, 1.0)

    def test_manifest_matches_model(self):
        wing = AlphaWing()
        self.assertEqual(
            wing.input_types().entries,
            load_manifest("wings/alpha/manifest.yaml").input_types.entries,
        )

    def test_seeded_registry_loads_all_six_wings(self):
        registry = InMemoryRegistry(core_epoch="1.x", seed_dir="wings")
        self.assertEqual(len(registry.installed()), 6)

    def test_all_wings_have_valid_artifact_hash(self):
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
            manifest = load_manifest(f"wings/{name}/manifest.yaml")
            self.assertNotEqual(manifest.integrity.artifact_hash, "stub-unchecked-in-s1")
            self.assertEqual(len(manifest.integrity.artifact_hash), 64)  # sha256 hex

    def test_wings_have_real_provenance(self):
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
            manifest = load_manifest(f"wings/{name}/manifest.yaml")
            self.assertNotEqual(manifest.provenance.training_dataset_ids[0], "stub")
            self.assertTrue(manifest.provenance.training_dataset_ids[0].startswith("umna-s3-"))

    def test_core_epoch_consistency(self):
        for name in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
            manifest = load_manifest(f"wings/{name}/manifest.yaml")
            self.assertEqual(manifest.core_epoch, "1.x")


if __name__ == "__main__":
    unittest.main()
