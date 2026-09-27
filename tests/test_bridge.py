"""Tests for Bridge fusion operators."""

import unittest

import numpy as np

from runtime.abi import CoreContext, LatentPacket, LatentSchema, TypeExpr, TypeSet
from runtime.bridge import (
    Bridge,
    GenericBridge,
    TrainedBridge,
    BridgeRegistry,
    train_bridge,
    get_bridge_registry,
    fuse_with_bridge,
)


class TestGenericBridge(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.bridge = GenericBridge([8, 8], 8, rng=self.rng)
        self.ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    def test_manifest(self):
        manifest = self.bridge.manifest()
        self.assertIn("generic_bridge", manifest.name)
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.core_epoch, "1.x")
        self.assertEqual(len(manifest.input_types), 2)
        self.assertEqual(len(manifest.output_types), 1)

    def test_input_output_types(self):
        inputs = self.bridge.input_types()
        outputs = self.bridge.output_types()
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)

    def test_execute_concat_project(self):
        inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random(8).astype(np.float32))
        inp2 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random(8).astype(np.float32))
        
        output = self.bridge.execute(self.ctx, (inp1, inp2))
        
        self.assertIsInstance(output.latent.data, np.ndarray)
        self.assertEqual(output.latent.data.shape[-1], 8)
        self.assertEqual(output.local_score, 1.0)
        self.assertEqual(len(output.evidence), 1)
        self.assertEqual(output.evidence[0].claim_type, "fusion")

    def test_execute_batch(self):
        # Batch inputs
        inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random((4, 8)).astype(np.float32))
        inp2 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random((4, 8)).astype(np.float32))
        
        output = self.bridge.execute(self.ctx, (inp1, inp2))
        
        self.assertEqual(output.latent.data.shape[0], 4)
        self.assertEqual(output.latent.data.shape[1], 8)


class TestTrainedBridge(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
        
        # Create simple training data
        self.training_data = [
            ((self.rng.random(8).astype(np.float32), self.rng.random(8).astype(np.float32)), 
             self.rng.random(8).astype(np.float32))
            for _ in range(50)
        ]
        
        self.bridge = train_bridge(
            "test_bridge",
            ["wing_a", "wing_b"],
            self.training_data,
            output_dim=8,
            epochs=20,
            rng=self.rng,
        )

    def test_manifest(self):
        manifest = self.bridge.manifest()
        self.assertEqual(manifest.name, "test_bridge")
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.core_epoch, "1.x")
        self.assertEqual(len(manifest.input_types), 2)

    def test_execute(self):
        inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random(8).astype(np.float32))
        inp2 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random(8).astype(np.float32))
        
        output = self.bridge.execute(self.ctx, (inp1, inp2))
        
        self.assertIsInstance(output.latent.data, np.ndarray)
        self.assertEqual(output.latent.data.shape[-1], 8)
        self.assertEqual(len(output.evidence), 1)
        self.assertEqual(output.evidence[0].claim_type, "fusion")

    def test_wrong_input_count(self):
        inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), self.rng.random(8).astype(np.float32))
        
        with self.assertRaises(ValueError):
            self.bridge.execute(self.ctx, (inp1,))


class TestBridgeRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = BridgeRegistry()
        self.rng = np.random.default_rng(42)
        self.ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    def test_register_and_get(self):
        bridge = GenericBridge([8, 8], 8, rng=self.rng)
        name = self.registry.register(bridge)
        
        self.assertEqual(name, bridge.manifest().name)
        self.assertIs(self.registry.get(name), bridge)

    def test_duplicate_registration_fails(self):
        bridge = GenericBridge([8, 8], 8, rng=self.rng)
        self.registry.register(bridge)
        
        with self.assertRaises(ValueError):
            self.registry.register(bridge)

    def test_get_or_create_generic(self):
        bridge1 = self.registry.get_or_create_generic((8, 8))
        bridge2 = self.registry.get_or_create_generic((8, 8))
        
        # Should return same instance
        self.assertIs(bridge1, bridge2)
        
        # Different dims should create new
        bridge3 = self.registry.get_or_create_generic((8, 16))
        self.assertIsNot(bridge1, bridge3)

    def test_list_bridges(self):
        self.assertEqual(self.registry.list_bridges(), [])
        
        bridge = GenericBridge([8, 8], 8, rng=self.rng)
        self.registry.register(bridge)
        
        self.assertIn(bridge.manifest().name, self.registry.list_bridges())


class TestFuseWithBridge(unittest.TestCase):
    def test_fuse_with_bridge(self):
        rng = np.random.default_rng(42)
        bridge = GenericBridge([8, 8], 8, rng=rng)
        ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
        
        inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), rng.random(8).astype(np.float32))
        inp2 = LatentPacket(LatentSchema("1.x", 8, "float32"), rng.random(8).astype(np.float32))
        
        output = fuse_with_bridge(bridge, ctx, (inp1, inp2))
        
        self.assertEqual(output.latent.data.shape[-1], 8)
        self.assertEqual(len(output.evidence), 1)


class TestGetBridgeRegistry(unittest.TestCase):
    def test_global_registry(self):
        reg1 = get_bridge_registry()
        reg2 = get_bridge_registry()
        self.assertIs(reg1, reg2)


if __name__ == "__main__":
    unittest.main()