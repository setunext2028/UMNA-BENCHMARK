"""Tests for second modality support."""

import unittest

import numpy as np

from runtime.abi import CoreContext, LatentPacket, LatentSchema, TypeExpr, TypeSet
from runtime.modality import (
    ImageEncoderWing,
    ImageClassifierWing,
    ModalityInfo,
    ModalityRegistry,
    get_modality_registry,
)


class TestImageEncoderWing(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.encoder = ImageEncoderWing(self.rng)
        self.ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 64, "float32"))

    def test_manifest(self):
        manifest = self.encoder.manifest()
        self.assertEqual(manifest.name, "image_encoder")
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.core_epoch, "1.x")
        
        inputs = manifest.input_types
        outputs = manifest.output_types
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        
        # Check input type is Latent<Image>
        self.assertEqual(inputs[0].family, "Latent")
        self.assertEqual(inputs[0].args, ("Image",))
        
        # Check output type is Latent<Vec8>
        self.assertEqual(outputs[0].family, "Latent")
        self.assertEqual(outputs[0].args, ("Vec8",))

    def test_input_output_types(self):
        inputs = self.encoder.input_types()
        outputs = self.encoder.output_types()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)

    def test_execute_single(self):
        img = LatentPacket(
            LatentSchema("1.x", 64, "float32"),
            self.rng.random(64).astype(np.float32)
        )
        output = self.encoder.execute(self.ctx, img)
        
        self.assertIsInstance(output.latent.data, np.ndarray)
        self.assertEqual(output.latent.data.shape[-1], 8)
        self.assertEqual(output.local_score, 1.0)
        self.assertEqual(len(output.evidence), 1)
        self.assertEqual(output.evidence[0].claim_type, "modality_encoding")
        self.assertEqual(output.evidence[0].claim_value, "image_to_vec8")

    def test_execute_batch(self):
        img = LatentPacket(
            LatentSchema("1.x", 64, "float32"),
            self.rng.random((4, 64)).astype(np.float32)
        )
        output = self.encoder.execute(self.ctx, img)
        
        self.assertEqual(output.latent.data.shape[0], 4)
        self.assertEqual(output.latent.data.shape[1], 8)


class TestImageClassifierWing(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.classifier = ImageClassifierWing("test", self.rng)
        self.ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    def test_manifest(self):
        manifest = self.classifier.manifest()
        self.assertEqual(manifest.name, "image_classifier_test")
        self.assertEqual(manifest.version, "1.0.0")
        self.assertEqual(manifest.core_epoch, "1.x")
        
        inputs = manifest.input_types
        outputs = manifest.output_types
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        
        # Input is Latent<Vec8>
        self.assertEqual(inputs[0].family, "Latent")
        self.assertEqual(inputs[0].args, ("Vec8",))
        
        # Output is Latent<Score>
        self.assertEqual(outputs[0].family, "Latent")
        self.assertEqual(outputs[0].args, ("Score",))

    def test_execute(self):
        vec8 = LatentPacket(
            LatentSchema("1.x", 8, "float32"),
            self.rng.random(8).astype(np.float32)
        )
        output = self.classifier.execute(self.ctx, vec8)
        
        self.assertIsInstance(output.latent.data, np.ndarray)
        self.assertEqual(output.latent.data.shape[-1], 1)
        self.assertEqual(len(output.evidence), 1)
        self.assertEqual(output.evidence[0].claim_type, "class")


class TestModalityRegistry(unittest.TestCase):
    def setUp(self):
        # Use fresh registry
        from runtime.modality import ModalityRegistry
        self.registry = ModalityRegistry()
        self.rng = np.random.default_rng(42)

    def test_default_image_modality(self):
        modalities = self.registry.list_modalities()
        self.assertIn("Image", modalities)
        
        img_mod = self.registry.get_modality("Image")
        self.assertIsNotNone(img_mod)
        self.assertEqual(img_mod.name, "Image")
        self.assertEqual(img_mod.raw_dim, 64)
        self.assertEqual(img_mod.encoder_wing_name, "image_encoder")

    def test_get_encoder(self):
        encoder = self.registry.get_encoder("Image")
        self.assertIsNotNone(encoder)
        self.assertIsInstance(encoder, ImageEncoderWing)

    def test_get_classifier(self):
        # No classifiers registered by default
        classifier = self.registry.get_classifier("test")
        self.assertIsNone(classifier)

    def test_register_classifier(self):
        classifier = ImageClassifierWing("test", self.rng)
        self.registry.register_classifier("test", classifier)
        
        retrieved = self.registry.get_classifier("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.manifest().name, "image_classifier_test")

    def test_list_modalities(self):
        modalities = self.registry.list_modalities()
        self.assertEqual(modalities, ["Image"])


class TestModalityIntegration(unittest.TestCase):
    """Test encoder + classifier pipeline."""
    
    def test_encoder_classifier_pipeline(self):
        rng = np.random.default_rng(42)
        encoder = ImageEncoderWing(rng)
        classifier = ImageClassifierWing("test", rng)
        
        ctx_enc = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 64, "float32"))
        img = LatentPacket(
            LatentSchema("1.x", 64, "float32"),
            rng.random(64).astype(np.float32)
        )
        
        # Encode image to Vec8
        enc_out = encoder.execute(ctx_enc, img)
        self.assertEqual(enc_out.latent.data.shape[-1], 8)
        
        # Classify encoded latent
        ctx_cls = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
        cls_out = classifier.execute(ctx_cls, enc_out.latent)
        
        self.assertEqual(cls_out.latent.data.shape[-1], 1)
        self.assertIn(cls_out.latent.data, [0.0, 1.0])


class TestGlobalModalityRegistry(unittest.TestCase):
    def test_global_registry(self):
        reg1 = get_modality_registry()
        reg2 = get_modality_registry()
        self.assertIs(reg1, reg2)


if __name__ == "__main__":
    unittest.main()