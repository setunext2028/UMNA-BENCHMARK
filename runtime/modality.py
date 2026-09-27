"""Second modality support: Image modality with encoder to Core latent space.

This implements Phase 5: Multimodal extension inside one Core epoch.
The Core remains frozen; new modality Wings map their inputs to the
existing Latent<Vec8> interface.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime.abi import (
    CoreContext,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Manifest,
    Polarity,
    TypeExpr,
    TypeSet,
    WingOutput,
)
from runtime.registry import load_manifest


# Image modality constants
IMAGE_MODALITY = "Image"
IMAGE_LATENT_DIM = 64  # Raw image feature dimension
CORE_LATENT_DIM = 8    # Core expects Vec8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class ImageEncoderWing:
    """Encoder wing: maps raw image features to Core Latent<Vec8>.
    
    This wing enables the image modality by encoding images to the
    Core's expected latent interface. The Core itself remains frozen.
    """
    
    def __init__(self, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        
        # Encoder: IMAGE_LATENT_DIM -> 16 -> CORE_LATENT_DIM
        self._enc_w1 = (rng.uniform(-1.0, 1.0, size=(IMAGE_LATENT_DIM, 16)).astype(np.float32) * 0.5)
        self._enc_b1 = np.zeros(16, dtype=np.float32)
        self._enc_w2 = (rng.uniform(-1.0, 1.0, size=(16, CORE_LATENT_DIM)).astype(np.float32) * 0.5)
        self._enc_b2 = np.zeros(CORE_LATENT_DIM, dtype=np.float32)
        
        # Make read-only after init
        for arr in [self._enc_w1, self._enc_b1, self._enc_w2, self._enc_b2]:
            arr.setflags(write=False)
        
        self._manifest = Manifest(
            name="image_encoder",
            version="1.0.0",
            core_epoch="1.x",
            author="system",
            input_types=TypeSet((TypeExpr("Latent", (IMAGE_MODALITY,)),)),
            output_types=TypeSet((TypeExpr("Latent", ("Vec8",)),)),
            capabilities=("modality-encoder", "image"),
        )
    
    def manifest(self) -> Manifest:
        return self._manifest
    
    def input_types(self) -> TypeSet:
        return self._manifest.input_types
    
    def output_types(self) -> TypeSet:
        return self._manifest.output_types
    
    def execute(self, ctx: CoreContext, x: LatentPacket) -> WingOutput:
        # Expect Latent<Image> input
        data = x.data
        if isinstance(data, np.ndarray):
            batch = data.astype(np.float32, copy=False)
        else:
            batch = np.array(data, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        
        # Encode: image features -> 16 (ReLU) -> 8 (linear)
        hidden = np.maximum(batch @ self._enc_w1 + self._enc_b1, 0.0)
        latent = hidden @ self._enc_w2 + self._enc_b2
        
        # Output Latent<Vec8> for Core
        out_schema = LatentSchema(ctx.core_epoch, CORE_LATENT_DIM, "float32")
        # Preserve batch dimension: (batch_size, 8) or (8,) for single
        if latent.shape[0] == 1:
            out_data = latent[0]  # Shape (8,)
        else:
            out_data = latent  # Shape (batch_size, 8)
        latent_packet = LatentPacket(out_schema, out_data, provenance="modality/image_encoder")
        
        evidence = EvidencePacket(
            claim_type="modality_encoding",
            claim_value="image_to_vec8",
            polarity=Polarity.SUPPORT,
            source_wing="image_encoder",
            source_version=self._manifest.version,
            route_id=ctx.binding_id or "direct",
            provenance="modality/image_encoder",
            local_score=1.0,
            calibration_binding=ctx.binding_id or "direct",
            validity_domain="modality:image",
            timestamp="now",
        )
        
        return WingOutput(
            latent=latent_packet,
            schema=out_schema,
            evidence=(evidence,),
            local_score=1.0,
            wing_version=self._manifest.version,
            provenance="modality/image_encoder",
        )


class ImageClassifierWing:
    """Classifier wing: operates on Latent<Vec8> for image tasks.
    
    This wing sits downstream of the image_encoder and performs
    classification on the encoded latent. It uses the frozen Core
    as a feature processor.
    """
    
    def __init__(self, task_name: str, rng: np.random.Generator | None = None):
        if rng is None:
            rng = np.random.default_rng(0)
        
        # Classifier head on Core latent: 8 -> 1
        self._head_w = (rng.uniform(-1.0, 1.0, size=(CORE_LATENT_DIM, 1)).astype(np.float32) * 0.25)
        self._head_b = np.zeros(1, dtype=np.float32)
        
        for arr in [self._head_w, self._head_b]:
            arr.setflags(write=False)
        
        self._task_name = task_name
        self._manifest = Manifest(
            name=f"image_classifier_{task_name}",
            version="1.0.0",
            core_epoch="1.x",
            author="system",
            input_types=TypeSet((TypeExpr("Latent", ("Vec8",)),)),
            output_types=TypeSet((TypeExpr("Latent", ("Score",)),)),
            capabilities=(f"image-classifier-{task_name}",),
        )
    
    def manifest(self) -> Manifest:
        return self._manifest
    
    def input_types(self) -> TypeSet:
        return self._manifest.input_types
    
    def output_types(self) -> TypeSet:
        return self._manifest.output_types
    
    def execute(self, ctx: CoreContext, x: LatentPacket) -> WingOutput:
        data = x.data
        if isinstance(data, np.ndarray):
            batch = data.astype(np.float32, copy=False)
        else:
            batch = np.array(data, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        
        # Core forward (simulated - in practice would use Core trunk)
        # For now, identity pass through Core latent
        latent = batch
        
        # Classifier head
        logits = latent @ self._head_w + self._head_b
        probs = _sigmoid(logits)
        predictions = (probs >= 0.5).astype(np.float32)
        
        out_schema = LatentSchema(ctx.core_epoch, 1, "float32")
        # Preserve batch dimension: (batch_size, 1) or (1,) for single
        if predictions.shape[0] == 1:
            out_data = predictions[0]  # Shape (1,)
        else:
            out_data = predictions  # Shape (batch_size, 1)
        latent_packet = LatentPacket(out_schema, out_data, provenance=f"modality/image_classifier_{self._task_name}")
        
        evidence = EvidencePacket(
            claim_type="class",
            claim_value=str(int(predictions[0, 0])),
            polarity=Polarity.SUPPORT,
            source_wing=f"image_classifier_{self._task_name}",
            source_version=self._manifest.version,
            route_id=ctx.binding_id or "direct",
            provenance=f"modality/image_classifier_{self._task_name}",
            local_score=float(probs[0, 0]),
            calibration_binding=ctx.binding_id or "direct",
            validity_domain=f"modality:image, task:{self._task_name}",
            timestamp="now",
        )
        
        return WingOutput(
            latent=latent_packet,
            schema=out_schema,
            evidence=(evidence,),
            local_score=float(probs[0, 0]) if probs.ndim > 1 else float(probs),
            wing_version=self._manifest.version,
            provenance=f"modality/image_classifier_{self._task_name}",
        )


# Modality registry for managing multiple modalities
@dataclass
class ModalityInfo:
    """Information about a registered modality."""
    name: str
    raw_dim: int
    encoder_wing_name: str
    supported_tasks: tuple[str, ...]


class ModalityRegistry:
    """Registry of supported modalities and their encoders."""
    
    def __init__(self):
        self._modalities: dict[str, ModalityInfo] = {}
        self._encoders: dict[str, ImageEncoderWing] = {}
        self._classifiers: dict[str, ImageClassifierWing] = {}
        
        # Register default image modality
        encoder = ImageEncoderWing()
        self.register_modality(
            name=IMAGE_MODALITY,
            raw_dim=IMAGE_LATENT_DIM,
            encoder=encoder,
            supported_tasks=("classification", "detection"),
        )
    
    def register_modality(
        self,
        name: str,
        raw_dim: int,
        encoder: ImageEncoderWing,
        supported_tasks: tuple[str, ...],
    ) -> None:
        if name in self._modalities:
            raise ValueError(f"Modality {name} already registered")
        self._modalities[name] = ModalityInfo(
            name=name,
            raw_dim=raw_dim,
            encoder_wing_name=encoder.manifest().name,
            supported_tasks=supported_tasks,
        )
        self._encoders[name] = encoder
    
    def get_modality(self, name: str) -> ModalityInfo | None:
        return self._modalities.get(name)
    
    def get_encoder(self, name: str) -> ImageEncoderWing | None:
        return self._encoders.get(name)
    
    def register_classifier(self, task_name: str, classifier: ImageClassifierWing) -> None:
        self._classifiers[task_name] = classifier
    
    def get_classifier(self, task_name: str) -> ImageClassifierWing | None:
        return self._classifiers.get(task_name)
    
    def list_modalities(self) -> list[str]:
        return list(self._modalities.keys())


# Global default instance
_DEFAULT_MODALITY_REGISTRY: ModalityRegistry | None = None


def get_modality_registry() -> ModalityRegistry:
    global _DEFAULT_MODALITY_REGISTRY
    if _DEFAULT_MODALITY_REGISTRY is None:
        _DEFAULT_MODALITY_REGISTRY = ModalityRegistry()
    return _DEFAULT_MODALITY_REGISTRY


if __name__ == "__main__":
    # Quick test
    rng = np.random.default_rng(0)
    
    # Test image encoder
    encoder = ImageEncoderWing(rng)
    print(f"Encoder manifest: {encoder.manifest().name}")
    print(f"Input types: {encoder.input_types()}")
    print(f"Output types: {encoder.output_types()}")
    
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", IMAGE_LATENT_DIM, "float32"))
    img = LatentPacket(LatentSchema("1.x", IMAGE_LATENT_DIM, "float32"), rng.random(IMAGE_LATENT_DIM).astype(np.float32))
    out = encoder.execute(ctx, img)
    print(f"Encoder output: {out.latent.data.shape}, score={out.local_score:.3f}")
    
    # Test image classifier
    classifier = ImageClassifierWing("test", rng)
    print(f"Classifier manifest: {classifier.manifest().name}")
    
    ctx2 = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", CORE_LATENT_DIM, "float32"))
    vec8 = LatentPacket(LatentSchema("1.x", CORE_LATENT_DIM, "float32"), rng.random(CORE_LATENT_DIM).astype(np.float32))
    out2 = classifier.execute(ctx2, vec8)
    print(f"Classifier output: {out2.latent.data}, score={out2.local_score:.3f}")
    
    # Test modality registry
    mod_reg = get_modality_registry()
    print(f"Modalities: {mod_reg.list_modalities()}")
    img_mod = mod_reg.get_modality("Image")
    print(f"Image modality: {img_mod}")