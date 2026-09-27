"""Bridge fusion operators for composing Wing outputs.

A Bridge is a trained fusion module that takes multiple Wing outputs and
produces a combined latent representation. Bridges are first-class
capability modules with manifests, versions, and calibration bindings.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    WingRef,
)
from runtime.registry import load_manifest, InMemoryRegistry, Registry


# Generic typed fusion: simple concatenation + projection
# Specialized Bridges can be trained for high-value interactions


class Bridge(Protocol):
    """Bridge capability module contract."""
    
    def manifest(self) -> Manifest: ...
    def input_types(self) -> TypeSet: ...
    def output_types(self) -> TypeSet: ...
    def execute(self, ctx: CoreContext, inputs: tuple[LatentPacket, ...]) -> WingOutput: ...


@dataclass(frozen=True)
class BridgeManifest:
    """Manifest for a Bridge fusion operator."""
    name: str
    version: str
    core_epoch: str
    author: str
    input_types: TypeSet  # Multiple input types (one per source Wing)
    output_types: TypeSet
    capabilities: tuple[str, ...] = ()
    dependencies: tuple = ()
    resource_ceilings = None
    integrity = None
    provenance = None


class GenericBridge:
    """Generic typed fusion bridge: concatenates inputs and projects to latent.
    
    This is the O(1) fallback for common composition classes.
    Specialized trained Bridges are created for high-value validated interactions.
    """
    
    def __init__(self, input_dims: list[int], output_dim: int = 8, rng: np.random.Generator | None = None):
        self._input_dims = input_dims
        self._output_dim = output_dim
        self._total_in = sum(input_dims)
        
        if rng is None:
            rng = np.random.default_rng(0)
        
        # Projection: concat -> output_dim
        self._proj_w = (rng.uniform(-1.0, 1.0, size=(self._total_in, output_dim)).astype(np.float32) * 0.5)
        self._proj_b = np.zeros(output_dim, dtype=np.float32)
    
    def manifest(self) -> Manifest:
        return Manifest(
            name=f"generic_bridge_{len(self._input_dims)}x{self._input_dims[0]}",
            version="1.0.0",
            core_epoch="1.x",
            author="system",
            input_types=TypeSet(tuple(TypeExpr("Latent", (f"Vec{d}",)) for d in self._input_dims)),
            output_types=TypeSet((TypeExpr("Latent", (f"Vec{self._output_dim}",)),)),
            capabilities=("generic-fusion",),
        )
    
    def input_types(self) -> TypeSet:
        return self.manifest().input_types
    
    def output_types(self) -> TypeSet:
        return self.manifest().output_types
    
    def execute(self, ctx: CoreContext, inputs: tuple[LatentPacket, ...]) -> WingOutput:
        # Concatenate all inputs
        data_list = []
        for inp in inputs:
            d = inp.data
            if isinstance(d, np.ndarray):
                arr = d.astype(np.float32, copy=False)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            else:
                arr = np.array([d], dtype=np.float32).reshape(1, -1)
            data_list.append(arr)
        
        concat = np.concatenate(data_list, axis=1)
        
        # Project
        out = concat @ self._proj_w + self._proj_b
        
        schema = LatentSchema(ctx.core_epoch, self._output_dim, "float32")
        latent = LatentPacket(schema, out, provenance=f"bridge/generic")
        
        evidence = EvidencePacket(
            claim_type="fusion",
            claim_value="generic_concat_project",
            polarity=Polarity.SUPPORT,
            source_wing=self.manifest().name,
            source_version=self.manifest().version,
            route_id=ctx.binding_id or "direct",
            provenance="bridge/generic",
            local_score=1.0,
            calibration_binding=ctx.binding_id or "direct",
            validity_domain="fusion:generic",
            timestamp="now",
        )
        
        return WingOutput(
            latent=latent,
            schema=schema,
            evidence=(evidence,),
            local_score=1.0,
            wing_version=self.manifest().version,
            provenance="bridge/generic",
        )


class TrainedBridge:
    """A trained Bridge with learned fusion weights.
    
    Trained on specific Wing pairs for high-value interactions.
    Has its own manifest, artifact, and calibration.
    """
    
    def __init__(
        self,
        name: str,
        input_wings: list[str],  # Names of source wings
        proj_w: np.ndarray,
        proj_b: np.ndarray,
        core_epoch: str = "1.x",
    ):
        self._name = name
        self._input_wings = input_wings
        self._proj_w = proj_w.astype(np.float32, copy=False)
        self._proj_b = proj_b.astype(np.float32, copy=False)
        self._core_epoch = core_epoch
        self._total_in = proj_w.shape[0]
        self._output_dim = proj_w.shape[1]
        
        self._manifest = Manifest(
            name=name,
            version="1.0.0",
            core_epoch=core_epoch,
            author="bridge_trainer",
            input_types=TypeSet(tuple(TypeExpr("Latent", ("Vec",)) for _ in input_wings)),
            output_types=TypeSet((TypeExpr("Latent", (f"Vec{self._output_dim}",)),)),
            capabilities=("trained-fusion",),
        )
    
    def manifest(self) -> Manifest:
        return self._manifest
    
    def input_types(self) -> TypeSet:
        return self._manifest.input_types
    
    def output_types(self) -> TypeSet:
        return self._manifest.output_types
    
    def execute(self, ctx: CoreContext, inputs: tuple[LatentPacket, ...]) -> WingOutput:
        if len(inputs) != len(self._input_wings):
            raise ValueError(f"Expected {len(self._input_wings)} inputs, got {len(inputs)}")
        
        # Concatenate all inputs
        data_list = []
        for inp in inputs:
            d = inp.data
            if isinstance(d, np.ndarray):
                arr = d.astype(np.float32, copy=False)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            else:
                arr = np.array([d], dtype=np.float32).reshape(1, -1)
            data_list.append(arr)
        
        concat = np.concatenate(data_list, axis=1)
        
        # Project
        out = concat @ self._proj_w + self._proj_b
        
        schema = LatentSchema(ctx.core_epoch, self._output_dim, "float32")
        latent = LatentPacket(schema, out, provenance=f"bridge/{self._name}")
        
        evidence = EvidencePacket(
            claim_type="fusion",
            claim_value=f"trained_{self._name}",
            polarity=Polarity.SUPPORT,
            source_wing=self._name,
            source_version=self._manifest.version,
            route_id=ctx.binding_id or "direct",
            provenance=f"bridge/{self._name}",
            local_score=1.0,
            calibration_binding=ctx.binding_id or "direct",
            validity_domain=f"fusion:{self._name}",
            timestamp="now",
        )
        
        return WingOutput(
            latent=latent,
            schema=schema,
            evidence=(evidence,),
            local_score=1.0,
            wing_version=self._manifest.version,
            provenance=f"bridge/{self._name}",
        )


def train_bridge(
    bridge_name: str,
    input_wings: list[str],
    training_data: list[tuple[tuple[np.ndarray, ...], np.ndarray]],
    output_dim: int = 8,
    epochs: int = 100,
    lr: float = 0.01,
    rng: np.random.Generator | None = None,
) -> TrainedBridge:
    """Train a Bridge on paired Wing outputs -> target latent.
    
    Args:
        bridge_name: Name for the bridge
        input_wings: Names of source wings
        training_data: List of ((wing1_out, wing2_out, ...), target_latent)
        output_dim: Output latent dimension
        epochs: Training epochs
        lr: Learning rate
    
    Returns:
        TrainedBridge with learned projection
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    n_inputs = len(input_wings)
    # Get input dims from first sample
    sample_inputs, _ = training_data[0]
    input_dims = [inp.shape[-1] for inp in sample_inputs]
    total_in = sum(input_dims)
    
    # Initialize projection
    proj_w = (rng.uniform(-1.0, 1.0, size=(total_in, output_dim)).astype(np.float32) * 0.5)
    proj_b = np.zeros(output_dim, dtype=np.float32)
    
    # Ensure all inputs are 2D for concatenation
    def to_2d(arr):
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr
    
    # Concatenate each sample, then stack
    X_list = []
    for inps, _ in training_data:
        concat = np.concatenate([to_2d(inp) for inp in inps], axis=1)
        X_list.append(concat)
    X = np.vstack(X_list)  # Shape: (n_samples, total_in)
    y = np.vstack([t.reshape(1, -1) if t.ndim == 1 else t for _, t in training_data])
    
    for _ in range(epochs):
        # Forward
        preds = X @ proj_w + proj_b
        error = preds - y
        
        # Gradients
        grad_w = X.T @ error / len(X)
        grad_b = error.mean(axis=0)
        
        proj_w -= lr * grad_w
        proj_b -= lr * grad_b
    
    return TrainedBridge(bridge_name, input_wings, proj_w, proj_b)


class BridgeRegistry:
    """Registry for Bridge fusion operators."""
    
    def __init__(self):
        self._bridges: dict[str, Bridge] = {}
        self._generic_cache: dict[tuple[int, ...], GenericBridge] = {}
    
    def register(self, bridge: Bridge) -> str:
        name = bridge.manifest().name
        if name in self._bridges:
            raise ValueError(f"Bridge {name} already registered")
        self._bridges[name] = bridge
        return name
    
    def get(self, name: str) -> Bridge | None:
        return self._bridges.get(name)
    
    def get_or_create_generic(self, input_dims: tuple[int, ...]) -> GenericBridge:
        """Get or create a generic fusion bridge for the given input dimensions."""
        if input_dims not in self._generic_cache:
            self._generic_cache[input_dims] = GenericBridge(list(input_dims))
        return self._generic_cache[input_dims]
    
    def list_bridges(self) -> list[str]:
        return list(self._bridges.keys())


# Global default instance
_DEFAULT_BRIDGE_REGISTRY: BridgeRegistry | None = None


def get_bridge_registry() -> BridgeRegistry:
    global _DEFAULT_BRIDGE_REGISTRY
    if _DEFAULT_BRIDGE_REGISTRY is None:
        _DEFAULT_BRIDGE_REGISTRY = BridgeRegistry()
    return _DEFAULT_BRIDGE_REGISTRY


def fuse_with_bridge(
    bridge: Bridge,
    ctx: CoreContext,
    inputs: tuple[LatentPacket, ...],
) -> WingOutput:
    """Execute a Bridge fusion on the given inputs."""
    return bridge.execute(ctx, inputs)


if __name__ == "__main__":
    # Quick test
    rng = np.random.default_rng(0)
    
    # Generic bridge
    gb = GenericBridge([8, 8], 8)
    print(f"Generic bridge: {gb.manifest().name}")
    print(f"Input types: {gb.input_types()}")
    print(f"Output types: {gb.output_types()}")
    
    # Test execute
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
    inp1 = LatentPacket(LatentSchema("1.x", 8, "float32"), rng.random(8).astype(np.float32))
    inp2 = LatentPacket(LatentSchema("1.x", 8, "float32"), rng.random(8).astype(np.float32))
    out = gb.execute(ctx, (inp1, inp2))
    print(f"Output: {out.latent.data.shape}")
    
    # Train a specialized bridge
    training = [
        ((rng.random(8).astype(np.float32), rng.random(8).astype(np.float32)), rng.random(8).astype(np.float32))
        for _ in range(50)
    ]
    tb = train_bridge("test_bridge", ["wing_a", "wing_b"], training, output_dim=8, epochs=20)
    print(f"Trained bridge: {tb.manifest().name}")
    out2 = tb.execute(ctx, (inp1, inp2))
    print(f"Output: {out2.latent.data.shape}")
    
    # Registry
    reg = BridgeRegistry()
    reg.register(tb)
    print(f"Registry: {reg.list_bridges()}")