"""Wing beta: trained redundant classifier for task t_ab (S3)."""

from __future__ import annotations

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

_MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")
_MANIFEST = load_manifest(_MANIFEST_PATH)
_PARAMS_PATH = Path(__file__).with_name("params.npz")
_CORE_ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "core" / "artifacts" / "core-1.x-8d.npz"

# Load trained wing parameters
_wing_params = np.load(_PARAMS_PATH, allow_pickle=False)
_TRANSFORM_W = _wing_params["transform.w"].astype(np.float32, copy=False)
_TRANSFORM_B = _wing_params["transform.b"].astype(np.float32, copy=False)
_HEAD_W = _wing_params["head.w"].astype(np.float32, copy=False)
_HEAD_B = _wing_params["head.b"].astype(np.float32, copy=False)
_CORE_EPOCH = str(_wing_params["core_epoch"])
_CORE_SHA256 = str(_wing_params["core_sha256"])
_WING_SHA256 = str(_wing_params["wing_sha256"])

# Load frozen Core trunk parameters
_core_archive = np.load(_CORE_ARTIFACT_PATH, allow_pickle=False)
_CORE_PARAMS = {
    "w0": _core_archive["w0"].copy(),
    "b0": _core_archive["b0"].copy(),
    "w1": _core_archive["w1"].copy(),
    "b1": _core_archive["b1"].copy(),
}
# Make Core parameters read-only (write firewall)
for arr in _CORE_PARAMS.values():
    arr.setflags(write=False)

# Verify core hash matches expected
import hashlib
_canonical = b""
for key in sorted(_CORE_PARAMS):
    arr = _CORE_PARAMS[key]
    _canonical += key.encode("utf-8") + b"\x00"
    _canonical += np.asarray(arr, dtype=np.float64, order="C").tobytes() + b"\x00"
_verified_core_hash = hashlib.sha256(_canonical).hexdigest()
assert _verified_core_hash == _CORE_SHA256, "Core hash mismatch"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass: x -> transform -> core -> head -> logit, prob."""
    # Wing transform: 8 -> 16 (ReLU)
    pre = x @ _TRANSFORM_W + _TRANSFORM_B
    features = np.maximum(pre, 0.0)
    # Core trunk: 16 -> 12 (ReLU) -> 8 (linear)
    hidden = features
    for name in ("w0", "w1"):
        pre_core = hidden @ _CORE_PARAMS[name] + _CORE_PARAMS[f"b{name[1]}"]
        hidden = pre_core if name == "w1" else np.maximum(pre_core, 0.0)
    latent = hidden
    # Head: 8 -> 1
    logits = latent @ _HEAD_W + _HEAD_B
    probs = _sigmoid(logits)
    return logits, probs


class BetaWing:
    """Trained Wing for task t_ab (redundant)."""

    def manifest(self) -> Manifest:
        return _MANIFEST

    def input_types(self) -> TypeSet:
        return _MANIFEST.input_types

    def output_types(self) -> TypeSet:
        return _MANIFEST.output_types

    def execute(self, ctx: CoreContext, x: LatentPacket) -> WingOutput:
        # Verify Core epoch matches
        if ctx.core_epoch != _CORE_EPOCH:
            raise ValueError(f"Core epoch mismatch: wing expects {_CORE_EPOCH}, got {ctx.core_epoch}")

        # Expect Latent<Vec8> input
        data = x.data
        if isinstance(data, np.ndarray):
            batch = data.astype(np.float32, copy=False)
        else:
            batch = np.array(data, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        logits, probs = _forward(batch)
        predictions = (probs >= 0.5).astype(np.float32)

        # Create output latent packet
        out_schema = LatentSchema(core_epoch=_CORE_EPOCH, dimensions=1, dtype="float32")
        out_data = predictions.reshape(-1) if predictions.shape[0] > 1 else predictions[0, 0]
        latent = LatentPacket(schema=out_schema, data=out_data, provenance="wing/beta")

        # Evidence packet
        evidence = EvidencePacket(
            claim_type="class",
            claim_value=str(int(predictions[0, 0])) if predictions.ndim > 1 else str(int(predictions)),
            polarity=Polarity.SUPPORT,
            source_wing="beta",
            source_version=_MANIFEST.version,
            route_id=ctx.binding_id or "direct",
            provenance="wing/beta",
            local_score=float(probs[0, 0]) if probs.ndim > 1 else float(probs),
            calibration_binding=ctx.binding_id or "direct",
            validity_domain="task:t_ab",
            timestamp="now",
        )

        return WingOutput(
            latent=latent,
            schema=out_schema,
            evidence=(evidence,),
            local_score=float(probs[0, 0]) if probs.ndim > 1 else float(probs),
            wing_version=_MANIFEST.version,
            provenance="wing/beta",
        )


if __name__ == "__main__":
    from runtime.abi import CoreContext, LatentPacket, LatentSchema
    wing = BetaWing()
    print(f"Manifest: {wing.manifest().name} v{wing.manifest().version}")
    print(f"Input types: {wing.input_types()}")
    print(f"Output types: {wing.output_types()}")

    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
    packet = LatentPacket(schema=ctx.latent_schema, data=np.array([0.5, -0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    output = wing.execute(ctx, packet)
    print(f"Output: {output.latent.data}, score={output.local_score:.3f}")
