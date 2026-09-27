"""The parameter write firewall (S3, Section 6 of the paper).

A Wing trains downstream of the frozen Core, so backprop *traverses* the
Core to reach the Wing's own parameters. The Core parameter gradient is
genuinely non-zero. What is forbidden is an *update*: the trainer must
never see a writable reference to a Core parameter, so any accidental
assignment raises instead of corrupting the epoch.

Old Wing parameters are never handed to this trainer at all; write
isolation, not gradient isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import (
    EPOCH,
    FrozenCore,
    backward,
    forward,
    params_sha256,
)
from .tasks import INPUT_DIM

WING_WIDTH = 16
WING_SEED = 0
WING_STEPS = 600
WING_BATCH = 64
WING_LR = 0.05
WING_WEIGHT_DECAY = 1e-4
WING_SIGMOID_CLIP = 30.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -WING_SIGMOID_CLIP, WING_SIGMOID_CLIP)))


def _init_wing_params(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w": (
            rng.uniform(-1.0, 1.0, size=(INPUT_DIM, WING_WIDTH)).astype(np.float32)
            * 0.5
        ),
        "b": np.zeros(WING_WIDTH, dtype=np.float32),
    }


def _init_head_params(
    rng: np.random.Generator, latent_dim: int = 8
) -> dict[str, np.ndarray]:
    return {
        "w": (
            rng.uniform(-1.0, 1.0, size=(latent_dim, 1)).astype(np.float32)
            * 0.25
        ),
        "b": np.zeros(1, dtype=np.float32),
    }


def _predict(
    wing_params: dict[str, np.ndarray],
    head_params: dict[str, np.ndarray],
    core: FrozenCore,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward path x -> Wing transform -> Core trunk -> head logit."""

    pre = x @ wing_params["w"] + wing_params["b"]
    features = np.maximum(pre, 0.0)
    latent, _ = core.forward(features)
    logits = latent @ head_params["w"] + head_params["b"]
    return logits, pre


@dataclass(frozen=True)
class WingArtifact:
    """A trained Wing: its own parameters plus the Core it was built for."""

    name: str
    params: dict[str, dict[str, np.ndarray]]
    core_epoch: str
    core_sha256: str

    @property
    def transform(self) -> dict[str, np.ndarray]:
        return self.params["transform"]

    @property
    def head(self) -> dict[str, np.ndarray]:
        return self.params["head"]

    @property
    def sha256(self) -> str:
        return params_sha256(
            {
                f"transform.{k}": v for k, v in self.transform.items()
            }
            | {f"head.{k}": v for k, v in self.head.items()}
        )


def train_wing_through_core(
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = WING_SEED,
    steps: int = WING_STEPS,
    batch_size: int = WING_BATCH,
    learning_rate: float = WING_LR,
    weight_decay: float = WING_WEIGHT_DECAY,
) -> WingArtifact:
    """Train one Wing through a frozen Core and prove the Core did not move.

    The Core parameters are read-only mounts: the optimizer only ever
    receives and writes the Wing-owned transform and head. Gradients flow
    through the Core (Section 6); Core parameter updates would raise
    ValueError because the arrays are immutable.
    """

    if core.epoch != EPOCH:
        raise ValueError(f"core epoch {core.epoch!r} is not {EPOCH!r}")
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError("x must be a 2-D array of floats")
    if y.shape != (len(x),):
        raise ValueError(f"y shape {y.shape!r} does not match x rows {len(x)}")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")

    core_before = core.sha256
    rng = np.random.default_rng(seed)
    wing = _init_wing_params(rng)
    head = _init_head_params(rng)

    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32).reshape(-1, 1)

    for _ in range(steps):
        rows = rng.integers(0, len(x), size=batch_size)
        batch_x = x[rows]
        batch_y = y[rows]
        logits, pre = _predict(wing, head, core, batch_x)
        grad_logits = (_sigmoid(logits) - batch_y) / batch_size
        grad_head_w = logits.T @ grad_logits  # placeholder, corrected below
        latent, cache = core.forward(np.maximum(batch_x @ wing["w"] + wing["b"], 0.0))
        grad_logits = (_sigmoid(latent @ head["w"] + head["b"]) - batch_y) / batch_size
        grad_head_w = latent.T @ grad_logits + weight_decay * head["w"]
        grad_head_b = grad_logits.sum(axis=0)
        grad_latent = grad_logits @ head["w"].T
        _, grad_features = backward(
            core.params, np.maximum(pre, 0.0), cache, grad_latent
        )
        grad_pre = grad_features * (pre > 0.0)
        grad_w = batch_x.T @ grad_pre + weight_decay * wing["w"]
        grad_b = grad_pre.sum(axis=0)
        wing["w"] -= learning_rate * grad_w
        wing["b"] -= learning_rate * grad_b
        head["w"] -= learning_rate * grad_head_w
        head["b"] -= learning_rate * grad_head_b

    if core.sha256 != core_before:
        raise RuntimeError("core hash changed during wing training")
    return WingArtifact(
        name=name,
        params={"transform": wing, "head": head},
        core_epoch=core.epoch,
        core_sha256=core.sha256,
    )


def assert_wing_through_core_isolates_writes(core: FrozenCore) -> None:
    """Prove the read-only mount rejects writes to Core parameters.

    This is the construction-enforced guard (docs/06 failsafe table):
    any in-place update of a Core parameter must raise rather than
    silently mutate the epoch.
    """

    for name in core.params:
        if core.params[name].flags.writeable:
            raise AssertionError(f"core parameter {name!r} is writable")
    try:
        core.params["w0"][0, 0] = 0.0
    except (ValueError, RuntimeError):
        return
    raise AssertionError("write to frozen core parameter was accepted")


if __name__ == "__main__":
    from .train import train_core

    core = train_core(steps=60, batch_size=64)
    print(f"frozen core {core.epoch}: {core.sha256[:16]}")
    assert_wing_through_core_isolates_writes(core)
    print("write firewall: unauthorised write raises as expected")
