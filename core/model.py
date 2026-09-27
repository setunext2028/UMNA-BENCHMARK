"""The Core: a small numpy trunk, its frozen artifact, and its canonical hash.

One Core epoch, one closed compatibility domain. The trunk maps Wing
features to a shared latent; per-task probe heads live at training time
only. After freeze the parameters are mounted read-only and never change:
the sha256 is over canonical bytes derived from the values the optimizer
saw.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tasks import HONEST_TASKS

EPOCH = "1.x"
FEATURE_DIM = 16
LATENT_DIM = 8
WIDTHS = (FEATURE_DIM, 12, LATENT_DIM)
SCALE = 0.5


def init_params(widths: tuple[int, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """He initialised trunk parameters as float32 arrays."""

    params = {}
    for index, (in_dim, out_dim) in enumerate(zip(widths, widths[1:])):
        params[f"w{index}"] = (
            rng.uniform(-1.0, 1.0, size=(in_dim, out_dim)).astype(np.float32)
            * SCALE
        )
        params[f"b{index}"] = np.zeros(out_dim, dtype=np.float32)
    return params


def forward(
    params: Mapping[str, np.ndarray],
    x: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trunk forward pass. Returns the latent and the activation cache.

    The cache holds pre-activation values so a downstream trainer can
    backpropagate through the frozen trunk. Exposing activations is a
    read; parameter writes are what the firewall blocks.
    """

    cache = {}
    hidden = x
    for index, name in enumerate(("w0", "w1")):
        pre = hidden @ params[name] + params[f"b{index}"]
        cache[name] = pre
        hidden = pre if index == 1 else np.maximum(pre, 0.0)
    return hidden, cache


def backward(
    params: Mapping[str, np.ndarray],
    x: np.ndarray,
    cache: Mapping[str, np.ndarray],
    grad_out: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Gradients of every trunk parameter for the given output gradient.

    The gradient of the frozen Core parameters is genuinely non-zero
    (Section 6 of the paper): backpropagation must traverse the trunk.
    Computing it here is honest; *applying* it is what the firewall
    forbids. Callers apply only the parameter set they own.
    """

    grads = {}
    hidden = np.maximum(cache["w0"], 0.0)
    grads["w1"] = hidden.T @ grad_out
    grads["b1"] = grad_out.sum(axis=0)
    grad = grad_out @ params["w1"].T * (cache["w0"] > 0.0)
    grads["w0"] = x.T @ grad
    grads["b0"] = grad.sum(axis=0)
    return grads, grad @ params["w0"].T


def _canonical_bytes(params: Mapping[str, np.ndarray]) -> bytes:
    payload = []
    for key in sorted(params):
        array = np.asarray(params[key])
        payload.append(key.encode("utf-8"))
        payload.append(b"\x00")
        payload.append(
            np.asarray(array, dtype=np.float64, order="C").tobytes()
        )
        payload.append(b"\x00")
    return b"".join(payload)


def params_sha256(params: Mapping[str, np.ndarray]) -> str:
    """Deterministic content hash; byte-identical values hash byte-identical."""

    return hashlib.sha256(_canonical_bytes(params)).hexdigest()


@dataclass(frozen=True)
class FrozenCore:
    """The trained, frozen, hashed Core artifact for one epoch."""

    epoch: str
    params: dict[str, np.ndarray]
    meta: dict
    sha256: str

    def __post_init__(self):
        if self.epoch != EPOCH:
            raise ValueError(f"core epoch {self.epoch!r} is not {EPOCH!r}")
        if tuple(sorted(self.params)) != tuple(sorted(("b0", "b1", "w0", "w1"))):
            raise ValueError("core parameters must be exactly w0, b0, w1, b1")
        actual = params_sha256(self.params)
        if actual != self.sha256:
            raise ValueError(
                f"core hash mismatch: {actual[:12]} != {self.sha256[:12]}"
            )
        for name in self.params:
            array = self.params[name]
            array.setflags(write=False)
        self.meta.setdefault("kind", "core")
        self.meta.setdefault("epoch", self.epoch)
        self.meta.setdefault("latent_dim", LATENT_DIM)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run the frozen trunk. The returned cache is fresh per call."""

        return forward(self.params, x)

    def at(self, name: str) -> np.ndarray:
        return self.params[name]


def artifact_bytes(meta: Mapping) -> bytes:
    return json.dumps(
        meta, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def save_artifact(path: Path, core: FrozenCore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: core.params[name] for name in core.params},
        meta=artifact_bytes(core.meta),
        sha256=core.sha256,
    )


def load_artifact(path: Path) -> FrozenCore:
    """Load an artifact and verify the embedded hash before returning it."""

    with np.load(path, allow_pickle=False) as archive:
        params = {name: archive[name].copy() for name in ("w0", "b0", "w1", "b1")}
        meta = json.loads(bytes(archive["meta"].tolist()).decode("utf-8"))
        sha256 = str(archive["sha256"])
    return FrozenCore(epoch=meta["epoch"], params=params, meta=meta, sha256=sha256)


def artifact_path(epoch: str = EPOCH) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / f"core-{epoch}-{LATENT_DIM}d.npz"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    trunk = init_params(WIDTHS, rng)
    x = rng.uniform(-1.0, 1.0, size=(4, FEATURE_DIM))
    latent, cache = forward(trunk, x)
    print(f"forward ok: latent.shape={latent.shape} cache={tuple(cache)}")
    print(f"sha256: {params_sha256(trunk)[:16]}...")
    print(f"tasks visible to core training: {HONEST_TASKS}")
