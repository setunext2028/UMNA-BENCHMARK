"""Train the epoch 1.x Core once, then freeze and hash it."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .model import EPOCH, FrozenCore, WIDTHS, backward, forward, init_params, params_sha256
from .tasks import HONEST_TASKS, INPUT_DIM, TaskSplit, load_tasks

CORE_SEED = 1729
DATA_SEED = 2718
CORE_STEPS = 1200
CORE_BATCH_SIZE = 128
CORE_LEARNING_RATE = 0.04
CORE_WEIGHT_DECAY = 1e-4


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _validate_training(
    tasks: Mapping[str, TaskSplit],
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    if set(tasks) != set(HONEST_TASKS):
        raise ValueError(f"core training requires exactly {HONEST_TASKS!r}")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    for task_id, split in tasks.items():
        if split.x.ndim != 2 or split.x.shape[1] != INPUT_DIM:
            raise ValueError(f"{task_id}: expected x with shape (n, {INPUT_DIM})")
        if split.y.shape != (len(split.x),):
            raise ValueError(f"{task_id}: y length does not match x")
        if len(split.x) < batch_size:
            raise ValueError(f"{task_id}: fewer rows than batch_size")


def _probe_accuracy(
    tasks: Mapping[str, TaskSplit],
    encoder: Mapping[str, np.ndarray],
    core_params: Mapping[str, np.ndarray],
    heads: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, float]:
    accuracy = {}
    for task_id in HONEST_TASKS:
        split = tasks[task_id]
        features = np.maximum(split.x @ encoder["w"] + encoder["b"], 0.0)
        latent, _ = forward(core_params, features)
        logits = latent @ heads[task_id]["w"] + heads[task_id]["b"]
        predicted = (_sigmoid(logits[:, 0]) >= 0.5).astype(np.float32)
        accuracy[task_id] = float(np.mean(predicted == split.y))
    return accuracy


def train_core(
    tasks: Mapping[str, TaskSplit] | None = None,
    *,
    seed: int = CORE_SEED,
    data_seed: int = DATA_SEED,
    steps: int = CORE_STEPS,
    batch_size: int = CORE_BATCH_SIZE,
    learning_rate: float = CORE_LEARNING_RATE,
    weight_decay: float = CORE_WEIGHT_DECAY,
) -> FrozenCore:
    """Jointly train probe heads and the shared trunk, then discard probes.

    The temporary input encoder and heads exist only to give the Core a
    multi-task training signal. Wings later learn their own upstream
    transforms and heads through the frozen trunk.
    """

    if tasks is None:
        loaded = load_tasks(data_seed)
        tasks = {task_id: loaded[task_id] for task_id in HONEST_TASKS}
    _validate_training(tasks, steps, batch_size, learning_rate)
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")

    rng = np.random.default_rng(seed)
    core_params = init_params(WIDTHS, rng)
    encoder = {
        "w": (
            rng.uniform(-1.0, 1.0, size=(INPUT_DIM, WIDTHS[0])).astype(np.float32)
            * 0.5
        ),
        "b": np.zeros(WIDTHS[0], dtype=np.float32),
    }
    heads = {
        task_id: {
            "w": (
                rng.uniform(-1.0, 1.0, size=(WIDTHS[-1], 1)).astype(np.float32)
                * 0.25
            ),
            "b": np.zeros(1, dtype=np.float32),
        }
        for task_id in HONEST_TASKS
    }

    for step in range(steps):
        task_id = HONEST_TASKS[step % len(HONEST_TASKS)]
        split = tasks[task_id]
        rows = rng.integers(0, len(split.x), size=batch_size)
        x = split.x[rows].astype(np.float32, copy=False)
        y = split.y[rows, None].astype(np.float32, copy=False)

        encoder_pre = x @ encoder["w"] + encoder["b"]
        features = np.maximum(encoder_pre, 0.0)
        latent, cache = forward(core_params, features)
        head = heads[task_id]
        logits = latent @ head["w"] + head["b"]
        grad_logits = (_sigmoid(logits) - y) / batch_size

        head_w_before = head["w"].copy()
        grad_head_w = latent.T @ grad_logits + weight_decay * head["w"]
        grad_head_b = grad_logits.sum(axis=0)
        grad_latent = grad_logits @ head_w_before.T
        core_grads, grad_features = backward(
            core_params, features, cache, grad_latent
        )
        grad_encoder_pre = grad_features * (encoder_pre > 0.0)
        grad_encoder_w = x.T @ grad_encoder_pre + weight_decay * encoder["w"]
        grad_encoder_b = grad_encoder_pre.sum(axis=0)

        head["w"] -= learning_rate * grad_head_w
        head["b"] -= learning_rate * grad_head_b
        encoder["w"] -= learning_rate * grad_encoder_w
        encoder["b"] -= learning_rate * grad_encoder_b
        for name in core_params:
            decay = weight_decay * core_params[name] if name.startswith("w") else 0.0
            core_params[name] -= learning_rate * (core_grads[name] + decay)

    accuracy = _probe_accuracy(tasks, encoder, core_params, heads)
    meta = {
        "kind": "core",
        "epoch": EPOCH,
        "seed": seed,
        "data_seed": data_seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "tasks": list(HONEST_TASKS),
        "widths": list(WIDTHS),
        "training_probe_accuracy": accuracy,
    }
    digest = params_sha256(core_params)
    return FrozenCore(EPOCH, core_params, meta, digest)


if __name__ == "__main__":
    core = train_core()
    print(f"Core {core.epoch}: {core.sha256}")
    for task_id, accuracy in core.meta["training_probe_accuracy"].items():
        print(f"{task_id}: training probe accuracy={accuracy:.3f}")
