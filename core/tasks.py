"""The deterministic tabular task family for Core epoch 1.x (S3).

One modality, one task family, exact reproducibility. Every task is a
binary decision over an 8-dimensional input in [-1, 1]^8 and carries the
role structure the MVP needs: alpha/beta are redundant, gamma/delta are
complementary (each sees half of the deciding features), epsilon is
misleading (trained on permuted labels), zeta is cheap-but-weak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INPUT_DIM = 8

TASK_AB = "t_ab"
TASK_CD = "t_cd"
TASK_M = "t_m"
TASK_Z = "t_z"

HONEST_TASKS = (TASK_AB, TASK_CD, TASK_M, TASK_Z)

TRAIN_ROWS = 3000
HOLDOUT_ROWS = 1000
LABEL_FLIP = 0.05


@dataclass(frozen=True)
class TaskSplit:
    """Labeled inputs for one task. `features` is the Wing's own column view."""

    task_id: str
    features: tuple[int, ...]
    x: np.ndarray
    y: np.ndarray

    def view(self) -> "TaskSplit":
        return TaskSplit(
            task_id=self.task_id,
            features=self.features,
            x=self.x[:, list(self.features)],
            y=self.y,
        )


def _sign(v: np.ndarray) -> np.ndarray:
    return np.where(v > 0.0, 1.0, 0.0)


def generate_task(
    task_id: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample inputs and labels for one task from a seeded generator."""

    x = rng.uniform(-1.0, 1.0, size=(TRAIN_ROWS + HOLDOUT_ROWS, INPUT_DIM))
    if task_id == TASK_AB:
        y = _sign(x[:, 0] + x[:, 1])
    elif task_id == TASK_CD:
        y = _sign(x[:, 2] + x[:, 3])
    elif task_id == TASK_M:
        y = _sign(x[:, 4])
    elif task_id == TASK_Z:
        y = _sign(x[:, 7])
    else:
        raise ValueError(f"unknown task {task_id!r}")
    flip = rng.random(len(y)) < LABEL_FLIP
    y = np.where(flip, 1.0 - y, y)
    return x, y


def split_rows(seed: int) -> tuple[slice, slice]:
    train_stop = TRAIN_ROWS
    return slice(0, train_stop), slice(train_stop, None)


def load_tasks(seed: int) -> dict[str, TaskSplit]:
    """All honest tasks, one deterministic sample per task."""

    rng = np.random.default_rng(seed)
    train, holdout = split_rows(seed)
    tasks = {}
    for task_id in HONEST_TASKS:
        x, y = generate_task(task_id, rng)
        tasks[task_id] = TaskSplit(
            task_id=task_id,
            features=(),
            x=x[train],
            y=y[train],
        )
        tasks[f"{task_id}:holdout"] = TaskSplit(
            task_id=f"{task_id}:holdout",
            features=(),
            x=x[holdout],
            y=y[holdout],
        )
    return tasks


def task_features(task_id: str) -> tuple[int, ...]:
    if task_id in (TASK_AB,):
        return (0, 1)
    if task_id == TASK_CD:
        return (2, 3)
    if task_id == TASK_M:
        return (4,)
    if task_id == TASK_Z:
        return (7,)
    raise ValueError(f"unknown task {task_id!r}")


def corrupted_labels(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """The misleading Wing's training labels: a fixed permutation of the
    honest ones. Same class balance, no learnable signal."""

    order = rng.permutation(len(y))
    return y[order]


if __name__ == "__main__":
    tasks = load_tasks(0)
    for task_id, split in tasks.items():
        if task_id.endswith(":holdout"):
            continue
        print(
            f"{task_id}: x.shape={split.x.shape}, "
            f"positive rate={split.y.mean():.3f}, "
            f"features={task_features(task_id)}"
        )