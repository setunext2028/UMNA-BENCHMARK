"""Hard budget types. Constraints, never penalties.

Rule 5: C(S) <= B_C, L(S) <= B_L, M(S) <= B_M are constraints. The default
planner operates with lambda = 0 and these budgets as its only cost terms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeBudget:
    """Active compute ceiling, in FLOPs."""

    max_flops: int


@dataclass(frozen=True)
class LatencyBudget:
    """End-to-end latency ceiling, in milliseconds."""

    max_ms: float


@dataclass(frozen=True)
class MemoryBudget:
    """Peak memory ceiling, in megabytes."""

    max_mb: float


@dataclass(frozen=True)
class Budgets:
    """The triple of hard constraints handed to the planner."""

    compute: ComputeBudget
    latency: LatencyBudget
    memory: MemoryBudget