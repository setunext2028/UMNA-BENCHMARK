"""S5 arm: independent sigmoid gating per wing, budgets arbitrate (S5)."""

from planner.estimator import (
    _gates_search,
    UtilityEstimator,
    _default_registry,
)
from runtime.abi import WingRef, LatentPacket
from runtime.budgets import Budgets
from runtime.graph import Graph
from runtime.registry import Registry


def run(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry | None = None,
    estimator: UtilityEstimator | None = None,
) -> Graph:
    """Each wing passes its own gate; the allocator resolves budget ties."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _gates_search(packet, candidates, budgets, reg, estimator)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result