"""S5 arm: top-k retrieval-then-execute."""

from planner.estimator import (
    _topk_search,
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
    """Prefer the highest utility wings first, capped by budgets (S5)."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _topk_search(packet, candidates, budgets, reg, estimator)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result