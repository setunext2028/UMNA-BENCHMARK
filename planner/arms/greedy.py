"""S5 arm: greedy marginal utility, Mode A of the paper (S5)."""

from planner.estimator import (
    _greedy_search,
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
    """Add the wing with the highest marginal utility until budgets bind."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _greedy_search(packet, candidates, budgets, reg, estimator)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result