"""S5 arm: restricted oracle over subsets of size <= 3 under a fixed fusion
topology. Reports regret against its declared restriction (paper 29.2).
"""

from planner.estimator import (
    _oracle_search,
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
    """Exhaustive evaluation over the declared restriction (S5)."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _oracle_search(packet, candidates, budgets, reg, estimator, max_size=3)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result