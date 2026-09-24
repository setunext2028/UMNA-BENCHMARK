"""NICE-R v0.2 arm: marginal-value stopping with complementarity."""

from planner.estimator import (
    _nice_r_search,
    UtilityEstimator,
    _default_registry,
    Mode,
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
    """NICE-R v0.2: marginal-value stopping with complementarity modeling."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _nice_r_search(packet, candidates, budgets, reg, estimator)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result