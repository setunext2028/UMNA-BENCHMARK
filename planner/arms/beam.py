"""S5 arm: beam search over partial graphs, Mode C of the paper (S5)."""

from planner.estimator import (
    _beam_search,
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
    """Maintain a beam of width b over legal graph continuations (S5)."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)
    result = _beam_search(packet, candidates, budgets, reg, estimator, beam_width=4)
    if result is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return result