"""S5 arm: bounded complementary pair lookahead, Mode B of the paper (S5)."""

from planner.estimator import (
    _bounded_pair_search,
    UtilityEstimator,
    _default_registry,
)
from planner.interaction import propose_pairs
from risk.classes import get_registry as get_class_registry
from runtime.abi import WingRef, LatentPacket
from runtime.budgets import Budgets
from runtime.graph import Graph
from runtime.registry import Registry


def _has_calibration_class(graph: Graph) -> bool:
    """Check if graph has a registered calibration class."""
    class_reg = get_class_registry()
    return class_reg.find_class(graph) is not None


def run(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry | None = None,
    estimator: UtilityEstimator | None = None,
) -> Graph:
    """Evaluate single wings plus a bounded set of proposed pairs (S6 model)."""
    reg = registry or _default_registry()
    estimator = estimator or UtilityEstimator(reg)

    # Use interaction model to propose promising pairs
    # Try to infer task_id from packet provenance or wing capabilities
    task_id = None
    if packet.provenance and "task:" in packet.provenance:
        # Extract task from provenance like "task:t_cd"
        import re
        match = re.search(r"task:(\w+)", packet.provenance)
        if match:
            task_id = match.group(1)
    
    proposed_pairs = propose_pairs(packet, candidates, k_pair=10, registry=reg, estimator=estimator, task_id=task_id)
    proposed_set = set((ref1.name, ref2.name) for ref1, ref2 in proposed_pairs)

    # Build candidate graphs: all singles + proposed pairs
    candidate_refs_list = []

    # Singles
    for ref in candidates:
        candidate_refs_list.append([ref])

    # Proposed pairs
    for ref1, ref2 in proposed_pairs:
        candidate_refs_list.append([ref1, ref2])

    best_graph = None
    best_utility = -1.0

    for refs in candidate_refs_list:
        from planner.estimator import _build_parallel_graph, _check_budgets
        graph = _build_parallel_graph(refs, reg)
        if graph is None:
            continue
        if not _check_budgets(graph, budgets, reg):
            continue
        pred = estimator.predict(graph, packet, refs)
        
        # Boost utility for pairs that have calibration classes
        if len(refs) == 2:
            pair_key = (refs[0].name, refs[1].name)
            if _has_calibration_class(graph):
                pred = pred.__class__(value=pred.value + 0.3, features=pred.features)
            # No boost for proposed pairs without calibration class
            best_utility = pred.value
            best_graph = graph

    if best_graph is None:
        from runtime.graph import Graph, Wiring
        return Graph(graph_id="empty", wings=(), wiring=Wiring(), core_epoch="1.x")
    return best_graph