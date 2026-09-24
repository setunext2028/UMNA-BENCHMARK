"""Host-measured utility estimator and the plan seam — NICE-R v0.2.

The planner maximizes predicted utility subject to hard budgets with
marginal-value stopping, complementarity modeling, and Pareto pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

from runtime.abi import WingRef, TypeExpr, LatentPacket, input_observables
from runtime.budgets import Budgets
from runtime.graph import Graph, Cost, BuildError, cost, Wiring, build
from runtime.registry import Registry, HostMetrics, InMemoryRegistry


class Mode(Enum):
    """The planner modes of the paper, section 10.1."""

    GREEDY = "greedy"                # Mode A
    BOUNDED_PAIR = "bounded_pair"    # Mode B
    BEAM = "beam"                    # Mode C
    NICE_R = "nice_r"                # Mode D: NICE-R v0.2


@dataclass(frozen=True)
class PredictedUtility:
    """Host-predicted task utility of a graph for one input."""

    value: float
    features: tuple[float, ...] = ()
    # For confidence-aware stopping
    uncertainty: float = 0.0


@dataclass(frozen=True)
class GraphCost:
    """Host-measured cost of a graph."""
    flops: float
    latency_ms: float
    memory_mb: float
    size: int  # number of wings

    def total(self, alpha: float, beta: float, gamma: float, eta: float) -> float:
        return alpha * self.flops + beta * self.latency_ms + gamma * self.memory_mb + eta * self.size


class UtilityEstimator:
    """Host-side utility estimator trained on controlled validation."""

    def __init__(self, registry: Registry):
        self._registry = registry
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def _extract_features(
        self,
        graph: Graph,
        packet: LatentPacket,
        refs: list[WingRef],
    ) -> np.ndarray:
        """Extract host-measured features for utility prediction."""
        features = []

        try:
            c = cost(graph, self._registry)
            features.extend([c.compute_flops, c.latency_ms, c.memory_mb])
        except Exception:
            features.extend([0.0, 0.0, 0.0])

        if refs:
            latencies = []
            memories = []
            reliabilities = []
            flops = []
            util_feats_list = []

            for ref in refs:
                try:
                    metrics: HostMetrics = self._registry.measured(ref)
                    latencies.append(metrics.measured_latency_ms)
                    memories.append(metrics.measured_memory_mb)
                    reliabilities.append(metrics.measured_reliability)
                    flops.append(metrics.measured_active_flops)
                    util_feats_list.extend(metrics.measured_utility_features)
                except Exception:
                    latencies.append(0.0)
                    memories.append(0.0)
                    reliabilities.append(0.0)
                    flops.append(0.0)

            features.append(sum(latencies))
            features.append(np.mean(latencies) if latencies else 0.0)
            features.append(max(latencies) if latencies else 0.0)

            features.append(sum(memories))
            features.append(np.mean(memories) if memories else 0.0)
            features.append(max(memories) if memories else 0.0)

            features.append(np.mean(reliabilities) if reliabilities else 0.0)
            features.append(min(reliabilities) if reliabilities else 0.0)

            features.append(sum(flops))
            features.append(np.mean(flops) if flops else 0.0)

            if util_feats_list:
                features.append(np.mean(util_feats_list))
                features.append(max(util_feats_list))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0] * 12)

        features.append(len(refs))

        input_feats = input_observables(packet)
        features.extend(input_feats)

        return np.array(features, dtype=np.float32)

    def train(
        self,
        training_graphs: list[Graph],
        training_packets: list[LatentPacket],
        training_refs: list[list[WingRef]],
        training_utilities: list[float],
    ) -> None:
        if not training_graphs:
            self._weights = None
            self._bias = 0.0
            self._residual_std = 0.1
            return

        X = []
        for g, packet, refs in zip(training_graphs, training_packets, training_refs):
            X.append(self._extract_features(g, packet, refs))
        X = np.array(X, dtype=np.float32)
        y = np.array(training_utilities, dtype=np.float32)

        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        xtx = X_centered.T @ X_centered
        xty = X_centered.T @ y_centered
        ridge = 1e-3 * np.eye(xtx.shape[0], dtype=np.float32)
        self._weights = np.linalg.pinv(xtx + ridge) @ xty
        self._bias = float(y.mean() - (X.mean(axis=0) @ self._weights))

        # Compute residual standard error for uncertainty estimation
        y_pred = X_centered @ self._weights + self._bias
        residuals = y - y_pred
        self._residual_std = float(np.std(residuals))

    def predict(
        self,
        graph: Graph,
        packet: LatentPacket,
        refs: list[WingRef],
    ) -> PredictedUtility:
        feats = self._extract_features(graph, packet, refs)
        if self._weights is None:
            return PredictedUtility(value=0.0, features=tuple(feats), uncertainty=0.0)
        value = float(self._bias + feats @ self._weights)
        # Uncertainty: use residual standard error from training as proxy
        if not hasattr(self, '_residual_std') or self._residual_std is None:
            uncertainty = 0.1  # default
        else:
            # Prediction variance ≈ x^T Cov(w) x + sigma^2
            # Simplified: use residual std as uncertainty proxy
            uncertainty = float(self._residual_std)
        return PredictedUtility(value=max(0.0, value), features=tuple(feats), uncertainty=uncertainty)


def _default_registry() -> Registry:
    return InMemoryRegistry()


def get_estimator(registry: Registry | None = None) -> UtilityEstimator:
    return UtilityEstimator(registry or _default_registry())


def estimate_utility(
    graph: Graph, packet: LatentPacket, refs: list[WingRef], registry: Registry | None = None
) -> PredictedUtility:
    return get_estimator(registry).predict(graph, packet, refs)


def _build_parallel_graph(
    refs: list[WingRef], registry: Registry
) -> Graph | None:
    try:
        artifacts = [registry.artifact(ref) for ref in refs]
        g = build(artifacts, Wiring())
        return g
    except (BuildError, Exception):
        return None


def _build_sequential_graph(
    refs: list[WingRef], registry: Registry
) -> Graph | None:
    if len(refs) < 2:
        return _build_parallel_graph(refs, registry)
    try:
        artifacts = [registry.artifact(ref) for ref in refs]
        edges = []
        for i in range(len(refs) - 1):
            src_name = artifacts[i].manifest().name
            dst_name = artifacts[i + 1].manifest().name
            edges.append((src_name, 0, dst_name, 0))
        wiring = Wiring(tuple(edges))
        g = build(artifacts, wiring)
        return g
    except (BuildError, Exception):
        return None


def _build_star_graph(
    refs: list[WingRef], registry: Registry, center_idx: int = 0
) -> Graph | None:
    if len(refs) < 2:
        return _build_parallel_graph(refs, registry)
    try:
        artifacts = [registry.artifact(ref) for ref in refs]
        center_name = artifacts[center_idx].manifest().name
        edges = []
        for i, artifact in enumerate(artifacts):
            if i == center_idx:
                continue
            src_name = artifact.manifest().name
            edges.append((src_name, 0, center_name, 0))
        wiring = Wiring(tuple(edges))
        g = build(artifacts, wiring)
        return g
    except (BuildError, Exception):
        return None


def _build_graph_with_wiring_patterns(
    refs: list[WingRef], registry: Registry
) -> list[Graph]:
    graphs = []

    g = _build_parallel_graph(refs, registry)
    if g is not None:
        graphs.append(g)

    g = _build_sequential_graph(refs, registry)
    if g is not None:
        graphs.append(g)

    for center_idx in range(len(refs)):
        g = _build_star_graph(refs, registry, center_idx)
        if g is not None:
            graphs.append(g)

    return graphs


def _get_graph_cost(graph: Graph, registry: Registry) -> GraphCost:
    """Compute host-measured cost for a graph."""
    c = cost(graph, registry)
    return GraphCost(
        flops=c.compute_flops,
        latency_ms=c.latency_ms,
        memory_mb=c.memory_mb,
        size=len(graph.wings),
    )


def _check_budgets(graph: Graph, budgets: Budgets, registry: Registry) -> bool:
    try:
        c = cost(graph, registry)
        if c.compute_flops > budgets.compute.max_flops:
            return False
        if c.latency_ms > budgets.latency.max_ms:
            return False
        if c.memory_mb > budgets.memory.max_mb:
            return False
        return True
    except Exception:
        return False


# --- Complementarity / Redundancy Modeling ---

def _compute_pair_complementarity(
    ref1: WingRef,
    ref2: WingRef,
    registry: Registry,
    estimator: UtilityEstimator,
    packet: LatentPacket,
) -> float:
    """
    Compute κ(i,j) = U({i,j}) - U({i}) - U({j}) + U(∅).
    Positive = synergistic, negative = redundant.
    """
    try:
        g_pair = _build_parallel_graph([ref1, ref2], registry)
        g_1 = _build_parallel_graph([ref1], registry)
        g_2 = _build_parallel_graph([ref2], registry)
        if g_pair is None or g_1 is None or g_2 is None:
            return 0.0
        u_pair = estimator.predict(g_pair, packet, [ref1, ref2]).value
        u_1 = estimator.predict(g_1, packet, [ref1]).value
        u_2 = estimator.predict(g_2, packet, [ref2]).value
        return u_pair - u_1 - u_2
    except Exception:
        return 0.0


def _compute_redundancy_penalty(
    refs: list[WingRef],
    registry: Registry,
    estimator: UtilityEstimator,
    packet: LatentPacket,
) -> float:
    """Sum of pairwise redundancy penalties for all pairs in refs."""
    penalty = 0.0
    for i, ref1 in enumerate(refs):
        for ref2 in refs[i+1:]:
            # Negative complementarity = redundancy
            kappa = _compute_pair_complementarity(ref1, ref2, registry, estimator, packet)
            if kappa < 0:
                penalty += abs(kappa)
    return penalty


# --- NICE-R v0.2 Search with Marginal-Value Stopping ---

def _nice_r_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
    # Cost weights (alpha, beta, gamma, eta for F, L, M, |S|) - normalized to budget fractions
    cost_weights: tuple[float, float, float, float] = (1e-5, 0.01, 0.01, 0.2),
    # Complementarity weight
    lambda_K: float = 1.0,
    # Redundancy penalty weight
    lambda_R: float = 1.0,
    # Stopping threshold
    tau: float = 0.05,
    # UCB exploration parameter
    kappa_ucb: float = 0.5,
    max_wings: int = 6,
) -> Graph | None:
    """
    NICE-R v0.2: Marginal-value stopping with complementarity and redundancy.
    
    At each step, adds wing w if:
        max_w [ UCB(ΔU) - λ_C ΔC - λ_R ΔR + λ_K ΔK ] > τ
    """
    from itertools import combinations

    # Pre-compute singleton utilities and costs
    singleton_scores = {}
    for ref in candidates:
        graphs = _build_graph_with_wiring_patterns([ref], registry)
        best_score = -1.0
        best_graph = None
        for g in graphs:
            if _check_budgets(g, budgets, registry):
                pred = estimator.predict(g, packet, [ref])
                if pred.value > best_score:
                    best_score = pred.value
                    best_graph = g
        if best_graph is not None:
            singleton_scores[ref] = (best_score, best_graph)

    if not singleton_scores:
        return None

    # Stage 1: Top-m singletons by utility
    m = min(len(singleton_scores), 6)
    top_singletons = sorted(singleton_scores.items(), key=lambda x: x[1][0], reverse=True)[:m]
    seed_refs = [ref for ref, _ in top_singletons]

    # Stage 2: Generate complementary pairs from seeds
    pair_scores = []
    for ref1, ref2 in combinations(seed_refs, 2):
        graphs = _build_graph_with_wiring_patterns([ref1, ref2], registry)
        for g in graphs:
            if not _check_budgets(g, budgets, registry):
                continue
            pred = estimator.predict(g, packet, [ref1, ref2])
            # Complementarity bonus
            kappa = _compute_pair_complementarity(ref1, ref2, registry, estimator, packet)
            score = pred.value + lambda_K * max(0.0, kappa)
            pair_scores.append((score, [ref1, ref2], g, kappa))

    pair_scores.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pair_scores[:4]  # top 4 complementary pairs

    # Stage 3: Beam search over pairs with third-wing additions
    beam_width = 4
    beam = []
    for score, refs, g, kappa in top_pairs:
        beam.append((g, refs, score, kappa))

    best_overall = beam[0] if beam else None

    for _ in range(max_wings - 2):
        if len(beam[0][1]) >= max_wings:
            break

        new_beam = []
        for graph, refs, score, kappa in beam:
            # Try adding each remaining candidate
            for ref in candidates:
                if ref in refs:
                    continue
                trial_refs = refs + [ref]
                graphs = _build_graph_with_wiring_patterns(trial_refs, registry)

                for new_graph in graphs:
                    if not _check_budgets(new_graph, budgets, registry):
                        continue

                    pred = estimator.predict(new_graph, packet, trial_refs)

                    # Marginal utility with UCB
                    marginal_util = pred.value - score
                    ucb = marginal_util + kappa_ucb * pred.uncertainty

                    # Marginal cost
                    old_cost = _get_graph_cost(graph, registry)
                    new_cost = _get_graph_cost(new_graph, registry)
                    delta_flops = new_cost.flops - old_cost.flops
                    delta_lat = new_cost.latency_ms - old_cost.latency_ms
                    delta_mem = new_cost.memory_mb - old_cost.memory_mb
                    delta_size = 1

                    # Redundancy penalty
                    redundancy = _compute_redundancy_penalty(trial_refs, registry, estimator, packet)

                    # Admission value
                    alpha_c, beta_c, gamma_c, eta_c = cost_weights
                    admission = (
                        ucb
                        - alpha_c * delta_flops / 1e6  # normalize
                        - beta_c * delta_lat
                        - gamma_c * delta_mem
                        - eta_c * delta_size
                        - lambda_R * redundancy
                    )

                    if admission > tau:
                        new_beam.append((new_graph, trial_refs, pred.value, redundancy))

        if not new_beam:
            break

        # Pareto pruning: remove dominated graphs
        pareto_beam = []
        for item in new_beam:
            g, refs, val, red = item
            dominated = False
            for other in new_beam:
                if other is item:
                    continue
                og, orefs, oval, ored = other
                if oval >= val and len(orefs) <= len(refs) and (oval > val or len(orefs) < len(refs)):
                    dominated = True
                    break
            if not dominated:
                pareto_beam.append(item)

        pareto_beam.sort(key=lambda x: x[2], reverse=True)
        beam = pareto_beam[:beam_width]

        if beam:
            if best_overall is None or beam[0][2] > best_overall[2]:
                best_overall = beam[0]

    return best_overall[0] if best_overall else None


# Keep existing search functions for backward compatibility / comparison

def _greedy_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
) -> Graph | None:
    """Greedy marginal utility search (Mode A) over wings with wiring patterns."""
    best_graph = None
    best_utility = -1.0
    current_refs: list[WingRef] = []
    remaining = list(candidates)

    while remaining:
        best_candidate = None
        best_marginal = -1.0
        best_candidate_graph = None

        for ref in remaining:
            trial_refs = current_refs + [ref]
            graphs = _build_graph_with_wiring_patterns(trial_refs, registry)
            for graph in graphs:
                if not _check_budgets(graph, budgets, registry):
                    continue

                pred = estimator.predict(graph, packet, trial_refs)
                marginal = pred.value
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_candidate = ref
                    best_candidate_graph = graph

        if best_candidate is None:
            break

        current_refs.append(best_candidate)
        remaining.remove(best_candidate)
        if best_candidate_graph is not None:
            best_graph = best_candidate_graph
        else:
            graphs = _build_graph_with_wiring_patterns(current_refs, registry)
            best_graph = graphs[0] if graphs else None

    return best_graph


def _topk_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
) -> Graph | None:
    scored = []
    for ref in candidates:
        graphs = _build_graph_with_wiring_patterns([ref], registry)
        for graph in graphs:
            if not _check_budgets(graph, budgets, registry):
                continue
            pred = estimator.predict(graph, packet, [ref])
            scored.append((pred.value, ref, graph))

    scored.sort(key=lambda t: t[0], reverse=True)

    best_graph = None
    current_refs: list[WingRef] = []

    for _, ref, _ in scored:
        trial_refs = current_refs + [ref]
        graphs = _build_graph_with_wiring_patterns(trial_refs, registry)
        for graph in graphs:
            if not _check_budgets(graph, budgets, registry):
                continue
            current_refs.append(ref)
            best_graph = graph
            break

    return best_graph


def _gates_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
) -> Graph | None:
    scored = []
    for ref in candidates:
        graphs = _build_graph_with_wiring_patterns([ref], registry)
        for graph in graphs:
            if not _check_budgets(graph, budgets, registry):
                continue
            pred = estimator.predict(graph, packet, [ref])
            gate_score = 1.0 / (1.0 + np.exp(-pred.value))
            scored.append((gate_score, pred.value, ref, graph))

    scored.sort(key=lambda t: t[0], reverse=True)

    best_graph = None
    current_refs: list[WingRef] = []

    for _, _, ref, _ in scored:
        trial_refs = current_refs + [ref]
        graphs = _build_graph_with_wiring_patterns(trial_refs, registry)
        for graph in graphs:
            if not _check_budgets(graph, budgets, registry):
                continue
            current_refs.append(ref)
            best_graph = graph
            break

    return best_graph


def _bounded_pair_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
    k_pair: int = 10,
) -> Graph | None:
    best_graph = None
    best_utility = -1.0

    all_graphs: list[tuple[Graph, list[WingRef], float]] = []

    from itertools import combinations

    for size in range(1, min(len(candidates) + 1, 4)):
        for combo in combinations(candidates, size):
            refs = list(combo)
            graphs = _build_graph_with_wiring_patterns(refs, registry)
            for graph in graphs:
                if not _check_budgets(graph, budgets, registry):
                    continue
                pred = estimator.predict(graph, packet, refs)
                all_graphs.append((graph, refs, pred.value))

    if not all_graphs:
        return None

    all_graphs.sort(key=lambda t: t[2], reverse=True)

    seen_combos = set()
    count = 0

    for graph, refs, util in all_graphs:
        combo_key = tuple(sorted([r.name for r in refs]))
        if combo_key not in seen_combos and count < k_pair:
            if best_graph is None or util > best_utility:
                best_graph = graph
                best_utility = util
            seen_combos.add(combo_key)
            count += 1

    return best_graph


def _beam_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
    beam_width: int = 4,
) -> Graph | None:
    """Beam search over wing subsets with wiring patterns (Mode C)."""
    from itertools import combinations

    beam: list[tuple[Graph, list[WingRef], float]] = []

    for ref in candidates[:beam_width]:
        graphs = _build_graph_with_wiring_patterns([ref], registry)
        for graph in graphs:
            if _check_budgets(graph, budgets, registry):
                pred = estimator.predict(graph, packet, [ref])
                beam.append((graph, [ref], pred.value))

    beam.sort(key=lambda t: t[2], reverse=True)
    beam = beam[:beam_width]

    for _ in range(len(candidates)):
        new_beam = []
        for graph, refs, _ in beam:
            for ref in candidates:
                if ref in refs:
                    continue
                trial_refs = refs + [ref]
                new_graphs = _build_graph_with_wiring_patterns(trial_refs, registry)
                for new_graph in new_graphs:
                    if not _check_budgets(new_graph, budgets, registry):
                        continue
                    pred = estimator.predict(new_graph, packet, trial_refs)
                    new_beam.append((new_graph, trial_refs, pred.value))

        if not new_beam:
            break

        new_beam.sort(key=lambda t: t[2], reverse=True)
        beam = new_beam[:beam_width]

    return beam[0][0] if beam else None


def _oracle_search(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    registry: Registry,
    estimator: UtilityEstimator,
    max_size: int = 3,
) -> Graph | None:
    from itertools import combinations

    best_graph = None
    best_utility = -1.0

    for size in range(1, min(max_size + 1, len(candidates) + 1)):
        for combo in combinations(candidates, size):
            refs = list(combo)
            graphs = _build_graph_with_wiring_patterns(refs, registry)
            for graph in graphs:
                if not _check_budgets(graph, budgets, registry):
                    continue
                pred = estimator.predict(graph, packet, refs)
                if pred.value > best_utility:
                    best_utility = pred.value
                    best_graph = graph

    return best_graph


_MODE_FUNCS: dict[Mode, Callable] = {
    Mode.GREEDY: _greedy_search,
    Mode.BEAM: _beam_search,
    Mode.BOUNDED_PAIR: _bounded_pair_search,
    Mode.NICE_R: _nice_r_search,
}


def plan(
    packet: LatentPacket,
    candidates: list[WingRef],
    budgets: Budgets,
    mode: Mode = Mode.GREEDY,
    registry: Registry | None = None,
) -> Graph:
    reg = registry or _default_registry()
    estimator = get_estimator(reg)

    if not candidates:
        raise ValueError("empty candidate set")

    func = _MODE_FUNCS.get(mode)
    if func is None:
        raise ValueError(f"unknown mode: {mode}")

    result = func(packet, candidates, budgets, reg, estimator)

    if result is None:
        return Graph(
            graph_id="empty",
            wings=(),
            wiring=Wiring(),
            core_epoch="1.x",
        )

    return result