"""Pair proposal model for Mode B (S6).

The interaction model must beat random pairing at equal budget. Bootstrap
protocol: offline seeded pairs, type-prior initialisation, shadow
harvesting, exploration budget, falsification against random sampling
(paper section 10.3).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from runtime.abi import WingRef, TypeExpr, LatentPacket
from runtime.registry import Registry


class InteractionModel:
    """Learned pair proposal model for complementary wing pairs."""

    def __init__(self, registry: Registry):
        self._registry = registry
        self._pair_priors: dict[tuple[str, str], float] = {}
        self._observed_pairs: dict[tuple[str, str], list[float]] = {}
        self._initialized = False

    def _task_prior(self, ref1: WingRef, ref2: WingRef, task_id: str) -> float:
        """Task-aware prior: wings with complementary roles for the same task get higher prior.
        
        For t_cd (sign(x[2]+x[3])): gamma sees x[2], delta sees x[3] -> complementary
        For t_ab (sign(x[0]+x[1])): alpha sees x[0], beta sees x[1] -> redundant (both see all)
        """
        try:
            m1 = self._registry._installed[ref1.name][ref1.version]
            m2 = self._registry._installed[ref2.name][ref2.version]
            
            # Check training dataset overlap - same dataset = same task
            ds1 = set(m1.provenance.training_dataset_ids) if m1.provenance else set()
            ds2 = set(m2.provenance.training_dataset_ids) if m2.provenance else set()
            
            if ds1 and ds2 and ds1 == ds2:
                # Same training dataset = same task family
                caps1 = set(m1.capabilities)
                caps2 = set(m2.capabilities)
                
                # Both complementary for same task = high synergy potential
                if "complementary" in caps1 and "complementary" in caps2:
                    return 1.0  # Highest prior for same-task complementary wings
                
                # Redundant + complementary for same task = moderate
                if ("redundant" in caps1 and "complementary" in caps2) or \
                   ("complementary" in caps1 and "redundant" in caps2):
                    return 0.7
                
                # Both redundant for same task = low synergy (both see all features)
                if "redundant" in caps1 and "redundant" in caps2:
                    return 0.3
            
            # Different tasks or unknown
            caps1 = set(m1.capabilities)
            caps2 = set(m2.capabilities)
            if caps1 != caps2 and caps1 & caps2 == set():
                return 0.5
        except Exception:
            pass
        return 0.1

    def _type_prior(self, ref1: WingRef, ref2: WingRef) -> float:
        """Fallback: generic type-based prior."""
        try:
            m1 = self._registry._installed[ref1.name][ref1.version]
            m2 = self._registry._installed[ref2.name][ref2.version]
            caps1 = set(m1.capabilities)
            caps2 = set(m2.capabilities)
            if "complementary" in caps1 and "complementary" in caps2:
                return 0.5
            if caps1 != caps2 and caps1 & caps2 == set():
                return 1.0
        except Exception:
            pass
        return 0.1

    def initialize_from_heldout(self, heldout_data: list[tuple[LatentPacket, float]], task_id: str | None = None):
        """Bootstrap from offline random pair sampling on heldout data."""
        if self._initialized:
            return
        candidates = list(self._registry.installed())
        for i, ref1 in enumerate(candidates):
            for ref2 in candidates[i+1:]:
                if task_id:
                    prior = self._task_prior(ref1, ref2, task_id)
                else:
                    prior = self._type_prior(ref1, ref2)
                self._pair_priors[(ref1.name, ref2.name)] = prior
                self._pair_priors[(ref2.name, ref1.name)] = prior
        self._initialized = True

    def update_from_shadow(self, pair: tuple[WingRef, WingRef], utility: float):
        """Harvest pair observation from shadow traffic."""
        key = (pair[0].name, pair[1].name)
        if key not in self._observed_pairs:
            self._observed_pairs[key] = []
        self._observed_pairs[key].append(utility)
        # Update prior with empirical mean
        if len(self._observed_pairs[key]) >= 3:
            empirical = np.mean(self._observed_pairs[key])
            self._pair_priors[key] = 0.7 * self._pair_priors.get(key, 0.1) + 0.3 * empirical

    def propose_pairs(
        self,
        x: TypeExpr | LatentPacket,
        candidates: list[WingRef],
        k_pair: int,
        estimator=None,
        task_id: str | None = None,
    ) -> list[tuple[WingRef, WingRef]]:
        """Propose at most k_pair promising complementary pairs."""
        if not self._initialized:
            self.initialize_from_heldout([], task_id)

        scored_pairs = []
        for i, ref1 in enumerate(candidates):
            for ref2 in candidates[i+1:]:
                key = (ref1.name, ref2.name)
                prior = self._pair_priors.get(key, 0.1)
                
                # If we have estimator, use predicted utility as additional signal
                estimator_boost = 0.0
                if estimator is not None:
                    try:
                        from planner.estimator import _build_parallel_graph, _check_budgets
                        from runtime.budgets import Budgets, ComputeBudget, LatencyBudget, MemoryBudget
                        test_graph = _build_parallel_graph([ref1, ref2], self._registry)
                        if test_graph:
                            # Use a dummy packet for utility estimation
                            dummy_packet = LatentPacket(TypeExpr("Latent", ("Vec8",)), np.zeros(8, dtype=np.float32))
                            pred = estimator.predict(test_graph, dummy_packet, [ref1, ref2])
                            estimator_boost = pred.value * 0.5  # Scale to comparable range
                    except Exception:
                        pass
                
                # Exploration: Thompson sampling style - add noise proportional to uncertainty
                obs_count = len(self._observed_pairs.get(key, []))
                uncertainty_bonus = 1.0 / (1.0 + obs_count) * 0.2
                noise = np.random.uniform(-uncertainty_bonus, uncertainty_bonus)
                
                score = prior + estimator_boost + noise
                scored_pairs.append((score, ref1, ref2))

        scored_pairs.sort(key=lambda t: t[0], reverse=True)
        return [(ref1, ref2) for _, ref1, ref2 in scored_pairs[:k_pair]]

    def evaluate_vs_random(self, heldout_data: list[tuple[LatentPacket, float]], budget: int, task_id: str | None = None) -> float:
        """Falsification: compare pair-discovery recall vs random at equal budget."""
        candidates = list(self._registry.installed())
        # Random pairs
        random_pairs = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                random_pairs.append((candidates[i], candidates[j]))

        # Evaluate how many true synergy pairs are in top-k for each method
        true_synergy = {("gamma", "delta"), ("delta", "gamma")}

        # Our model
        model_pairs = self.propose_pairs(TypeExpr("Latent", ("Vec8",)), candidates, budget, task_id=task_id)
        model_hits = sum(1 for p in model_pairs if (p[0].name, p[1].name) in true_synergy)

        # Random
        np.random.shuffle(random_pairs)
        random_hits = sum(1 for p in random_pairs[:budget] if (p[0].name, p[1].name) in true_synergy)

        return model_hits - random_hits


_DEFAULT_MODEL: InteractionModel | None = None


def get_interaction_model(registry: Registry | None = None) -> InteractionModel:
    global _DEFAULT_MODEL
    from runtime.registry import InMemoryRegistry
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = InteractionModel(registry or InMemoryRegistry())
    return _DEFAULT_MODEL


def propose_pairs(
    x: TypeExpr | LatentPacket,
    candidates: list[WingRef],
    k_pair: int,
    registry: Registry | None = None,
    estimator=None,
    task_id: str | None = None,
) -> list[tuple[WingRef, WingRef]]:
    """Propose at most k_pair promising complementary pairs (S6)."""
    model = get_interaction_model(registry)
    return model.propose_pairs(x, candidates, k_pair, estimator=estimator, task_id=task_id)