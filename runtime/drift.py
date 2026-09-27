"""Cumulative drift gating for promotion and demotion.

Implements the CCE promotion protocol with step drift and origin drift
constraints, protection tiers (A/B/C), power budgeting, and interference
matrix tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from runtime.abi import WingRef, Decision
from runtime.registry import Registry, HostMetrics


class Tier(Enum):
    """Protection tiers for capabilities (S6/S7)."""
    A = "critical"      # Strict non-inferiority, family-wise error control, pinned origin
    B = "important"     # Non-inferiority, scheduled full regression
    C = "exploratory"   # FDR control; promotion after sufficient evidence


class DriftStatus(Enum):
    """Result of a drift check."""
    PASS = "pass"
    STEP_DRIFT = "step_drift"
    ORIGIN_DRIFT = "origin_drift"
    BOTH = "both"


@dataclass(frozen=True)
class DriftConfig:
    """Configuration for drift constraints per capability."""
    epsilon: float = 0.01    # Per-step tolerance (e.g., -0.01 = 1% degradation)
    E: float = 0.05          # Cumulative budget (e.g., -0.05 = 5% total degradation)


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a protection tier."""
    tier: Tier
    alpha: float = 0.05      # Significance level
    beta: float = 0.20       # Type II error (1 - power)
    delta: float = 0.02      # Minimum effect size to detect
    min_samples: int = 100   # Minimum samples for test

    def __post_init__(self):
        if self.tier == Tier.A:
            object.__setattr__(self, 'alpha', 0.05)
            object.__setattr__(self, 'beta', 0.20)
            object.__setattr__(self, 'delta', 0.02)
            object.__setattr__(self, 'min_samples', 100)
        elif self.tier == Tier.B:
            object.__setattr__(self, 'alpha', 0.10)
            object.__setattr__(self, 'beta', 0.20)
            object.__setattr__(self, 'delta', 0.03)
            object.__setattr__(self, 'min_samples', 50)
        elif self.tier == Tier.C:
            object.__setattr__(self, 'alpha', 0.20)
            object.__setattr__(self, 'beta', 0.20)
            object.__setattr__(self, 'delta', 0.05)
            object.__setattr__(self, 'min_samples', 30)


@dataclass
class CapabilityState:
    """Tracks performance history and drift state for one capability."""
    ref: WingRef
    tier: Tier
    config: DriftConfig
    origin_performance: float | None = None  # P(0) - permanently pinned
    previous_performance: float | None = None  # P(t-1)
    current_performance: float | None = None  # P(t)
    step_drift: float | None = None
    origin_drift: float | None = None
    samples: int = 0
    is_eligible: bool = True
    demoted: bool = False
    demotion_reason: str = ""
    history: list[float] = field(default_factory=list)  # Full P(t) history


@dataclass
class DriftCheckResult:
    """Result of evaluating drift for one capability."""
    capability: WingRef
    status: DriftStatus
    step_drift: float
    origin_drift: float
    epsilon: float
    E: float
    current_performance: float
    previous_performance: float
    origin_performance: float


@dataclass
class InterferenceEntry:
    """One cell of the promotion interference matrix."""
    promotion_event: int
    affected_capability: WingRef
    delta: float  # P_i(after) - P_i(before)
    tier: Tier


@dataclass
class PromotionRecord:
    """Audit record for a promotion event."""
    event_id: int
    promoted_wing: WingRef
    old_version: str
    new_version: str
    step_drift: float
    origin_drift: float
    interference: list[InterferenceEntry]
    admitted: bool
    reason: str
    timestamp: str


class DriftGate:
    """Cumulative drift gating for promotion and demotion decisions."""

    def __init__(
        self,
        registry: Registry,
        tier_configs: dict[Tier, TierConfig] | None = None,
        default_config: DriftConfig | None = None,
    ):
        self._registry = registry
        self._capabilities: dict[str, CapabilityState] = {}
        self._tier_configs = tier_configs or {
            Tier.A: TierConfig(Tier.A, alpha=0.05, beta=0.20, delta=0.02, min_samples=100),
            Tier.B: TierConfig(Tier.B, alpha=0.10, beta=0.20, delta=0.03, min_samples=50),
            Tier.C: TierConfig(Tier.C, alpha=0.20, beta=0.20, delta=0.05, min_samples=30),
        }
        self._default_config = default_config or DriftConfig(epsilon=0.01, E=0.05)
        self._promotion_events: list[PromotionRecord] = []
        self._interference_matrix: list[InterferenceEntry] = []
        self._event_counter = 0
        self._tier_a_capabilities: set[str] = set()
        self._tier_a_max: int | None = None  # Computed before first promotion
        self._total_samples: int = 0  # For power budget computation

    def register_capability(
        self,
        ref: WingRef,
        tier: Tier = Tier.B,
        config: DriftConfig | None = None,
    ) -> None:
        """Register a capability for drift tracking."""
        if ref.name in self._capabilities:
            raise ValueError(f"Capability {ref.name} already registered")
        self._capabilities[ref.name] = CapabilityState(
            ref=ref,
            tier=tier,
            config=config or self._default_config,
        )
        if tier == Tier.A:
            self._tier_a_capabilities.add(ref.name)

    def set_origin_performance(self, ref: WingRef, performance: float) -> None:
        """Pin the origin performance P(0). Must be called exactly once per capability."""
        if ref.name not in self._capabilities:
            raise ValueError(f"Capability {ref.name} not registered")
        state = self._capabilities[ref.name]
        if state.origin_performance is not None:
            raise ValueError(f"Origin performance already pinned for {ref.name}")
        state.origin_performance = performance
        state.current_performance = performance
        state.history = [performance]

    def update_performance(self, ref: WingRef, performance: float) -> None:
        """Update current performance P(t) from shadow/production evaluation."""
        if ref.name not in self._capabilities:
            raise ValueError(f"Capability {ref.name} not registered")
        state = self._capabilities[ref.name]
        state.previous_performance = state.current_performance
        state.current_performance = performance
        state.history.append(performance)
        state.samples += 1

        # Compute drifts
        if state.previous_performance is not None:
            state.step_drift = performance - state.previous_performance
        if state.origin_performance is not None:
            state.origin_drift = performance - state.origin_performance

    def check_drift(self, ref: WingRef) -> DriftCheckResult:
        """Evaluate both step and origin drift for a capability."""
        if ref.name not in self._capabilities:
            raise ValueError(f"Capability {ref.name} not registered")
        state = self._capabilities[ref.name]
        if state.current_performance is None or state.origin_performance is None:
            raise ValueError(f"Insufficient performance data for {ref.name}")

        step = state.step_drift if state.step_drift is not None else 0.0
        origin = state.origin_drift if state.origin_drift is not None else 0.0

        # Use small tolerance for floating point comparison
        tol = 1e-12
        step_ok = step >= -state.config.epsilon - tol
        origin_ok = origin >= -state.config.E - tol

        if step_ok and origin_ok:
            status = DriftStatus.PASS
        elif not step_ok and not origin_ok:
            status = DriftStatus.BOTH
        elif not step_ok:
            status = DriftStatus.STEP_DRIFT
        else:
            status = DriftStatus.ORIGIN_DRIFT

        return DriftCheckResult(
            capability=ref,
            status=status,
            step_drift=step,
            origin_drift=origin,
            epsilon=state.config.epsilon,
            E=state.config.E,
            current_performance=state.current_performance,
            previous_performance=state.previous_performance or state.origin_performance,
            origin_performance=state.origin_performance,
        )

    def can_promote(self, ref: WingRef) -> tuple[bool, DriftCheckResult]:
        """Check if a capability can be promoted (both drift gates pass)."""
        result = self.check_drift(ref)
        return result.status == DriftStatus.PASS, result

    def demote(self, ref: WingRef, reason: str) -> None:
        """Demote a capability from routing eligibility."""
        if ref.name not in self._capabilities:
            raise ValueError(f"Capability {ref.name} not registered")
        state = self._capabilities[ref.name]
        state.is_eligible = False
        state.demoted = True
        state.demotion_reason = reason

    def restore_eligibility(self, ref: WingRef) -> None:
        """Restore eligibility after demotion (requires full re-evaluation)."""
        if ref.name not in self._capabilities:
            raise ValueError(f"Capability {ref.name} not registered")
        state = self._capabilities[ref.name]
        state.is_eligible = True
        state.demoted = False
        state.demotion_reason = ""

    def is_eligible(self, ref: WingRef) -> bool:
        """Check if a capability is currently eligible for routing."""
        if ref.name not in self._capabilities:
            return True  # Unknown capabilities default to eligible
        return self._capabilities[ref.name].is_eligible

    def record_promotion(
        self,
        promoted_ref: WingRef,
        old_version: str,
        new_version: str,
        step_drift: float,
        origin_drift: float,
        admitted: bool,
        reason: str,
    ) -> PromotionRecord:
        """Record a promotion event and measure interference."""
        self._event_counter += 1
        interference = []

        # Measure interference on other capabilities
        # In practice, this would run evaluation on affected routes
        # For now, we record the structure
        for cap_name, state in self._capabilities.items():
            if cap_name != promoted_ref.name and not state.demoted:
                # Placeholder: actual measurement would run here
                interference.append(InterferenceEntry(
                    promotion_event=self._event_counter,
                    affected_capability=state.ref,
                    delta=0.0,  # Would be measured
                    tier=state.tier,
                ))

        record = PromotionRecord(
            event_id=self._event_counter,
            promoted_wing=promoted_ref,
            old_version=old_version,
            new_version=new_version,
            step_drift=step_drift,
            origin_drift=origin_drift,
            interference=interference,
            admitted=admitted,
            reason=reason,
            timestamp="now",  # Would use actual timestamp
        )
        self._promotion_events.append(record)
        self._interference_matrix.extend(interference)
        return record

    def get_interference_matrix(
        self,
        promotion_event: int | None = None,
    ) -> list[InterferenceEntry]:
        """Get the promotion interference matrix."""
        if promotion_event is None:
            return list(self._interference_matrix)
        return [e for e in self._interference_matrix if e.promotion_event == promotion_event]

    def compute_tier_a_max(self, total_samples: int | None = None) -> int:
        """Compute |A|max: largest k such that power(delta, n, alpha/k) >= 1 - beta.

        Uses normal approximation for two-sample proportion test.
        Must be called BEFORE the first promotion.
        """
        if self._tier_a_max is not None:
            return self._tier_a_max

        n = total_samples or self._total_samples
        if n == 0:
            raise ValueError("No samples available for power computation")

        tier_a = self._tier_configs[Tier.A]
        alpha = tier_a.alpha
        beta = tier_a.beta
        delta = tier_a.delta

        k = 1
        while True:
            power = compute_power(n, alpha / k, beta, delta)
            if power < 1 - beta:
                break
            k += 1
        self._tier_a_max = k - 1
        return self._tier_a_max

    def get_tier_a_count(self) -> int:
        """Get current number of Tier A capabilities."""
        return len(self._tier_a_capabilities)

    def get_all_status(self) -> dict[str, DriftCheckResult]:
        """Get drift status for all registered capabilities."""
        return {
            name: self.check_drift(state.ref)
            for name, state in self._capabilities.items()
        }

    def get_demoted(self) -> list[WingRef]:
        """Get all demoted capabilities."""
        return [
            state.ref for state in self._capabilities.values()
            if state.demoted
        ]

    def get_promotion_history(self) -> list[PromotionRecord]:
        """Get all promotion audit records."""
        return list(self._promotion_events)


# Global default instance
_DEFAULT_DRIFT_GATE: DriftGate | None = None


def get_drift_gate(registry: Registry | None = None) -> DriftGate:
    """Get or create the process-wide drift gate."""
    global _DEFAULT_DRIFT_GATE
    from runtime.registry import InMemoryRegistry
    if _DEFAULT_DRIFT_GATE is None:
        _DEFAULT_DRIFT_GATE = DriftGate(registry or InMemoryRegistry())
    return _DEFAULT_DRIFT_GATE


def compute_power(n: int, alpha: float, beta: float, delta: float) -> float:
    """Compute power of one-sided test with normal approximation."""
    try:
        from scipy import stats
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(1 - beta)
        return float(stats.norm.cdf(delta * math.sqrt(n) - z_alpha))
    except ImportError:
        # Fallback without scipy
        z_alpha = 1.645 if alpha == 0.05 else 1.282 if alpha == 0.10 else 0.842
        z_beta = 0.842 if beta == 0.20 else 1.282 if beta == 0.10 else 1.645
        return float(1.0 / (1.0 + math.exp(-(delta * math.sqrt(n) - z_alpha - z_beta))))


if __name__ == "__main__":
    from runtime.registry import InMemoryRegistry
    from runtime.abi import WingRef

    registry = InMemoryRegistry()
    gate = DriftGate(registry)

    # Register capabilities
    ref_alpha = WingRef("alpha", "1.0.0")
    ref_gamma = WingRef("gamma", "1.0.0")

    gate.register_capability(ref_alpha, Tier.A)
    gate.register_capability(ref_gamma, Tier.B)

    # Set origin performance
    gate.set_origin_performance(ref_alpha, 0.95)
    gate.set_origin_performance(ref_gamma, 0.90)

    # Update with slight degradation
    gate.update_performance(ref_alpha, 0.945)  # -0.005 step, -0.005 origin
    gate.update_performance(ref_gamma, 0.89)   # -0.01 step, -0.01 origin

    for ref in [ref_alpha, ref_gamma]:
        can, result = gate.can_promote(ref)
        print(f"{ref.name}: can_promote={can}, step={result.step_drift:.4f}, "
              f"origin={result.origin_drift:.4f}, status={result.status.value}")

    # Check tier A max
    gate._total_samples = 500
    print(f"|A|max = {gate.compute_tier_a_max()}")
    print(f"Current Tier A count: {gate.get_tier_a_count()}")