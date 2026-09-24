"""CCE (Conservative Capability Extension) promotion pipeline.

Wires together:
- ShadowRouter: collects counterfactual evidence from shadow execution
- DriftGate: enforces step/origin drift constraints before promotion
- Admission: checks eligibility before production

The pipeline:
1. ShadowRouter runs candidate graphs in shadow, collects traces
2. Router learns to predict graph utility from shadow traces
3. When router has enough evidence (>=10 executions, >80% accept), it proposes promotion
4. DriftGate checks step/origin drift constraints for all protected capabilities
5. If drift gates pass, capability is promoted to production eligibility
6. Admission checks is_eligible before granting PRODUCTION disposition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from runtime.abi import WingRef, LatentPacket, Decision
from runtime.graph import Graph
from runtime.registry import Registry
from runtime.drift import DriftGate, Tier, get_drift_gate


if TYPE_CHECKING:
    from runtime.shadow import ShadowRouter, ShadowTrace, get_shadow_router, run_shadow, collect_shadow_traces
    from runtime.admission import Admission, AdmissionRecord, AdmissionError
    from runtime.drift import DriftCheckResult


@dataclass
class PromotionProposal:
    """A proposal to promote a shadow graph to production."""
    graph: Graph
    wing_ref: WingRef  # The new capability being promoted
    old_version: str
    new_version: str
    shadow_evidence: list  # ShadowTrace
    accept_rate: float
    n_shadow_executions: int


@dataclass
class PromotionDecision:
    """Result of a promotion evaluation."""
    proposal: PromotionProposal
    promoted: bool
    drift_results: dict[str, "DriftCheckResult"]
    reason: str


class CCEPromotionPipeline:
    """Conservative Capability Extension promotion pipeline.
    
    Orchestrates shadow execution, router learning, drift gating, and promotion.
    """
    
    def __init__(self, registry: Registry):
        from runtime.shadow import get_shadow_router
        self._registry = registry
        self._shadow_router = get_shadow_router(registry)
        self._drift_gate = get_drift_gate(registry)
        self._pending_promotions: dict[str, PromotionProposal] = {}
        self._promotion_history: list[PromotionDecision] = []
    
    def register_new_capability(
        self,
        wing_ref: WingRef,
        tier: Tier = Tier.B,
        origin_performance: float | None = None,
    ) -> None:
        """Register a new capability for CCE tracking."""
        self._drift_gate.register_capability(wing_ref, tier=tier)
        if origin_performance is not None:
            self._drift_gate.set_origin_performance(wing_ref, origin_performance)
    
    def register_protected_capabilities(
        self,
        wing_refs: list[WingRef],
        performances: dict[str, float],
        tier: Tier = Tier.B,
    ) -> None:
        """Register existing capabilities as protected."""
        for ref in wing_refs:
            self._drift_gate.register_capability(ref, tier=tier)
            if ref.name in performances:
                self._drift_gate.set_origin_performance(ref, performances[ref.name])
    
    def collect_shadow_evidence(
        self,
        candidate_graphs: list[Graph],
        packets: list[LatentPacket],
        execution_context,
    ) -> list:
        """Run candidate graphs in shadow and collect traces."""
        from runtime.shadow import collect_shadow_traces
        traces = collect_shadow_traces(
            candidate_graphs, packets, execution_context, self._registry
        )
        for trace in traces:
            self._shadow_router.add_shadow_trace(trace)
        return traces
    
    def update_router(self) -> None:
        """Retrain router on accumulated shadow evidence."""
        self._shadow_router.train()
    
    def evaluate_promotion_readiness(
        self,
        graph: Graph,
        wing_ref: WingRef,
        old_version: str,
        new_version: str,
    ) -> PromotionProposal | None:
        """Check if a graph has enough shadow evidence for promotion."""
        if not self._shadow_router.should_promote(graph):
            return None
        
        relevant = [
            st for st in self._shadow_router._shadow_traces
            if st.graph.graph_id == graph.graph_id
        ]
        accept_rate = sum(
            1 for st in relevant if st.trace.decision.value == "accept"
        ) / len(relevant)
        
        proposal = PromotionProposal(
            graph=graph,
            wing_ref=wing_ref,
            old_version=old_version,
            new_version=new_version,
            shadow_evidence=relevant,
            accept_rate=accept_rate,
            n_shadow_executions=len(relevant),
        )
        return proposal
    
    def check_drift_constraints(self) -> dict[str, "DriftCheckResult"]:
        """Check drift gates for all registered capabilities."""
        results = {}
        for name, state in self._drift_gate._capabilities.items():
            if state.demoted:
                continue
            try:
                result = self._drift_gate.check_drift(state.ref)
                results[name] = result
            except ValueError:
                # Insufficient data
                pass
        return results
    
    def evaluate_promotion(self, proposal: PromotionProposal) -> PromotionDecision:
        """Evaluate a promotion proposal against drift constraints."""
        # Check drift gates for all protected capabilities
        drift_results = self.check_drift_constraints()
        
        # Check if ALL drift gates pass
        all_pass = all(
            result.status.value == "pass" for result in drift_results.values()
        )
        
        if all_pass:
            # Check tier A max
            tier_a_count = self._drift_gate.get_tier_a_count()
            tier_a_max = self._drift_gate.compute_tier_a_max()
            
            if tier_a_count >= tier_a_max:
                reason = f"Tier A capacity full ({tier_a_count}/{tier_a_max})"
                promoted = False
            else:
                # Record promotion and compute interference
                step_drift = drift_results.get(proposal.wing_ref.name, None)
                step_val = step_drift.step_drift if step_drift else 0.0
                origin_val = drift_results.get(proposal.wing_ref.name, None)
                origin_val = origin_val.origin_drift if origin_val else 0.0
                
                self._drift_gate.record_promotion(
                    promoted_ref=proposal.wing_ref,
                    old_version=proposal.old_version,
                    new_version=proposal.new_version,
                    step_drift=step_val,
                    origin_drift=origin_val,
                    admitted=True,
                    reason="Drift gates passed",
                )
                promoted = True
                reason = "Drift gates passed, promoted to production"
        else:
            # Find failed capabilities
            failed = [
                f"{name}({result.status.value})"
                for name, result in drift_results.items()
                if result.status.value != "pass"
            ]
            # Demote failed capabilities
            for name, result in drift_results.items():
                if result.status.value != "pass":
                    self._drift_gate.demote(
                        self._drift_gate._capabilities[name].ref,
                        f"Drift gate failed: {result.status.value}"
                    )
            promoted = False
            reason = f"Drift gates failed: {', '.join(failed)}"
        
        decision = PromotionDecision(
            proposal=proposal,
            promoted=promoted,
            drift_results=drift_results,
            reason=reason,
        )
        self._promotion_history.append(decision)
        return decision
    
    def update_performance_from_trace(
        self,
        trace,
        graph: Graph,
    ) -> None:
        """Update capability performance from execution trace."""
        # Extract per-wing performance from trace
        for ref in graph.wings:
            wing_name = ref.name
            if wing_name in self._drift_gate._capabilities:
                # Find evidence from this wing
                wing_evidence = [
                    ev for ev in trace.evidence if ev.source_wing == wing_name
                ]
                if wing_evidence:
                    # Use local_score as performance proxy
                    perf = sum(ev.local_score for ev in wing_evidence) / len(wing_evidence)
                    self._drift_gate.update_performance(ref, perf)
    
    def check_admission_eligibility(self, graph: Graph) -> tuple[bool, str]:
        """Check if all wings in a graph are eligible for production routing."""
        for ref in graph.wings:
            if not self._drift_gate.is_eligible(ref):
                state = self._drift_gate._capabilities.get(ref.name)
                reason = f"Wing {ref.name} demoted: {state.demotion_reason if state else 'unknown'}"
                return False, reason
        return True, "All wings eligible"
    
    def run_promotion_cycle(
        self,
        candidate_graphs: list[Graph],
        packets: list[LatentPacket],
        execution_context,
        wing_ref: WingRef,
        old_version: str,
        new_version: str,
    ) -> list[PromotionDecision]:
        """Run a full CCE promotion cycle."""
        # 1. Collect shadow evidence
        self.collect_shadow_evidence(candidate_graphs, packets, execution_context)
        
        # 2. Update router
        self.update_router()
        
        # 3. Check promotion readiness for each candidate
        decisions = []
        for graph in candidate_graphs:
            proposal = self.evaluate_promotion_readiness(
                graph, wing_ref, old_version, new_version
            )
            if proposal:
                decision = self.evaluate_promotion(proposal)
                decisions.append(decision)
        
        return decisions


def check_cce_admission(
    graph: Graph,
    binding,
    registry: Registry,
):
    """Enhanced admission check that includes CCE drift gating.
    
    Sits between plan and execute. Checks:
    1. Standard calibration class admission
    2. CCE drift gate eligibility for all wings in graph
    """
    from runtime.admission import Admission, AdmissionRecord, AdmissionError
    from runtime.drift import get_drift_gate
    from risk.classes import get_registry as get_cal_registry
    
    # Standard admission check
    if binding.graph.graph_id != graph.graph_id:
        raise AdmissionError("binding graph does not match the graph to admit")
    if binding.core_epoch != graph.core_epoch:
        raise AdmissionError("binding Core epoch does not match the graph")

    cal_registry = get_cal_registry()
    cls = cal_registry.find_class(graph)
    if cls is not None:
        # Attach calibrator to binding for risk assessment
        object.__setattr__(binding, "_calibrator", cls.calibrator)
        record = AdmissionRecord(
            disposition=Admission.PRODUCTION,
            graph_id=graph.graph_id,
            binding_id=binding.binding_id,
            calibration_class=cls.class_id,
        )
    else:
        record = AdmissionRecord(
            disposition=Admission.SHADOW_ONLY,
            graph_id=graph.graph_id,
            binding_id=binding.binding_id,
            calibration_class=None,
        )
    
    if record.disposition == Admission.PRODUCTION and registry is not None:
        # Additional CCE check: drift gate eligibility
        drift_gate = get_drift_gate(registry)
        
        # Check all wings in the graph
        for ref in graph.wings:
            if not drift_gate.is_eligible(ref):
                state = drift_gate._capabilities.get(ref.name)
                reason = f"Wing {ref.name} demoted: {state.demotion_reason if state else 'unknown'}"
                return AdmissionRecord(
                    disposition=Admission.SHADOW_ONLY,
                    graph_id=graph.graph_id,
                    binding_id=binding.binding_id,
                    calibration_class=None,
                )
    
    return record


if __name__ == "__main__":
    # Quick test
    from runtime.registry import InMemoryRegistry
    from runtime.abi import WingRef
    from runtime.shadow import ShadowRouter
    from runtime.drift import DriftGate, Tier
    
    registry = InMemoryRegistry()
    pipeline = CCEPromotionPipeline(registry)
    
    # Register protected capabilities
    ref_alpha = WingRef("alpha", "1.0.0")
    ref_beta = WingRef("beta", "1.0.0")
    pipeline.register_protected_capabilities(
        [ref_alpha, ref_beta],
        {"alpha": 0.95, "beta": 0.93},
        tier=Tier.B
    )
    
    # Register new capability
    ref_gamma = WingRef("gamma", "1.0.0")
    pipeline.register_new_capability(ref_gamma, tier=Tier.B, origin_performance=0.90)
    
    print("CCE Promotion Pipeline initialized")
    print(f"Registered capabilities: {list(pipeline._drift_gate._capabilities.keys())}")
    print(f"Shadow router: {pipeline._shadow_router}")
    print(f"Drift gate: {pipeline._drift_gate}")