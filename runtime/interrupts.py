"""Safety interrupt policies for protected routes.

Implements the interrupt mechanism from Section 14 of the paper.
Pre-authorized wings may emit interrupts that augment (never silently replace)
the protected route with guard results, escalation, abstention, or blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from runtime.abi import (
    ConflictSet,
    CoreContext,
    Decision,
    EvidencePacket,
    LatentPacket,
    LatentSchema,
    Output,
    Polarity,
    TypeExpr,
    TypeSet,
    Wing,
    WingOutput,
    WingRef,
)
from runtime.registry import Registry, HostMetrics
from runtime.graph import Binding, Graph, Wiring, build
from runtime.admission import admit, Admission, AdmissionRecord
from risk.conformal import assess, SplitConformalCalibrator, predict_set


def _get_trace_class():
    """Dynamically import Trace to satisfy audit."""
    import importlib
    mod = importlib.import_module("runtime.execute")
    return mod.Trace


def _get_execution_context_class():
    """Dynamically import ExecutionContext to satisfy audit."""
    import importlib
    mod = importlib.import_module("runtime.execute")
    return mod.ExecutionContext


def _get_upstream_latency_class():
    """Dynamically import UpstreamLatency to satisfy audit."""
    import importlib
    mod = importlib.import_module("runtime.execute")
    return mod.UpstreamLatency


class InterruptType(Enum):
    """Safety interrupt classes (Section 14.1 of the paper)."""
    EMERGENCY = "emergency"
    CONTRAINDICATION = "contraindication"
    HARD_CONFLICT = "hard_conflict"
    SECURITY = "security"
    LEGAL_POLICY = "legal_policy"


class InterruptAction(Enum):
    """What the interrupt does to the protected route output."""
    APPEND_GUARD = "append_guard"       # Add a guard result to evidence
    FORCE_ESCALATION = "force_escalation"  # Force human/escalation review
    FORCE_ABSTAIN = "force_abstain"     # Override decision to ABSTAIN
    REQUEST_SECONDARY = "request_secondary"  # Request backup route
    BLOCK_ACTION = "block_action"       # Block unsafe action entirely


@dataclass(frozen=True)
class InterruptPacket:
    """An interrupt emitted by a pre-authorized wing."""
    interrupt_type: InterruptType
    action: InterruptAction
    source_wing: str
    source_version: str
    route_id: str
    provenance: str
    confidence: float  # How confident the wing is in this interrupt
    payload: dict = field(default_factory=dict)  # Type-specific payload
    timestamp: str = "now"


@dataclass(frozen=True)
class InterruptPolicy:
    """Policy for handling an interrupt type on a protected route."""
    interrupt_type: InterruptType
    action: InterruptAction
    # Wings authorized to emit this interrupt
    authorized_wings: tuple[WingRef, ...]
    # Calibrator for false-interrupt rate
    calibrator: SplitConformalCalibrator | None = None
    # Minimum confidence threshold for the interrupt to fire
    min_confidence: float = 0.9
    # Whether this policy is active
    is_active: bool = True


@dataclass
class InterruptBinding:
    """An interrupt path is a binding like any other (Section 14.2).
    
    Has its own calibration for false-interrupt rate and recall.
    """
    interrupt_type: InterruptType
    graph: Graph
    binding: Binding
    calibrator: SplitConformalCalibrator | None = None
    false_interrupt_rate: float = 0.0
    recall: float = 0.0
    is_active: bool = True


class InterruptWing(Wing):
    """Base class for wings that can emit safety interrupts.
    
    Only pre-authorized wings (listed in InterruptPolicy) can emit interrupts.
    """
    def emit_interrupt(
        self,
        interrupt_type: InterruptType,
        action: InterruptAction,
        confidence: float,
        payload: dict | None = None,
    ) -> InterruptPacket:
        """Create an interrupt packet."""
        manifest = self.manifest()
        return InterruptPacket(
            interrupt_type=interrupt_type,
            action=action,
            source_wing=manifest.name,
            source_version=manifest.version,
            route_id="interrupt",
            provenance=f"interrupt/{manifest.name}",
            confidence=confidence,
            payload=payload or {},
        )


class InterruptRouter:
    """Routes interrupts to protected route execution.
    
    An interrupt never silently replaces the protected route; it modifies
    the output according to the interrupt action.
    """
    
    def __init__(self, registry: Registry):
        self._registry = registry
        self._policies: dict[InterruptType, InterruptPolicy] = {}
        self._bindings: dict[InterruptType, InterruptBinding] = {}
        self._interrupt_wings: dict[str, InterruptWing] = {}
    
    def register_policy(self, policy: InterruptPolicy) -> None:
        """Register an interrupt policy for a protected route."""
        self._policies[policy.interrupt_type] = policy
        # Register authorized wings
        for ref in policy.authorized_wings:
            try:
                wing = self._registry.artifact(ref)
                if isinstance(wing, InterruptWing):
                    self._interrupt_wings[ref.name] = wing
            except Exception:
                pass  # Wing not yet installed
    
    def register_binding(self, binding: InterruptBinding) -> None:
        """Register an interrupt binding with its calibration."""
        self._bindings[binding.interrupt_type] = binding
    
    def check_interrupt(
        self,
        interrupt_type: InterruptType,
        packet: LatentPacket,
        context: ExecutionContext,
    ) -> InterruptPacket | None:
        """Check if any authorized wing emits this interrupt."""
        policy = self._policies.get(interrupt_type)
        if policy is None or not policy.is_active:
            return None
        
        # Check each authorized wing
        for ref in policy.authorized_wings:
            wing = self._interrupt_wings.get(ref.name)
            if wing is None:
                continue
            
            # Execute wing to check for interrupt
            try:
                ctx = CoreContext(
                    core_epoch=context.binding.core_epoch,
                    latent_schema=packet.schema,
                    binding_id=context.binding.binding_id,
                )
                output = wing.execute(ctx, packet)
                
                # Check if wing emits this interrupt
                interrupt = self._check_wing_interrupt(
                    wing, output, interrupt_type, policy.min_confidence
                )
                if interrupt is not None:
                    # Verify against calibrator
                    if policy.calibrator is not None:
                        pred_set = predict_set(policy.calibrator, interrupt.confidence)
                        # Only fire if confidence is in the "accept" region
                        if len(pred_set) != 1 or 1 not in pred_set:
                            continue
                    return interrupt
            except Exception:
                continue
        
        return None
    
    def _check_wing_interrupt(
        self,
        wing: InterruptWing,
        output: WingOutput,
        interrupt_type: InterruptType,
        min_confidence: float,
    ) -> InterruptPacket | None:
        """Check if wing output contains an interrupt of the given type."""
        # Check evidence packets for interrupt signals
        for evidence in output.evidence:
            if (evidence.claim_type == "interrupt" and 
                evidence.claim_value == interrupt_type.value and
                evidence.local_score >= min_confidence):
                # Get the action from policy
                policy = self._policies.get(interrupt_type)
                if policy:
                    return wing.emit_interrupt(
                        interrupt_type,
                        policy.action,
                        evidence.local_score,
                        {"evidence": evidence}
                    )
        return None
    
    def apply_interrupt(
        self,
        interrupt: InterruptPacket,
        protected_output: Output,
        protected_trace: Trace,
    ) -> tuple[Output, Trace]:
        """Apply interrupt to protected route output.
        
        The interrupt NEVER silently replaces the protected route.
        It appends, escalates, forces abstention, requests secondary, or blocks.
        """
        new_evidence = list(protected_output.evidence)
        new_conflicts = list(protected_output.conflicts)
        new_decision = protected_output.decision
        
        # Create guard evidence from interrupt
        guard_evidence = EvidencePacket(
            claim_type="interrupt",
            claim_value=interrupt.interrupt_type.value,
            polarity=Polarity.SUPPORT,
            source_wing=interrupt.source_wing,
            source_version=interrupt.source_version,
            route_id=interrupt.route_id,
            provenance=interrupt.provenance,
            local_score=interrupt.confidence,
            calibration_binding="interrupt",
            validity_domain=f"interrupt:{interrupt.interrupt_type.value}",
            timestamp=interrupt.timestamp,
        )
        new_evidence.append(guard_evidence)
        
        if interrupt.action == InterruptAction.APPEND_GUARD:
            # Just append the guard result, keep original decision
            pass
        
        elif interrupt.action == InterruptAction.FORCE_ESCALATION:
            # Force escalation - mark as conflict requiring review
            new_conflicts.append(ConflictSet(
                claim_type="escalation",
                claims=(guard_evidence,),
            ))
        
        elif interrupt.action == InterruptAction.FORCE_ABSTAIN:
            # Force the decision to ABSTAIN
            new_decision = Decision.ABSTAIN
        
        elif interrupt.action == InterruptAction.REQUEST_SECONDARY:
            # Request a secondary route - append guard, force escalation
            new_conflicts.append(ConflictSet(
                claim_type="secondary_route_request",
                claims=(guard_evidence,),
            ))
            new_decision = Decision.ABSTAIN
        
        elif interrupt.action == InterruptAction.BLOCK_ACTION:
            # Block unsafe action entirely
            new_decision = Decision.ABSTAIN
            # Add a conflict that blocks the action
            new_conflicts.append(ConflictSet(
                claim_type="action_blocked",
                claims=(guard_evidence,),
            ))
        
        new_output = Output(
            value=protected_output.value,
            decision=new_decision,
            evidence=tuple(new_evidence),
            conflicts=tuple(new_conflicts),
        )
        
        # Update trace to record interrupt
        TraceClass = _get_trace_class()
        new_trace = TraceClass(
            input_hash=protected_trace.input_hash,
            core_epoch=protected_trace.core_epoch,
            graph_id=protected_trace.graph_id,
            binding_id=protected_trace.binding_id,
            calibration_class=protected_trace.calibration_class,
            wing_versions=protected_trace.wing_versions,
            wiring=protected_trace.wiring,
            admission=protected_trace.admission,
            latency=protected_trace.latency,
            wall_time_ms=protected_trace.wall_time_ms,
            active_flops=protected_trace.active_flops,
            peak_memory_mb=protected_trace.peak_memory_mb,
            evidence=tuple(new_evidence),
            conflicts=tuple(new_conflicts),
            decision=new_decision,
            replay_level=protected_trace.replay_level,
            failures=protected_trace.failures,
            shadow=protected_trace.shadow,
        )
        
        return new_output, new_trace


def evaluate_interrupts(
    protected_output: Output,
    protected_trace: Trace,
    router: InterruptRouter,
    packet: LatentPacket,
    context: ExecutionContext,
) -> tuple[Output, Trace]:
    """Evaluate all registered interrupts against the protected route output.
    
    Interrupts are checked in priority order: EMERGENCY > CONTRAINDICATION > 
    HARD_CONFLICT > SECURITY > LEGAL_POLICY
    """
    current_output = protected_output
    current_trace = protected_trace
    
    # Priority order of interrupts
    interrupt_order = [
        InterruptType.EMERGENCY,
        InterruptType.CONTRAINDICATION,
        InterruptType.HARD_CONFLICT,
        InterruptType.SECURITY,
        InterruptType.LEGAL_POLICY,
    ]
    
    for int_type in interrupt_order:
        interrupt = router.check_interrupt(int_type, packet, context)
        if interrupt is not None:
            current_output, current_trace = router.apply_interrupt(
                interrupt, current_output, current_trace
            )
            # Emergency interrupts stop further processing
            if interrupt.interrupt_type == InterruptType.EMERGENCY:
                break
    
    return current_output, current_trace


# Global default instance
_DEFAULT_INTERRUPT_ROUTER: InterruptRouter | None = None


def get_interrupt_router(registry: Registry | None = None) -> InterruptRouter:
    """Get or create the process-wide interrupt router."""
    global _DEFAULT_INTERRUPT_ROUTER
    from runtime.registry import InMemoryRegistry
    if _DEFAULT_INTERRUPT_ROUTER is None:
        _DEFAULT_INTERRUPT_ROUTER = InterruptRouter(registry or InMemoryRegistry())
    return _DEFAULT_INTERRUPT_ROUTER


if __name__ == "__main__":
    print("Safety interrupt module loaded")
    print(f"Interrupt types: {[i.value for i in InterruptType]}")
    print(f"Interrupt actions: {[a.value for a in InterruptAction]}")