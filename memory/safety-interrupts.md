# Safety interrupts

Implemented in `runtime/interrupts.py` with tests in `tests/test_interrupts.py`.

## Design decisions

1. **Interrupt types** (Section 14.1 of the paper):
   - EMERGENCY: Critical safety situation requiring immediate action
   - CONTRAINDICATION: Medical/technical contraindication detected
   - HARD_CONFLICT: Irreconcilable conflict between capabilities
   - SECURITY: Security policy violation
   - LEGAL_POLICY: Legal/compliance policy violation

2. **Interrupt actions**:
   - APPEND_GUARD: Add guard result to evidence, keep original decision
   - FORCE_ESCALATION: Mark as conflict requiring review
   - FORCE_ABSTAIN: Override decision to ABSTAIN
   - REQUEST_SECONDARY: Request backup route, force ABSTAIN
   - BLOCK_ACTION: Block unsafe action entirely, force ABSTAIN

3. **Pre-authorization** - Only wings listed in `InterruptPolicy.authorized_wings` can emit interrupts. This prevents arbitrary wings from interrupting protected routes.

4. **Interrupt bindings are calibrated** (Section 14.2) - Each interrupt type has its own binding with calibrator for false-interrupt rate and recall. The false-interrupt rate and recall are separately calibrated and gated.

5. **Interrupts never silently replace the protected route** - They always append to evidence, add conflicts, or modify the decision. The original protected route output value is preserved.

6. **Priority order** - EMERGENCY > CONTRAINDICATION > HARD_CONFLICT > SECURITY > LEGAL_POLICY. Emergency interrupts stop further processing.

7. **Calibrator verification** - Before firing, the interrupt confidence is checked against the policy's calibrator using split conformal prediction. Only interrupts with confidence in the "accept" region fire.

## Key invariants

- Protected route output value is NEVER replaced
- Only pre-authorized wings can emit interrupts
- Interrupt paths have their own calibration (false-interrupt rate, recall)
- Interrupts are evaluated after protected route execution, as a post-processing step
- Emergency interrupts halt further interrupt evaluation

## Integration points

- Called via `evaluate_interrupts()` after protected route execution
- Uses `InterruptRouter` to check and apply interrupts
- Works with existing `Trace` and `Output` types
- Calibrators use split conformal from `risk.conformal`

## Testing

16 tests covering:
- All 5 interrupt types and 5 actions
- Pre-authorization enforcement
- Calibrator-based confidence thresholding
- Each interrupt action (append_guard, force_escalation, force_abstain, request_secondary, block_action)
- Priority order evaluation
- Protected route preservation (value and evidence)
- Global default router instance