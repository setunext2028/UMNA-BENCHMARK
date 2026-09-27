# Cumulative drift gating

Implemented in `runtime/drift.py` with tests in `tests/test_drift.py`.

## Design decisions

1. **Drift constraints** - Two hard constraints per capability:
   - Step drift: `P(t) - P(t-1) >= -epsilon_i` (per-change non-inferiority)
   - Origin drift: `P(t) - P(0) >= -E_i` (cumulative budget against pinned origin)
   - Both must pass for promotion. Origin performance P(0) is permanently pinned.

2. **Protection tiers** - Three tiers with different statistical policies:
   - Tier A (critical): Strict non-inferiority, family-wise error control, permanently pinned origin
   - Tier B (important): Non-inferiority under less conservative policy, scheduled full regression
   - Tier C (exploratory): FDR control; promotion only after sufficient evidence

3. **Power budget** - `|A|max` computed BEFORE first promotion:
   - Largest k such that `power(delta, n, alpha/k) >= 1 - beta`
   - Uses normal approximation for one-sided z-test
   - If more Tier A capabilities designated than `|A|max`, tests are underpowered

4. **Interference matrix** - Tracks promotion impact on other capabilities:
   - Records `delta = P_i(after) - P_i(before)` for each promotion event
   - Only measured on routes changed by promotion

5. **Demotion** - Capabilities can be demoted from routing eligibility:
   - Triggered by drift alarms (step or origin)
   - Demoted capabilities excluded from routing
   - Restoration requires full re-evaluation

6. **Global default instance** - `get_drift_gate()` provides process-wide default

## Key invariants

- Origin performance P(0) is NEVER re-snapshotted by promotion
- Demotion is from routing eligibility only; artifacts remain for shadow evaluation
- Power budget computed once before first promotion; cached thereafter
- Unknown capabilities default to eligible (fail-open for new wings)

## Integration points

- Planner should check `drift_gate.is_eligible(ref)` before including in graph
- Promotion workflow: shadow eval → update_performance → can_promote → record_promotion
- Demotion workflow: drift alarm → demote → audit log → exclude from routing

## Testing

31 tests covering:
- DriftConfig and TierConfig defaults per tier
- Registration, origin pinning, performance updates
- Step drift, origin drift, both, and pass cases
- Promotion gate (both constraints must pass)
- Demotion and restoration
- Interference matrix recording
- Tier A power budget computation
- Floating-point tolerance handling