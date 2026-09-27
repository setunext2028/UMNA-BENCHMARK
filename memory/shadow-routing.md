# Shadow routing and router learning

Implemented in `runtime/shadow.py` with tests in `tests/test_shadow.py`.

## Design decisions

1. **Shadow execution bypasses admission** - The `run_shadow` function creates a `SHADOW_ONLY` admission record and executes the graph without checking calibration class membership. This allows collecting counterfactual data for graphs that would otherwise be rejected.

2. **Router learns from shadow traces** - The `ShadowRouter` collects `ShadowTrace` objects containing the input packet, graph, execution trace, and original admission record. The `RouterModel` extracts features from these and trains a linear model to predict utility.

3. **Feature extraction** - Router features include:
   - Input observables (mean, std, max_abs, length)
   - Graph structure (num wings, has wiring, num edges)
   - Aggregate wing metrics (latency, memory, reliability, FLOPs statistics)

4. **Promotion gate** - `should_promote` requires ≥10 shadow executions with >80% accept rate. This prevents promoting graphs based on insufficient evidence.

5. **Global default router** - `get_shadow_router` provides a process-wide default instance, similar to the registry pattern.

## Key invariants

- Shadow execution must never serve production traffic (enforced by `shadow=True` in trace)
- Router training only uses host-measured features, never vendor-declared ones
- Promotion is evidence-gated, not utility-gated

## Integration points

- `bench/harness.py` and `bench/synergy_probe.py` could use shadow routing for data collection
- Future: router could inform planner arm selection (Mode B pair proposals)

## Testing

10 tests covering:
- Shadow execution produces valid traces with `shadow=True`
- Router initialization, trace collection, training
- Graph scoring and selection for shadow
- Promotion gate with ABSTAIN decisions (should not promote)
- Promotion gate with ACCEPT decisions (should promote)