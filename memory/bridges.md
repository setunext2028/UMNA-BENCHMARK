# Bridges (fusion operators)

Implemented in `runtime/bridge.py` with tests in `tests/test_bridge.py`.

## Design decisions

1. **GenericBridge** - O(1) fallback for common composition classes:
   - Concatenates input latents along feature dimension
   - Projects to output dimension via learned projection matrix
   - No training required; works for any pair of compatible wings
   - Cached in BridgeRegistry by input dimensions

2. **TrainedBridge** - Specialized fusion for high-value interactions:
   - Trained on specific Wing output pairs -> target latent
   - Gradient descent on projection matrix (MSE loss)
   - Has its own manifest, version, and calibration binding
   - Created lazily on observed route demand (Section 19.1)

3. **BridgeRegistry** - Manages all fusion operators:
   - Registers both generic and trained bridges
   - Caches generic bridges by input dimension tuple
   - Prevents O(N²) artifact explosion by using generic fallback

4. **Bridge Protocol** - Compatible with typed capability graph:
   - Has manifest with input/output types
   - Executes with CoreContext like Wings
   - Returns WingOutput with evidence packets
   - Can participate in calibration classes

## Key invariants

- Generic bridge is always available as fallback (no O(N²) pre-materialization)
- Trained bridges are created only for validated high-value pairs
- Bridge outputs are typed LatentPackets with provenance
- Bridge execution goes through same validation pipeline as Wings

## Integration points

- Planner can include Bridges in typed graphs
- Admission checks Bridge calibration bindings
- Execution runs Bridge.fuse() in the fusion phase
- Risk layer calibrates Bridge outputs like Wing outputs

## Testing

13 tests covering:
- GenericBridge: manifest, input/output types, single/batch execute
- TrainedBridge: manifest, execute, wrong input count validation
- BridgeRegistry: register/get, duplicate prevention, generic cache
- Global registry singleton