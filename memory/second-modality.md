# Second modality (Phase 5: Multimodal extension inside one Core epoch)

Implemented in `runtime/modality.py` with tests in `tests/test_modality.py`.

## Design decisions

1. **ImageEncoderWing** - Maps raw image features (Latent<Image>, 64-dim) to Core Latent<Vec8>:
   - Two-layer encoder: 64 -> 16 (ReLU) -> 8 (linear)
   - Core remains frozen; encoder is a Wing that plugs into the existing interface
   - Output type: Latent<Vec8> compatible with all downstream Wings

2. **ImageClassifierWing** - Operates on Latent<Vec8> for image classification:
   - Takes encoded latent (from ImageEncoderWing or any Vec8 source)
   - Linear head: 8 -> 1 with sigmoid
   - Output type: Latent<Score>

3. **ModalityRegistry** - Manages modalities and their components:
   - Registers modalities with raw dimension, encoder, supported tasks
   - Maintains encoder and classifier mappings
   - Global singleton via `get_modality_registry()`

4. **Core remains frozen** - Single epoch (1.x), no Core retraining needed:
   - New modality Wings map their inputs to existing Latent<Vec8> interface
   - Downstream Wings (classifiers, Bridges, etc.) work unchanged
   - This is "Multimodal extension inside one Core epoch" (Phase 5)

## Key invariants

- Core epoch 1.x never changes for second modality support
- All new modality Wings output Latent<Vec8> for Core compatibility
- Typed interfaces ensure encoder output matches classifier input
- ModalityRegistry provides discoverability of available modalities

## Integration points

- Planner can retrieve image encoder via typed retrieval (Latent<Image>)
- Encoder + classifier form a typed graph: Image -> Vec8 -> Score
- Bridges can fuse image modality outputs with other modalities
- Admission and calibration work on the composed graph

## Testing

13 tests covering:
- ImageEncoderWing: manifest, types, single/batch execute
- ImageClassifierWing: manifest, execute
- ModalityRegistry: default image modality, encoder/classifier management
- Encoder + classifier pipeline integration
- Global registry singleton