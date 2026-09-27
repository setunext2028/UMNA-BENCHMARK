# wings/alpha, wings/beta, wings/gamma

Developer: Koushik Das - Version 0.1.0

S1 uses constant Wings so registry, ABI, dependency, and graph failures can
be isolated from model behavior. Each Wing owns `manifest.yaml`, `model.py`,
and `test_vectors/sample.yaml`. No Wing imports another Wing.

## Shared model structure

Each `model.py` loads its sibling manifest once at module import and keeps
one module-local immutable `Manifest`.

### manifest()
Returns that parsed manifest. Graph construction and registry installation
therefore read the same declaration as the on-disk package.

### input_types()
Returns `manifest.input_types` without a second source of type information.

### output_types()
Returns `manifest.output_types` without a second source of type information.

### execute(ctx, x)
- Ignores model content by design; S1 is a control-plane stage.
- Builds a `LatentSchema` in `ctx.core_epoch` with float32 data.
- Returns a `LatentPacket` whose value is always `0.0` and a `WingOutput`
  carrying exact Wing version and stub provenance.
- Real Core traversal and learned behavior replace this method in S3.

## AlphaWing

- Input: `Latent<Text>`
- Output: `Latent<Summary>`
- Output dimensions: 64
- Dependency: none
- Ceiling: 64 MB memory, 100000 input units
- Capability tag: `summarize`

## BetaWing

- Input: `Latent<Text>`
- Output: `Latent<Claim>`
- Output dimensions: 32
- Dependency: `alpha >=1.0.0,<2`
- Ceiling: 128 MB memory, 100000 input units
- Capability tag: `claim_extraction`

## GammaWing

- Input: `Latent<Claim>`
- Output: `Latent<Decision>`
- Output dimensions: 16
- Dependency: `beta >=1.0.0,<2`
- Ceiling: 128 MB memory, 100000 input units
- Capability tag: `decision`

## Test vectors

Each fixed vector supplies one text input and declares expected constant
output `0.0`. These vectors check package wiring, not scientific quality;
reporting their success as an experiment would violate rule 7.
