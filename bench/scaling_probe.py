"""S7 Scaling Probe: Grow registry with proxy/metadata-only entries. Measure
retrieval recall and latency as N grows.
"""

from __future__ import annotations

import time
import numpy as np

from runtime.abi import TypeExpr, Manifest, TypeSet, Dependency, ResourceCeilings, ManifestIntegrity, ManifestProvenance
from runtime.registry import InMemoryRegistry, typed_retrieval
from runtime.graph import build, Wiring
from wings.alpha.model import AlphaWing


TARGET_TYPES = [
    TypeExpr("Latent", ("Vec8",)),
    TypeExpr("Latent", ("Vec4",)),
    TypeExpr("Latent", ("Vec2",)),
    TypeExpr("Latent", ("Image",)),
    TypeExpr("Latent", ("Text",)),
    TypeExpr("Latent", ("Audio",)),
    TypeExpr("Latent", ("Tabular",)),
]

REGISTRY_SIZES = [10, 50, 100, 200, 500, 1000, 2000, 5000]
M_VALUES = [1, 5, 10, 20, 50]
N_TRIALS = 100


def _make_proxy_manifest(name: str, input_type: TypeExpr, output_type: TypeExpr) -> Manifest:
    """Create a proxy wing manifest (metadata only, no artifact)."""
    return Manifest(
        name=name,
        version="1.0.0",
        core_epoch="1.x",
        author="proxy",
        input_types=TypeSet((input_type,)),
        output_types=TypeSet((output_type,)),
        capabilities=(f"proxy-{input_type.family.lower()}",),
        dependencies=(),
        resource_ceilings=ResourceCeilings(max_memory_mb=8, max_input_size=10000),
        integrity=ManifestIntegrity(artifact_hash="0" * 64),
        provenance=ManifestProvenance(
            training_dataset_ids=("proxy",),
            training_code_commit="proxy",
        ),
    )


def _build_registry(n_wings: int) -> InMemoryRegistry:
    """Build a registry with n_wings proxy entries + 6 real wings."""
    registry = InMemoryRegistry(core_epoch="1.x")

    # Add 6 real wings first
    for wing_class in [AlphaWing]:
        wing = wing_class()
        ref = registry.load(wing.manifest(), wing)
        # We don't set metrics for real wings in this probe

    # Add proxy wings
    n_proxies = n_wings - 6
    for i in range(n_proxies):
        input_type = TARGET_TYPES[i % len(TARGET_TYPES)]
        output_type = TypeExpr("Latent", ("Score",))
        manifest = _make_proxy_manifest(f"proxy_{i}", input_type, output_type)
        registry.load(manifest)

    return registry


def _measure_retrieval_latency(registry: InMemoryRegistry, query_type: TypeExpr, m: int, n_trials: int) -> float:
    """Measure average retrieval latency in microseconds."""
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter_ns()
        typed_retrieval(query_type, registry, m)
        latencies.append((time.perf_counter_ns() - start) / 1000.0)  # microseconds
    return float(np.mean(latencies))


def _measure_recall(registry: InMemoryRegistry, query_type: TypeExpr, m: int, n_trials: int) -> float:
    """Measure recall@m: fraction of relevant wings retrieved."""
    # Get all wings in registry matching the query type
    all_matching = []
    for ref in registry.installed():
        manifest = registry._installed[ref.name][ref.version]
        if any(
            t.family == query_type.family and (not query_type.args or t.args == query_type.args)
            for t in manifest.input_types
        ):
            all_matching.append(ref)

    if not all_matching:
        return 1.0  # Vacuously true if no matching wings

    hits = 0
    for _ in range(n_trials):
        retrieved = typed_retrieval(query_type, registry, m)
        if retrieved:
            # Check if any retrieved wing is relevant
            retrieved_names = {r.name for r in retrieved}
            matching_names = {r.name for r in all_matching}
            if retrieved_names & matching_names:
                hits += 1
    return hits / n_trials


def run_scaling_probe() -> None:
    """Run the S7 scaling probe experiment."""
    print("=" * 80)
    print("UMNA S7 SCALING PROBE")
    print("=" * 80)
    print(f"Registry sizes: {REGISTRY_SIZES}")
    print(f"M values: {M_VALUES}")
    print(f"Trials per measurement: {N_TRIALS}")
    print("=" * 80)

    results = {}

    for n_wings in REGISTRY_SIZES:
        print(f"\nBuilding registry with N={n_wings}...")
        registry = _build_registry(n_wings)

        results[n_wings] = {}

        for m in M_VALUES:
            if m > n_wings:
                continue

            print(f"  M={m}: ", end="", flush=True)

            latencies = []
            recalls = []

            for query_type in TARGET_TYPES:
                latency = _measure_retrieval_latency(registry, query_type, m, N_TRIALS)
                recall = _measure_recall(registry, query_type, m, N_TRIALS)
                latencies.append(latency)
                recalls.append(recall)

            avg_latency = float(np.mean(latencies))
            avg_recall = float(np.mean(recalls))

            results[n_wings][m] = {
                "latency_us": avg_latency,
                "recall": avg_recall,
                "per_type_latency": {str(qt): latencies[i] for i, qt in enumerate(TARGET_TYPES)},
                "per_type_recall": {str(qt): recalls[i] for i, qt in enumerate(TARGET_TYPES)},
            }

            print(f"latency={avg_latency:.1f}us, recall={avg_recall:.3f}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SCALING PROBE RESULTS SUMMARY")
    print("=" * 80)

    # Latency table
    print("\nRETRIEVAL LATENCY (microseconds):")
    header = f"{'N\\M':>8}"
    for m in M_VALUES:
        header += f"  {f'M={m}':>10}"
    print(header)
    print("-" * (8 + 13 * len(M_VALUES)))

    for n_wings in REGISTRY_SIZES:
        row = f"{n_wings:>8}"
        for m in M_VALUES:
            if m <= n_wings and m in results[n_wings]:
                row += f"  {results[n_wings][m]['latency_us']:>10.1f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)

    # Recall table
    print("\nRECALL@M:")
    header = f"{'N\\M':>8}"
    for m in M_VALUES:
        header += f"  {f'M={m}':>10}"
    print(header)
    print("-" * (8 + 13 * len(M_VALUES)))

    for n_wings in REGISTRY_SIZES:
        row = f"{n_wings:>8}"
        for m in M_VALUES:
            if m <= n_wings and m in results[n_wings]:
                row += f"  {results[n_wings][m]['recall']:>10.3f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)

    # Scaling analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    for m in M_VALUES:
        if m > REGISTRY_SIZES[0]:
            continue
        sizes = [n for n in REGISTRY_SIZES if m <= n]
        latencies = [results[n][m]['latency_us'] for n in sizes]
        # Fit linear scaling
        if len(sizes) > 2:
            coeff = np.polyfit(sizes, latencies, 1)
            print(f"M={m}: latency ~ {coeff[0]:.4f} * N + {coeff[1]:.1f} (slope = {coeff[0]:.4f} us/entry)")

    print("=" * 80)


if __name__ == "__main__":
    run_scaling_probe()