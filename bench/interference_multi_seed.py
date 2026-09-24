#!/usr/bin/env python3
"""Run interference probe across multiple seeds and compute confidence intervals."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SeedResult:
    seed: int
    target_delta: dict[str, float]
    mean_abs_interf: dict[str, float]
    max_interf: dict[str, float]
    min_interf: dict[str, float]
    interference: dict[str, dict[str, float]]


def run_single_seed(target: str, seed: int, samples: int, methods: list[str]) -> SeedResult | None:
    """Run one seed and parse JSON output."""
    out_file = f"interference_seed{seed}.json"
    cmd = [
        sys.executable, "-m", "bench.interference_probe",
        "--target", target,
        "--methods", *methods,
        "--samples", str(samples),
        "--seed", str(seed),
        "--output", out_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Seed {seed} failed: {result.stderr}")
        return None
    
    with open(out_file) as f:
        data = json.load(f)
    
    target_delta = {}
    mean_abs_interf = {}
    max_interf = {}
    min_interf = {}
    interference = {}
    
    for r in data:
        method = r["method"]
        target_delta[method] = r["target_delta"]
        mean_abs_interf[method] = r["mean_abs_interference"]
        max_interf[method] = r["max_interference"]
        deltas = list(r["interference_scores"].values())
        min_interf[method] = min(deltas) if deltas else 0.0
        interference[method] = r["interference_scores"]
    
    return SeedResult(
        seed=seed,
        target_delta=target_delta,
        mean_abs_interf=mean_abs_interf,
        max_interf=max_interf,
        min_interf=min_interf,
        interference=interference,
    )


def compute_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lower, upper) confidence interval."""
    if len(values) < 2:
        return values[0], values[0], values[0]
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    # t-distribution for small samples
    from scipy import stats
    t = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    margin = t * sem
    return mean, mean - margin, mean + margin


def main():
    target = "gamma"
    methods = ["UMNA", "LoRA", "Adapter", "FullFT"]
    samples = 500
    seeds = list(range(42, 42 + 10))  # 42, 43, ..., 51
    
    print(f"Running {len(seeds)} seeds for target={target}, samples={samples}")
    print(f"Methods: {methods}")
    print("=" * 60)
    
    results = []
    for seed in seeds:
        print(f"\nSeed {seed}...", end=" ", flush=True)
        r = run_single_seed(target, seed, samples, methods)
        if r:
            results.append(r)
            print(f"done (target Δ: {r.target_delta['UMNA']:+.4f})")
        else:
            print("FAILED")
    
    if not results:
        print("No successful runs!")
        return 1
    
    print("\n" + "=" * 80)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 80)
    
    for metric_name, metric_dict in [
        ("Target Δ (accuracy gain)", "target_delta"),
        ("Mean |Δ| (mean absolute interference)", "mean_abs_interf"),
        ("Max Δ (worst interference)", "max_interf"),
        ("Min Δ (best interference)", "min_interf"),
    ]:
        print(f"\n{metric_name}:")
        print(f"{'Method':<12} | {'Mean':>10} | {'95% CI Lower':>12} | {'95% CI Upper':>12}")
        print("-" * 55)
        for method in methods:
            values = [getattr(r, metric_dict)[method] for r in results]
            mean, lower, upper = compute_ci(values)
            print(f"{method:<12} | {mean:>+10.4f} | {lower:>+12.4f} | {upper:>+12.4f}")
    
    # Per-wing interference CIs
    print("\n" + "=" * 80)
    print("PER-WING INTERFERENCE CIs (Δ = after - before accuracy)")
    print("=" * 80)
    
    protected_wings = ["alpha", "beta", "delta", "epsilon", "zeta"]
    for method in methods:
        print(f"\n{method}:")
        print(f"{'Wing':<10} | {'Mean Δ':>10} | {'95% CI Lower':>12} | {'95% CI Upper':>12}")
        print("-" * 52)
        for wing in protected_wings:
            values = [r.interference[method][wing] for r in results]
            mean, lower, upper = compute_ci(values)
            print(f"{wing:<10} | {mean:>+10.4f} | {lower:>+12.4f} | {upper:>+12.4f}")
    
    # Save all raw data
    out_data = {
        "target": target,
        "methods": methods,
        "samples": samples,
        "seeds": seeds,
        "results": [
            {
                "seed": r.seed,
                "target_delta": r.target_delta,
                "mean_abs_interf": r.mean_abs_interf,
                "max_interf": r.max_interf,
                "min_interf": r.min_interf,
                "interference": r.interference,
            }
            for r in results
        ],
    }
    Path("interference_multi_seed.json").write_text(json.dumps(out_data, indent=2))
    print("\nSaved raw data to interference_multi_seed.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())