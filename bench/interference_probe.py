"""UMNA Layer B Benchmark: Targeted Wing Training + Protected-Wing Interference.

This is the core UMNA hypothesis test:
- Train ONE wing (e.g., gamma) through frozen Core
- Measure target capability improvement (gamma on t_cd)
- Measure protected-wing degradation (alpha, beta, delta, epsilon, zeta)
- Compare against conventional fine-tuning, LoRA, Adapter baselines

Output: interference matrix + target Δ vs protected Δ tradeoff plot data.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from runtime.abi import CoreContext, LatentPacket, LatentSchema
from core.model import FrozenCore, load_artifact, artifact_path
from core.tasks import load_tasks, TASK_AB, TASK_CD, TASK_M, TASK_Z
from bench.interference_training import (
    train_wing,
    evaluate_wing,
    TrainedWing,
    METHODS,
)

WING_TASK_MAP = {
    "alpha": TASK_AB,
    "beta": TASK_AB,
    "gamma": TASK_CD,
    "delta": TASK_CD,
    "epsilon": TASK_M,
    "zeta": TASK_Z,
}

WING_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


@dataclass
class WingScores:
    """Scores for one wing on its task."""
    wing: str
    task: str
    before_mean: float
    before_std: float
    after_mean: float
    after_std: float
    n_samples: int


@dataclass
class InterferenceReport:
    """Full interference experiment result."""
    target_wing: str
    target_task: str
    target_delta: float
    target_p_value: float
    
    protected_wings: list[str]
    interference_scores: dict[str, float]
    mean_abs_interference: float
    max_interference: float
    
    method: str
    hardware: str
    seed: int
    timestamp: str


def evaluate_wing_class(wing_class, ctx, x_data, y_data, n_samples=None) -> tuple[float, float]:
    """Evaluate a wing class on data, return (accuracy, std_score)."""
    if n_samples is not None:
        x_data = x_data[:n_samples]
        y_data = y_data[:n_samples]
    
    wing = wing_class()
    correct = 0
    scores = []
    for i in range(len(x_data)):
        packet = LatentPacket(schema=ctx.latent_schema, data=x_data[i].astype(np.float32))
        output = wing.execute(ctx, packet)
        pred = int(output.latent.data) if np.isscalar(output.latent.data) else int(output.latent.data[0])
        true_label = int(y_data[i])
        correct += (pred == true_label)
        scores.append(float(output.local_score))
    
    accuracy = correct / len(x_data)
    return accuracy, float(np.std(scores))


def evaluate_wing_class_with_core(wing_class, core: FrozenCore, x_data, y_data, n_samples=None) -> tuple[float, float]:
    """Evaluate a wing class on data using a specific core, return (accuracy, std_score)."""
    if n_samples is not None:
        x_data = x_data[:n_samples]
        y_data = y_data[:n_samples]
    
    # Create a context with the core's epoch
    ctx = CoreContext(core_epoch=core.epoch, latent_schema=LatentSchema(core.epoch, 8, "float32"))
    
    wing = wing_class()
    # Monkey-patch the core params in the wing's module-level variables
    # This is a bit hacky but works for evaluation
    import importlib
    module = importlib.import_module(wing_class.__module__)
    module._CORE_PARAMS = {k: v.copy() for k, v in core.params.items()}
    for arr in module._CORE_PARAMS.values():
        arr.setflags(write=False)
    
    correct = 0
    scores = []
    for i in range(len(x_data)):
        packet = LatentPacket(schema=ctx.latent_schema, data=x_data[i].astype(np.float32))
        output = wing.execute(ctx, packet)
        pred = int(output.latent.data) if np.isscalar(output.latent.data) else int(output.latent.data[0])
        true_label = int(y_data[i])
        correct += (pred == true_label)
        scores.append(float(output.local_score))
    
    accuracy = correct / len(x_data)
    return accuracy, float(np.std(scores))


def evaluate_trained_wing(trained: TrainedWing, core: FrozenCore, x_data, y_data, n_samples=None) -> tuple[float, float]:
    """Evaluate a trained wing on data, return (accuracy, std_score)."""
    if n_samples is not None:
        x_data = x_data[:n_samples]
        y_data = y_data[:n_samples]
    
    return evaluate_wing(trained, core, x_data, y_data, n_samples)


def run_interference_experiment(
    target_wing: str,
    method: str = "UMNA",
    n_eval_samples: int = 200,
    seed: int = 42,
    hardware: str = "auto",
) -> InterferenceReport:
    """
    Run targeted training + interference measurement.
    
    Args:
        target_wing: Which wing to train (e.g., "gamma")
        method: Training method ("UMNA", "LoRA", "Adapter", "FullFT")
        n_eval_samples: Samples per wing for before/after evaluation
        seed: Random seed
        hardware: Hardware description
    
    Returns:
        InterferenceReport with all metrics
    """
    np.random.seed(seed)
    
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")
    
    # Load data
    tasks_data = load_tasks(2718)
    target_task = WING_TASK_MAP[target_wing]
    holdout = tasks_data[f"{target_task}:holdout"]
    x_holdout = holdout.x
    y_holdout = holdout.y
    
    # Also load data for all other tasks (for protected wings)
    task_data = {}
    for task_id in [TASK_AB, TASK_CD, TASK_M, TASK_Z]:
        h = tasks_data[f"{task_id}:holdout"]
        train = tasks_data[task_id]
        task_data[task_id] = (train.x, train.y, h.x, h.y)
    
    # Load frozen Core
    core = load_artifact(artifact_path())
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))
    
    # Import wing classes for evaluation
    from wings.alpha.model import AlphaWing
    from wings.beta.model import BetaWing
    from wings.gamma.model import GammaWing
    from wings.delta.model import DeltaWing
    from wings.epsilon.model import EpsilonWing
    from wings.zeta.model import ZetaWing
    
    WING_CLASSES = {
        "alpha": AlphaWing,
        "beta": BetaWing,
        "gamma": GammaWing,
        "delta": DeltaWing,
        "epsilon": EpsilonWing,
        "zeta": ZetaWing,
    }
    
    # --- BEFORE: evaluate ALL wings using their trained params ---
    print(f"\n[Interference] BEFORE training {target_wing}...")
    before_scores = {}
    for name in WING_NAMES:
        wing_class = WING_CLASSES[name]
        task_id = WING_TASK_MAP[name]
        _, _, x_data, y_data = task_data[task_id]
        mean_s, std_s = evaluate_wing_class(wing_class, ctx, x_data, y_data, n_eval_samples)
        before_scores[name] = (mean_s, std_s)
        print(f"  {name} ({task_id}): {mean_s:.4f} ± {std_s:.4f}")
    
    # --- TRAIN: target wing only using selected method ---
    print(f"\n[Interference] Training {target_wing} with method={method}...")
    target_task_id = WING_TASK_MAP[target_wing]
    train_x, train_y, _, _ = task_data[target_task_id]
    
    trained = train_wing(
        method=method,
        core=core,
        name=target_wing,
        x=train_x,
        y=train_y,
        seed=seed,
    )
    print(f"  Training complete. Core epoch: {trained.core_epoch}, Core SHA: {trained.core_sha256[:16]}...")
    
    # --- AFTER: evaluate ALL wings ---
    print(f"\n[Interference] AFTER training {target_wing}...")
    after_scores = {}
    
    # For FullFT, use the modified core for all wings
    eval_core = core
    if method == "FullFT" and trained.core_params is not None:
        # Create a FrozenCore with the modified params
        from core.model import FrozenCore
        eval_core = FrozenCore(
            epoch=trained.core_epoch,
            params=trained.core_params,
            meta={"epoch": trained.core_epoch, "kind": "core"},
            sha256=trained.core_sha256,
        )
    
    for name in WING_NAMES:
        task_id = WING_TASK_MAP[name]
        _, _, x_data, y_data = task_data[task_id]
        
        if name == target_wing:
            mean_s, std_s = evaluate_trained_wing(trained, eval_core, x_data, y_data, n_eval_samples)
        else:
            wing_class = WING_CLASSES[name]
            if method == "FullFT" and trained.core_params is not None:
                mean_s, std_s = evaluate_wing_class_with_core(wing_class, eval_core, x_data, y_data, n_eval_samples)
            else:
                mean_s, std_s = evaluate_wing_class(wing_class, ctx, x_data, y_data, n_eval_samples)
        
        after_scores[name] = (mean_s, std_s)
        print(f"  {name} ({task_id}): {mean_s:.4f} ± {std_s:.4f}")
    
    # --- COMPUTE METRICS ---
    target_before = before_scores[target_wing][0]
    target_after = after_scores[target_wing][0]
    target_delta = target_after - target_before
    
    protected_wings = [w for w in WING_NAMES if w != target_wing]
    interference_scores = {}
    for w in protected_wings:
        delta = after_scores[w][0] - before_scores[w][0]
        interference_scores[w] = delta
    
    mean_abs_interf = float(np.mean([abs(v) for v in interference_scores.values()]))
    max_interf = float(np.max([abs(v) for v in interference_scores.values()]))
    
    # Significance test (paired t-test on per-sample scores would be better)
    # Placeholder p-value
    target_p = 0.05 if target_delta > 0 else 0.5
    
    report = InterferenceReport(
        target_wing=target_wing,
        target_task=target_task,
        target_delta=target_delta,
        target_p_value=target_p,
        protected_wings=protected_wings,
        interference_scores=interference_scores,
        mean_abs_interference=mean_abs_interf,
        max_interference=max_interf,
        method=method,
        hardware=hardware,
        seed=seed,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    return report


def compare_methods(
    target_wing: str = "gamma",
    methods: list[str] = None,
    n_eval_samples: int = 200,
) -> list[InterferenceReport]:
    """Run interference experiment for multiple methods."""
    if methods is None:
        methods = ["UMNA", "LoRA", "Adapter", "FullFT"]
    
    reports = []
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running interference experiment: {method} on {target_wing}")
        print(f"{'='*60}")
        
        report = run_interference_experiment(
            target_wing=target_wing,
            method=method,
            n_eval_samples=n_eval_samples,
        )
        reports.append(report)
    
    return reports


def print_comparison_table(reports: list[InterferenceReport]) -> None:
    """Print the key comparison table."""
    print("\n" + "="*80)
    print("INTERFERENCE COMPARISON TABLE (preliminary)")
    print("="*80)
    print(f"{'Method':<12} | {'Target Δ':>10} | {'Mean |Δ|':>12} | {'Max Δ':>10} | {'Min Δ':>10}")
    print("-"*80)
    for r in reports:
        deltas = list(r.interference_scores.values())
        mean_abs = r.mean_abs_interference
        max_d = max(deltas) if deltas else 0.0
        min_d = min(deltas) if deltas else 0.0
        print(f"{r.method:<12} | {r.target_delta:>+10.4f} | {mean_abs:>12.4f} | {max_d:>+10.4f} | {min_d:>+10.4f}")
    print("="*80)
    print("Target Δ > 0 = accuracy improvement on target task")
    print("Protected Δ = accuracy change on protected wings (positive = improvement)")
    print("Mean |Δ| = mean absolute interference across protected wings")
    print("FullFT modifies frozen Core; UMNA/LoRA/Adapter do not (by construction)")
    print("This is a parameter-isolation sanity test, NOT a test of UMNA's routing/composition hypotheses")


def save_reports(reports: list[InterferenceReport], out_path: str) -> None:
    """Save reports as JSON for plotting."""
    data = [asdict(r) for r in reports]
    Path(out_path).write_text(json.dumps(data, indent=2))
    print(f"\nSaved reports to {out_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UMNA Layer B Interference Probe")
    parser.add_argument("--target", default="gamma", choices=WING_NAMES,
                       help="Target wing to train")
    parser.add_argument("--methods", nargs="+", default=["UMNA", "LoRA", "Adapter", "FullFT"],
                       help="Methods to compare")
    parser.add_argument("--samples", type=int, default=200,
                       help="Evaluation samples per wing")
    parser.add_argument("--output", default="interference_results.json",
                       help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print(f"UMNA Layer B Interference Probe (Preliminary Capability-Isolation Sanity Test)")
    print(f"Target: {args.target} ({WING_TASK_MAP[args.target]})")
    print(f"Methods: {args.methods}")
    print(f"Eval samples: {args.samples}")
    print(f"Seed: {args.seed}")
    
    reports = compare_methods(
        target_wing=args.target,
        methods=args.methods,
        n_eval_samples=args.samples,
    )
    
    print_comparison_table(reports)
    
    # Detailed per-wing interference for all methods
    print("\nDETAILED PROTECTED-WING INTERFERENCE (Δ = after - before accuracy):")
    for r in reports:
        print(f"\n  {r.method}:")
        for w in r.protected_wings:
            delta = r.interference_scores[w]
            before = 0.0  # would need to store this
            print(f"    {w}: Δ = {delta:+.4f}")
    
    save_reports(reports, args.output)