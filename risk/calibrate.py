"""Calibrate and register calibration classes for S4 wings.

Each wing's task head is calibrated on its holdout data using split conformal.
"""

from __future__ import annotations

import numpy as np

from runtime.graph import Binding, Wiring, build
from runtime.abi import LatentSchema
from runtime.registry import InMemoryRegistry
from risk.conformal import fit_split_conformal, predict_set, coverage_holds
from risk.classes import CalibrationClass, register, get_registry
from wings.alpha.model import AlphaWing
from wings.beta.model import BetaWing
from wings.gamma.model import GammaWing
from wings.delta.model import DeltaWing
from wings.epsilon.model import EpsilonWing
from wings.zeta.model import ZetaWing
from core.tasks import load_tasks, TASK_AB, TASK_CD, TASK_M, TASK_Z


def collect_scores(wing, registry, graph, ctx, x_data, n_samples=500):
    """Run wing on inputs and collect probabilities and true labels."""
    # For now, just return the probabilities from the wing's execute
    # We'll use the holdout data from tasks
    pass


def calibrate_wing_task(wing_class, task_id, alpha=0.1, seed=42):
    """Calibrate a wing's task head using split conformal on holdout data."""
    # Load holdout data for this task
    tasks = load_tasks(2718)  # DATA_SEED
    holdout_key = f"{task_id}:holdout"
    if holdout_key not in tasks:
        raise ValueError(f"No holdout data for {task_id}")

    holdout = tasks[holdout_key]
    x_holdout = holdout.x
    y_holdout = holdout.y

    # Create wing instance
    wing = wing_class()

    # Get predictions on holdout data
    from runtime.abi import CoreContext, LatentPacket, LatentSchema
    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    probs = []
    for i in range(len(x_holdout)):
        packet = LatentPacket(schema=ctx.latent_schema, data=x_holdout[i].astype(np.float32))
        output = wing.execute(ctx, packet)
        prob = float(output.local_score)
        probs.append(prob)

    probs = np.array(probs)

    # Fit split conformal
    rng = np.random.default_rng(seed)
    calibrator = fit_split_conformal(probs, y_holdout, alpha, rng)

    # Check coverage
    covered = coverage_holds(calib, probs, y_holdout)
    print(f"  {wing_class.__name__} ({task_id}): threshold={calib.threshold:.3f}, coverage={covered}")

    return calibrator


def main():
    print("Calibrating wings for S4...")
    alpha = 0.1  # 90% coverage target

    # Task mapping
    wing_tasks = [
        (AlphaWing, TASK_AB, "t_ab"),
        (BetaWing, TASK_AB, "t_ab"),
        (GammaWing, TASK_CD, "t_cd"),
        (DeltaWing, TASK_CD, "t_cd"),
        (EpsilonWing, TASK_M, "t_m"),
        (ZetaWing, TASK_Z, "t_z"),
    ]

    # Create registry and graph for each wing
    registry = InMemoryRegistry(core_epoch="1.x", seed_dir="wings")

    for wing_class, task_id, class_suffix in wing_tasks:
        wing = wing_class()
        wing_name = wing.manifest().name

        # Build single-wing graph
        graph = build([wing], Wiring())
        binding = Binding("1.x", graph, "head", "test", "python-cpu")

        # Get holdout data
        tasks = load_tasks(2718)
        holdout = tasks[f"{task_id}:holdout"]
        x_holdout = holdout.x
        y_holdout = holdout.y

        # Get predictions
        from runtime.abi import CoreContext, LatentPacket, LatentSchema
        ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

        probs = []
        for i in range(len(x_holdout)):
            packet = LatentPacket(schema=ctx.latent_schema, data=x_holdout[i].astype(np.float32))
            output = wing.execute(ctx, packet)
            probs.append(float(output.local_score))

        probs = np.array(probs)

        # Fit calibrator
        rng = np.random.default_rng(42)
        calibrator = fit_split_conformal(probs, y_holdout, alpha, rng)

        # Check coverage
        cov = coverage_holds(calibrator, probs, y_holdout)
        print(f"{wing_name} ({task_id}): threshold={calibrator.threshold:.3f}, holdout coverage={cov}")

        # Register calibration class
        cls = CalibrationClass(
            class_id=f"class_{wing_name}",
            binding=binding,
            calibrator=calibrator,
            target_level=alpha,
            valid=True,
        )
        register(cls)
        print(f"  Registered: {cls.class_id}")

    print(f"\nTotal registered classes: {len(get_registry().classes)}")


if __name__ == "__main__":
    main()
