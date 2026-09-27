"""Calibrate and register multi-wing calibration classes for S4/S5.

Each multi-wing graph's task head is calibrated on its holdout data using split conformal.
"""

from __future__ import annotations

import numpy as np

from runtime.graph import Binding, Wiring, build
from runtime.abi import LatentSchema, CoreContext, LatentPacket
from runtime.registry import InMemoryRegistry
from risk.conformal import fit_split_conformal, coverage_holds
from risk.classes import CalibrationClass, register, get_registry
from wings.alpha.model import AlphaWing
from wings.beta.model import BetaWing
from wings.gamma.model import GammaWing
from wings.delta.model import DeltaWing
from wings.epsilon.model import EpsilonWing
from wings.zeta.model import ZetaWing
from core.tasks import load_tasks, TASK_AB, TASK_CD, TASK_M, TASK_Z


WING_CLASSES = {
    "alpha": AlphaWing,
    "beta": BetaWing,
    "gamma": GammaWing,
    "delta": DeltaWing,
    "epsilon": EpsilonWing,
    "zeta": ZetaWing,
}

# Multi-wing pairs per task
MULTI_WING_TASKS = [
    # (class_id_suffix, task_id, [wing_name1, wing_name2], combine_fn)
    ("alpha_beta", TASK_AB, ["alpha", "beta"], lambda s1, s2: (s1 + s2) / 2),
    ("gamma_delta", TASK_CD, ["gamma", "delta"], lambda s1, s2: (s1 + s2) / 2),
    ("epsilon_zeta", TASK_M, ["epsilon", "zeta"], lambda s1, s2: (s1 + s2) / 2),
    ("zeta_gamma", TASK_Z, ["zeta", "gamma"], lambda s1, s2: (s1 + s2) / 2),
]


def collect_joint_scores(wing1, wing2, ctx, x_data, combine_fn):
    """Run both wings on inputs and combine their scores."""
    probs = []
    for i in range(len(x_data)):
        packet = LatentPacket(schema=ctx.latent_schema, data=x_data[i].astype(np.float32))
        out1 = wing1.execute(ctx, packet)
        out2 = wing2.execute(ctx, packet)
        combined = combine_fn(float(out1.local_score), float(out2.local_score))
        probs.append(combined)
    return np.array(probs)


def main():
    print("Calibrating multi-wing classes for S4/S5...")
    alpha = 0.1  # 90% coverage target

    # Create registry and load all wings
    registry = InMemoryRegistry(core_epoch="1.x", seed_dir="wings")

    # Load holdout data once
    tasks = load_tasks(2718)

    ctx = CoreContext(core_epoch="1.x", latent_schema=LatentSchema("1.x", 8, "float32"))

    for class_suffix, task_id, wing_names, combine_fn in MULTI_WING_TASKS:
        wing1 = WING_CLASSES[wing_names[0]]()
        wing2 = WING_CLASSES[wing_names[1]]()

        # Build multi-wing graph
        graph = build([wing1, wing2], Wiring())
        binding = Binding("1.x", graph, "head", "test", "python-cpu")

        # Get holdout data
        holdout = tasks[f"{task_id}:holdout"]
        x_holdout = holdout.x
        y_holdout = holdout.y

        # Get joint predictions
        probs = collect_joint_scores(wing1, wing2, ctx, x_holdout, combine_fn)

        # Fit calibrator
        rng = np.random.default_rng(42)
        calibrator = fit_split_conformal(probs, y_holdout, alpha, rng)

        # Check coverage
        cov = coverage_holds(calibrator, probs, y_holdout)
        wing1_name = wing1.manifest().name
        wing2_name = wing2.manifest().name
        print(f"{wing1_name}+{wing2_name} ({task_id}): threshold={calibrator.threshold:.3f}, holdout coverage={cov}")

        # Register calibration class
        cls = CalibrationClass(
            class_id=f"class_{class_suffix}",
            binding=binding,
            calibrator=calibrator,
            target_level=alpha,
            valid=True,
        )
        register(cls)
        print(f"  Registered: {cls.class_id}")

    print(f"\nTotal registered classes: {len(get_registry().classes)}")
    for cid in sorted(get_registry().classes.keys()):
        print(f"  - {cid}")


if __name__ == "__main__":
    main()