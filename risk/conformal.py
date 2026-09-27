"""Split conformal prediction for binary classification (S4).

Given a trained head that outputs probabilities p(y=1|x), split conformal
constructs prediction sets C(x) with guaranteed marginal coverage:
P(y in C(x)) >= 1 - alpha.

For binary classification, we use the standard approach:
- Conformity score: 1 - p(y | x) where y is the true label
- Calibration: compute (1-alpha) quantile of scores on calibration set
- Prediction set: {y : 1 - p(y|x) <= threshold}
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from runtime.abi import Output
from runtime.graph import Binding
from runtime.abi import Decision


@dataclass(frozen=True)
class SplitConformalCalibrator:
    """Fitted split-conformal calibrator for a binary head.

    Holds the threshold quantile computed on calibration data.
    """

    alpha: float
    threshold: float
    scores: tuple[float, ...]  # calibration conformity scores


def fit_split_conformal(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> SplitConformalCalibrator:
    """Fit split-conformal threshold on calibration data.

    Args:
        probs: (n, 1) or (n,) array of p(y=1|x) probabilities
        labels: (n,) array of binary labels {0, 1}
        alpha: miscoverage level (e.g., 0.1 for 90% coverage)
        rng: optional RNG for tie-breaking

    Returns:
        SplitConformalCalibrator with threshold quantile
    """
    if probs.ndim == 2:
        probs = probs.ravel()
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if len(probs) != len(labels):
        raise ValueError("probs and labels length mismatch")

    # Conformity score: 1 - p(y|x)
    scores = np.where(labels == 1, 1.0 - probs, probs)
    if rng is None:
        rng = np.random.default_rng(0)

    # Split conformal threshold: (1-alpha) quantile with tie-breaking
    n = len(scores)
    # Standard split conformal: ceil((n+1)*(1-alpha)) / n quantile
    q = math.ceil((n + 1) * (1.0 - alpha)) / n
    threshold = float(np.quantile(scores, q, method="higher"))

    return SplitConformalCalibrator(
        alpha=alpha,
        threshold=threshold,
        scores=tuple(float(s) for s in scores),
    )


def predict_set(
    calibrator: SplitConformalCalibrator,
    prob: float,
) -> tuple[int, ...]:
    """Construct prediction set for a single probability.

    Returns:
        tuple of labels in {0, 1} with conformity score <= threshold
    """
    s0 = prob      # 1 - p(0|x) = p(1|x) = prob
    s1 = 1.0 - prob  # 1 - p(1|x) = p(0|x) = 1 - prob
    result = []
    if s0 <= calibrator.threshold:
        result.append(0)
    if s1 <= calibrator.threshold:
        result.append(1)
    return tuple(result)


def coverage_holds(
    calibrator: SplitConformalCalibrator,
    probs: np.ndarray,
    labels: np.ndarray,
) -> bool:
    """Check empirical coverage on test data."""
    covered = 0
    for p, y in zip(probs.ravel(), labels):
        pred_set = predict_set(calibrator, float(p))
        if y in pred_set:
            covered += 1
    return covered / len(labels) >= 1.0 - calibrator.alpha


def assess(output: Output, binding: Binding) -> Decision:
    """Post-execution risk test for a binding.

    Uses the binding's calibrator (if any) to produce a prediction set
    from the output's score. Returns:
    - ACCEPT: prediction set is a singleton {y} (high confidence)
    - ABSTAIN: prediction set is {0, 1} (uninformative)
    - EXPAND: empty set (model failure, need different graph)
    """
    # Extract the calibrator from the binding's metadata
    calibrator = getattr(binding, "_calibrator", None)
    if calibrator is None:
        # No calibrator registered: safe default is abstain
        return Decision.ABSTAIN

    # Get the probability from output (latent.data is the score)
    score = output.value
    if isinstance(score, np.ndarray):
        if score.size == 1:
            prob = float(score.ravel()[0])
        else:
            prob = float(score[0])
    else:
        prob = float(score)

    pred_set = predict_set(calibrator, prob)

    if len(pred_set) == 1:
        return Decision.ACCEPT
    elif len(pred_set) == 0:
        return Decision.EXPAND
    else:
        return Decision.ABSTAIN


if __name__ == "__main__":
    # Quick sanity check
    rng = np.random.default_rng(42)
    n_cal = 500
    n_test = 1000

    # Simulate a well-calibrated head
    cal_probs = rng.uniform(0.1, 0.9, n_cal)
    cal_labels = (rng.uniform(0, 1, n_cal) < cal_probs).astype(int)

    test_probs = rng.uniform(0.1, 0.9, n_test)
    test_labels = (rng.uniform(0, 1, n_test) < test_probs).astype(int)

    calib = fit_split_conformal(cal_probs, cal_labels, alpha=0.1)
    cov = coverage_holds(calib, test_probs, test_labels)
    print(f"Calibrated threshold: {calib.threshold:.3f}")
    print(f"Empirical coverage: {cov} (target >= 0.9)")

    # Test prediction sets
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"p={p:.1f} -> {predict_set(calib, p)}")
