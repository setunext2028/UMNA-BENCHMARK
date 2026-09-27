"""Risk package: post-execution risk test and the calibration-class
registry. Risk imports runtime; runtime never imports risk."""

from runtime.abi import Decision

from .classes import CalibrationClass, ClassRegistry, register
from .conformal import assess

VERSION = "0.1.0"

__all__ = [
    "Decision",
    "CalibrationClass",
    "ClassRegistry",
    "register",
    "assess",
    "VERSION",
]