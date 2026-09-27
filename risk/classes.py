"""Calibration classes: the pre-registered graph families allowed to serve.

Membership is established empirically at admission time (paper 8.1). The
planner's production freedom IS the union of registered classes; everything
else is shadow-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.abi import Decision
from runtime.graph import Binding, Graph
from risk.conformal import SplitConformalCalibrator


@dataclass(frozen=True)
class CalibrationClass:
    """One registered family of graphs sharing a valid calibrator.

    A class may be a fixed topology with a bounded substitution set, or a
    parameterised family fitted with graph identity as a covariate.
    """

    class_id: str
    binding: Binding
    calibrator: SplitConformalCalibrator
    target_level: float  # alpha
    valid: bool = True


@dataclass
class ClassRegistry:
    """The registry `admit` consults. One entry per registered class."""

    classes: dict[str, CalibrationClass] = field(default_factory=dict)

    def register(self, cls: CalibrationClass) -> None:
        """Add a class after empirical validation."""
        if cls.class_id in self.classes:
            raise ValueError(f"class {cls.class_id!r} already registered")
        self.classes[cls.class_id] = cls

    def get(self, class_id: str) -> CalibrationClass | None:
        return self.classes.get(class_id)

    def contains(self, graph: Graph) -> bool:
        return any(
            cls.binding.graph.graph_id == graph.graph_id and cls.valid
            for cls in self.classes.values()
        )

    def find_class(self, graph: Graph) -> CalibrationClass | None:
        for cls in self.classes.values():
            if cls.binding.graph.graph_id == graph.graph_id and cls.valid:
                return cls
        return None


# Global registry instance
_registry = ClassRegistry()


def get_registry() -> ClassRegistry:
    return _registry


def register(cls: CalibrationClass) -> None:
    """Add a class after empirical validation."""
    _registry.register(cls)


if __name__ == "__main__":
    from runtime.graph import Binding
    from runtime.abi import LatentSchema

    reg = get_registry()
    print(f"Registry empty: {len(reg.classes) == 0}")
