"""Calibration-class admission. Sits between plan and execute, the load
bearing invariant of the architecture.

No call path reaches execute.run without passing admit. A graph outside
every registered class is ShadowOnly: it may run for data, never serve.

With CCE: additionally checks drift gate eligibility for all wings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .graph import Binding, Graph
from .cce import check_cce_admission
from risk.classes import get_registry


class Admission(Enum):
    PRODUCTION = "production"
    SHADOW_ONLY = "shadow_only"


@dataclass(frozen=True)
class AdmissionRecord:
    """Disposition bound to the exact graph and execution binding."""

    disposition: Admission
    graph_id: str
    binding_id: str
    calibration_class: str | None = None


class AdmissionError(ValueError):
    """The graph and binding passed to admission do not describe one run."""


def admit(graph: Graph, binding: Binding) -> AdmissionRecord:
    """Decide whether a planned graph may serve.

    Looks up the graph in the calibration class registry. If the graph
    belongs to a registered class, returns PRODUCTION with that class ID.
    Otherwise returns SHADOW_ONLY.

    With CCE: also checks drift gate eligibility for all wings in the graph.
    """
    return check_cce_admission(graph, binding, None)


if __name__ == "__main__":
    from runtime.graph import Binding, Wiring, build
    from runtime.abi import Manifest, TypeExpr, TypeSet, TypeSet, type_unifies

    class TypedWing:
        def __init__(self, name, input_arg, output_arg):
            self._manifest = Manifest(
                name=name,
                version="1.0.0",
                core_epoch="1.x",
                author="test",
                input_types=TypeSet((TypeExpr("Latent", (input_arg,)),)),
                output_types=TypeSet((TypeExpr("Latent", (output_arg,)),)),
            )
        def manifest(self): return self._manifest
        def input_types(self): return self._manifest.input_types
        def output_types(self): return self._manifest.output_types

    alpha = TypedWing("alpha", "Text", "Summary")
    gamma = TypedWing("gamma", "Summary", "Claim")
    graph = build([alpha, gamma], Wiring((("alpha", 0, "gamma", 0),)))
    binding = Binding("1.x", graph, "head", "test", "cpu")

    record = admit(graph, binding)
    print(f"Disposition: {record.disposition}")
    print(f"Class: {record.calibration_class}")
