# Package entry points

Developer: Koushik Das - Version 0.1.0

## runtime/__init__.py

Re-exports ABI values, budget types, graph construction, admission types,
and registry APIs. It imports neither planner nor Wing implementations.
`VERSION` is pinned to `0.1.0`; `__all__` is the public runtime surface.

## planner/__init__.py

Re-exports `Mode`, predicted utility, estimator, and plan seam plus the
runtime types needed by callers. It contains no arm logic.

## planner/arms/__init__.py

Marks the S5 arm package. Each arm implementation stays in its own file.

## risk/__init__.py

Re-exports the decision enum, calibration-class types, class registration,
and conformal assessment seam.

## core/__init__.py

Pins project version `0.1.0` and the provisional MVP Core epoch `1.x`.
Core model behavior arrives in S3.

## wings/__init__.py

Declares the package rule: Wings compose through Graph, never by importing
one another.

## bench/__init__.py and bench/tasks/__init__.py

Mark the S5 benchmark and task packages. Nothing imports bench.
