# Calibration-class admission

Developer: Koushik Das - Version 0.1.0

## Why it exists

Calibration attaches to a binding B = (Core epoch, graph, head,
distribution, hardware profile). If the planner builds graphs per input,
every planned graph is an uncalibrated binding and no risk guarantee
survives to production. Admission is the mechanism that resolves this
collision.

## The algorithm

1. The planner returns a graph S*.
2. Look up every registered calibration class.
3. If S* belongs to a class with a currently valid calibrator:
   PRODUCTION. Execution may serve the result.
4. Else: SHADOW_ONLY. The graph may run for data and logging; its output
   is never served. Fall back to the nearest admissible graph, or abstain
   if none exists.
5. The inadmissible proposal is logged. Accumulated shadow evidence is what
   qualifies a new class for registration.

## Class membership

A class is a family of graphs sharing one validated calibrator, for
example: a fixed topology with a bounded substitution set at one node, or a
fixed wing set with permitted orderings provably order-invariant, or a
parameterised family using graph identity as a covariate.

Membership is established empirically at admission time, never asserted.

## Metrics this produces

- Class-admission rate: fraction of planned graphs admissible to
  production. First-class metric. A system that constantly proposes
  inadmissible graphs is under-registered or over-free.

## The invariant

Plan -> admit -> execute. Never plan -> execute.

Expansion returns to admission, not execution: a grown graph is a
different binding.

See flowchart `flowcharts/04_admit.txt`.