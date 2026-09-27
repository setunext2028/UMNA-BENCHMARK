# Utility and abstention prediction

Developer: Koushik Das - Version 0.1.0

## U_hat - the utility estimator

U_hat(S, x): predicted task utility of graph S on input x. Features come
from the host registry only (HostMetrics):

- measured latency profile
- measured memory profile
- reliability / failure history
- utility features from controlled validation and shadow telemetry
- simple observable features of x (type structure, length, modality)

The estimator is trained by the host on controlled data, historical
routes, and shadow trials. It is itself a research component: H1 depends on
it being good enough to guide search, so the estimator is evaluated in its
own right (calibration of the utility predictor, Experiment A), not just
as an input to search.

## A_hat - the abstention predictor

A_hat(S, x): predicted probability that the graph will fail the
post-execution risk test and waste its compute.

This is retained from earlier drafts but flagged honestly: predicting
whether a graph will abstain requires anticipating the conformal outcome
before execution - the same epistemic difficulty class as U_hat. It is a
soft EFFICIENCY term only. It never overrides the authoritative
post-execution test, and it is evaluated as an estimator in its own right.

## What may never enter either function

- any vendor-declared number (reliability, accuracy, expected latency,
  expected evidence gain, routing priority, quality, confidence);
- any number read from a manifest at plan time.

Every value feeding U_hat or A_hat is host-measured or host-derived. The
seam where this is enforced is the registry: HostMetrics is the only type
that carries routing-relevant numbers, and the manifest validator rejects
their declaration at load.