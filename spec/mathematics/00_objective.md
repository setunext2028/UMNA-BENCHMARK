# The NICE-R objective

Developer: Koushik Das - Version 0.1.0

## Setup

- x in D_e: the input, representable through the validated latent/type
  contract of Core epoch e.
- V_x subset of W_e: the host-retrieved candidate set (stage 1, size m).
- G(V_x): the space of type-conforming, dependency-satisfied subgraphs
  over V_x.
- S = (V_S, E_S): one valid typed execution subgraph.

## Objective

S* = arg max over S in G(V_x) of [ U_hat(S, x) - lambda * Gamma(S) ]

subject to the hard constraints:

C(S) <= B_C        (compute, FLOPs)
L(S) <= B_L        (latency, ms)
M(S) <= B_M        (memory, MB)
S in G(V_x)        (type and dependency validity)

where U_hat(S, x) is the host-predicted task utility and Gamma(S) is a
single scalar cost in a common currency.

## The default: lambda = 0

The default planner has zero free coefficients. All cost terms are
constraints. Nothing is traded off against quality by a tuned coefficient.
A scalarised variant (lambda > 0) is allowed only as a separately reported
arm whose conversion constants are MEASURED, and that arm is never the
primary result (Protocol 9.2).

## Marginal utility

The value of adding capability i to graph S:

Delta U_i(S, x) = U_hat(S union {i}, x) - U_hat(S, x)

estimated by the host from controlled data, historical routes, shadow
trials, or a learned host-side estimator. A vendor-supplied field such as
expected_evidence_gain = 0.25 is not admissible as a routing signal.

## Non-submodularity

U is a set function; UMNA does not assume submodularity. A supermodular
interaction is:

U({A,B}) - U(empty) > [U({A}) - U(empty)] + [U({B}) - U(empty)]

In the extreme: U({A}) ~= U({B}) ~= U(empty), U({A,B}) >> U(empty). A pure
greedy router selects neither. This is why Mode B and Mode C exist, and
why submodularity is measured per task family (Experiment B) rather than
assumed.

## Note on the abbreviations

U_hat is written U with a hat in the paper (estimated utility); D_e is the
compatibility domain of epoch e; G(V_x) is the graph space over the
candidate set. The same symbols appear throughout this directory.