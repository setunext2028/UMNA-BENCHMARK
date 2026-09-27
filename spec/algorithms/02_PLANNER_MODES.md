# Planner modes

Developer: Koushik Das - Version 0.1.0

The planner solves a constrained graph-selection problem. Search happens
over the graph space of type-conforming, dependency-satisfied subgraphs
over the candidate set V_x.

## The objective (all modes)

S* = arg max over feasible S of [ U_hat(S, x) - lambda * Gamma(S) ]

subject to C(S) <= B_C, L(S) <= B_L, M(S) <= B_M, S in G(V_x).

Default: lambda = 0, all costs expressed as hard budgets. When a cost is
converted, every conversion constant is measured, never tuned. See
`mathematics/01_cost_identification.md`.

## Mode A - Greedy

- Maintain a partial graph, initially empty.
- Repeatedly add the candidate with the highest host-predicted marginal
  utility Delta U_i(S, x) = U_hat(S union {i}, x) - U_hat(S, x).
- Stop when a budget binds or no addition improves utility.

Used only when task-family validation shows approximate diminishing
returns (submodularity diagnostics), or when compute constraints demand the
cheapest policy. Greedy has the classic (1 - 1/e) bound only under
monotone submodularity, which UMNA does not assume.

## Mode B - Bounded complementary lookahead

- Score all single-candidate graphs.
- Ask the interaction model for at most K_pair pair proposals.
- Include an exploration fraction that tests pairs the model rates
  uncertain, not promising - prevents the model confirming itself.
- Evaluate singles and proposed pairs; optionally keep a small beam of
  partial graphs.

Exists because the synergy structure of capability composition may be
supermodular: two wings worthless alone, strong together. A pure greedy
router never discovers such a pair. The interaction model itself is a
research component with a cold-start problem; its bootstrap protocol is in
`06_CCE_DEPLOYMENT.md`, and it must beat random pairing at equal budget
or Mode B is rejected for that task family (F3).

## Mode C - Beam search

- Maintain a beam of b valid graph states.
- At each step, expand every state by every legal continuation, score,
  keep the b highest-value valid states.
- Budgets prune: an expansion that crosses a hard budget is illegal.

## Restricted oracle (the S5 measuring arm, not a deployable mode)

- All subsets of size <= 3 under a fixed fusion topology, exhaustively
  evaluated.
- Oracle regret is reported against THIS declared restriction, never
  against an unreachable global optimum.

## Planner cost accounting

Planner time is a latency term of its own (T_plan) and is included in every
efficiency comparison. The desired operating regime: adaptive rounds much
smaller than the number of executed wings.