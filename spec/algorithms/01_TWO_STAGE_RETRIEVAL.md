# Two-stage candidate retrieval

Developer: Koushik Das - Version 0.1.0

Installed set W_e of size N. Stage 1 brings N down to m candidates; stage 2
plans over m.

## Stage 1 - typed and indexed retrieval (N -> m)

Filters applied in order:

1. **Type compatibility.** The input TypeExpr of the request must unify with
   a Wing's declared input types. No unify, no candidate.
2. **Core-epoch constraint.** The Wing's manifest must match the active
   epoch's latent contract. One epoch per deployment.
3. **Dependency and security filters.** Required dependencies must be
   present in the registry; integrity hashes and signatures must verify.
4. **Semantic retrieval.** Embedding retrieval over capability tags, mixed
   with host historical route statistics.

Cost of stage 1 grows with N (indexing and scanning). Cost of stage 2
depends on m, not N - this is true by construction and is not a research
claim. The falsifiable question (H5) is whether retrieval recall survives
growth in N.

## Stage 2 - planning intake (m -> S*)

- Apply hard resource ceilings.
- Attach host-measured attributes only: measured latency profile, measured
  memory profile, reliability / failure history, utility features.
- Score and plan as described in `02_PLANNER_MODES.md`.

## Scale targets and MVP numbers

| Quantity | MVP | Experiment I |
|---|---|---|
| Registry N | 20 | up to 1000 |
| Candidates m | ~8 | ~12 |
| Planned graph size | 1-3 | 2-4 |

## Recall is the measured quantity

- recall@m: fraction of oracle-containing candidate sets whose oracle wing
  made the cutoff.
- The oracle wing is defined by the restricted oracle of
  `02_PLANNER_MODES.md`.
- Plotted against N in S7. Degradation is F8.