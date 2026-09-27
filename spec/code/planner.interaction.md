# planner/interaction.py

Developer: Koushik Das - Version 0.1.0

The Mode B pair proposer — the research component the paper names
honestly: the mitigation for the combinatorial pair problem is itself an
unsolved learning problem, and it sits on the critical path for H2.

Imports abi. Imported by planner/arms/pairs.py. S6.

## Functions

### propose_pairs(x, candidates, k_pair)
- **Contract:** propose at most k_pair promising complementary pairs.
- **How it works (S6):** a host-learned model over pair embeddings, seeded
  by the bootstrap protocol:
  1. offline random pair sample over a held-out probe set at admission;
  2. type-prior initialisation — wings producing complementary evidence
     classes over the same input start with a higher synergy prior;
  3. shadow harvesting of pair observations from live traffic;
  4. an exploration fraction that must execute pairs the model rates
     UNCERTAIN, so the model cannot confirm itself;
  5. falsification: if after the seeding budget pair-discovery recall does
     not beat random pair sampling at equal cost, Mode B is rejected for
     the task family and that finding is reported (F3).
- **Failure mode:** overconfident proposals that only repeat what shadow
  traffic already showed; mitigated by the exploration fraction.
- **Boundary:** proposes pairs, never executes them. Execution cost stays
  with the executor and the trace.