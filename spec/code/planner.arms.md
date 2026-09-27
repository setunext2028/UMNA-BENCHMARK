# planner/arms/*.py

Developer: Koushik Das - Version 0.1.0

One planner policy per file, all exposing run(x, candidates, budgets).
These are the S5 experiment arms; each is tuned under the same budget as
every other, and the tuning budget is printed in the output (Protocol
9.2).

## topk.py — top-k retrieval-then-execute
- **Contract:** take the k highest-utility wings from retrieval order and
  execute them; budgets cap k.
- **How it works (S5):** utility from host retrieval score; k set by the
  budget envelope. The cheapest arm. If top-k beats NICE-R, the project's
  central hypothesis fails (F1).

## gates.py — independent sigmoid gating
- **Contract:** each wing passes its own gate; the allocator resolves
  budget ties.
- **How it works (S5):** per-wing scores mapped through independent
  gates; wings above threshold compete under the budgets. No global
  softmax denominator, but finite budgets still reintroduce competition
  (paper 21). This arm measures whether the simplest selection mechanism
  suffices.

## greedy.py — greedy marginal utility (Mode A)
- **Contract:** add the wing with the highest marginal utility until
  budgets bind.
- **How it works (S5):** standard marginal-gain loop over the candidates.
  The submodularity question is measured per task family (Experiment B),
  not assumed; greedy is the baseline that synergy tasks must beat.

## pairs.py — bounded complementary lookahead (Mode B)
- **Contract:** evaluate singles plus a bounded set of proposed pairs.
- **How it works (S5):** singles scored; up to K_pair pairs proposed by
  planner/interaction.py, including an exploration fraction of pairs the
  model rates uncertain; the best graph wins. The pair proposer must beat
  random pairing at equal budget or this arm is rejected for the family
  (F3).

## beam.py — beam search (Mode C)
- **Contract:** maintain a beam of width b over legal graph continuations.
- **How it works (S5):** expand every beam state by every legal
  continuation, keep the b best valid graphs; budgets prune. Beam width
  b = 4 in MVP; beam-width sensitivity is reported.

## oracle.py — restricted oracle
- **Contract:** exhaustive evaluation over all subsets of size <= 3 under
  a fixed fusion topology.
- **How it works (S5):** the measurement instrument every other arm is
  compared against. Oracle regret is reported against THIS declared
  restriction — never against an unreachable global optimum — and the
  restriction is printed with the number.