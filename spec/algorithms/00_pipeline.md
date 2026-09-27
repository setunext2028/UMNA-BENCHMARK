# The end-to-end pipeline algorithm

Developer: Koushik Das - Version 0.1.0

One input, from arrival to serve or abstain. This is Appendix A of the
paper, restated as an algorithm with the invariant points marked.

## Steps

1. **Input analysis.** Classify the input's modalities, semantic roles and
   structural requirements.
2. **Candidate retrieval (stage 1).** Reduce the installed registry
   (N wings) to a bounded candidate set V_x (m wings) using type
   compatibility, Core-epoch constraints, dependency and security filters,
   embedding retrieval over capability tags, and host route statistics.
3. **Candidate filtering (stage 2 intake).** Reject candidates violating
   Core epoch, ABI, dependencies, integrity, security or policy. Apply hard
   resource ceilings. Attach host-measured attributes only.
4. **Protected route check.** If a protected live route exists for this task
   family, execute it; optionally run NICE-R in shadow on a sampled
   fraction s; evaluate safety interrupts; return. Never serve the shadow
   result.
5. **Planning.** Score single-candidate graphs. Mode B adds a bounded set of
   pair proposals (up to K_pair) from the interaction model, including an
   exploration fraction. Mode C keeps a beam of width b over legal partial
   graphs. Pick S* maximising U_hat(S, x) - lambda * Gamma(S) subject to
   C(S) <= B_C, L(S) <= B_L, M(S) <= B_M. Default lambda = 0.
6. **Calibration-class admission.** If S* belongs to a registered class with
   a valid calibrator, proceed to production. Otherwise S* is shadow-only:
   fall back to the nearest admissible graph, or abstain if none exists.
   Log the inadmissible proposal - it is evidence toward a new class.
7. **Compile, schedule, execute.** Independent branches run in parallel.
   Decomposed latency is recorded per term.
8. **Fuse.** Bridges and the task head combine Wing outputs.
9. **Risk control.** Apply the binding-level selective-risk test.
10. **Decide.** Risk criterion met -> output. Adaptive budget remains and
    the test failed -> one bounded expansion round, returning to step 6
    (re-admission). Otherwise -> abstain.
11. **Log.** Exact versions, graph, binding, control values, decomposed
    latency, replay level, outcome, failures.

## Load-bearing invariants

- Step 6 sits between planning and execution. A graph cannot serve without
  admission.
- Step 10 returns to admission, not execution. A grown graph is a different
  binding.
- Step 3 attaches only host-measured numbers. No vendor-declared routing
  metric exists in the system.
- Every failure path ends in abstain, never in a guess.