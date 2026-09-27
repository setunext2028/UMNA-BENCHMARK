# Traps

Developer: Koushik Das · Version 0.1.0

Each of these cost something. Read before debugging; add to it when you lose
an afternoon.

---

**Gradients do flow through the frozen Core.** The Wing sits upstream, so
backprop must traverse the Core to reach it. `dL/d(core)` is genuinely
non-zero. Only the *write* is blocked. Any comment or diagram claiming
gradient flow is blocked is wrong and will be flagged in review.

**A hash check on parameters the optimiser never received cannot fail.**
It belongs in CI. It is not evidence of anything, and reporting it as a
finding invites the obvious objection.

**Pinned routes make the interference matrix all zeros.** Measuring
interference on frozen parameters and cache-pinned routes measures nothing.
Measure at promotion events, where behaviour is allowed to change.

**Six trade-off coefficients beat a one-threshold baseline on tuning alone.**
This nearly invalidated the whole benchmark. Default to hard budgets with
`lambda = 0`, and print every arm's tuning budget in the output.

**Binding-level calibration collides with dynamic graphs.** If calibration is
indexed by graph and the planner builds graphs per input, every graph is
uncalibrated. Hence admission, and hence calibration classes. Anyone tidying
the call path will be tempted to inline `admit` into `plan`. Do not let them.

**Expansion returns to admission, not to execution.** A graph that grew is a
different binding. Re-admit it.

**A deny-list of forbidden manifest fields loses.** The vendor renames the
field. Validate against an allow-list.

**Host measurement fixes lying metadata, not lying behaviour.** A Wing can
still detect validation probes and behave well only on them. Probes must be
randomised and held out, and production behaviour compared against the
admission profile.

**Near-threshold float noise changes which Wing runs.** L2 replay does not
imply L3. Use hysteresis, decision margins, or quantised bins on every
control-path comparison.

**Shadow execution is not free.** It doubles compute on sampled traffic. If
efficiency numbers exclude it, they are excluding the cost of the mechanism
that keeps the router trained.

**"Planner cost depends on m, not N" is true by construction.** Two-stage
retrieval guarantees it. The falsifiable question is whether retrieval
*recall* survives growth in N. Watch for this pattern generally: a hypothesis
that cannot fail has usually been restated from an implementation detail.

**Oracle regret needs a declared restriction.** The graph space includes
topologies and orderings, not only subsets. Exhaustive search over subsets of
size <= 3 under a fixed topology — say so explicitly when reporting.

**The interaction model must beat random pairing at equal budget.** Non-zero
pair-discovery recall is not a result; beating the cheapest alternative is.
It also has a cold start: it needs pair observations and exists to avoid
executing pairs. Seed it offline.

**Conflict resolved by taking the higher score is a silent failure.** Default
is abstain and escalate. Overriding that needs an explicit, versioned,
calibrated policy per task family.
