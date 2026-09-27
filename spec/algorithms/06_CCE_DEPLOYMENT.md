# CCE: Conservative Capability Extension

Developer: Koushik Das - Version 0.1.0

CCE is a deployment protocol, not a learning theorem.

## Live route and shadow route

A protected live route stays pinned. NICE-R plans in shadow on a sampled
fraction s of traffic:

- the router keeps receiving data from common high-traffic cases instead
  of only the unprotected tail;
- proposed route changes are evaluated without altering live output;
- shadow cost is real: amortised overhead = T_live + s * T_shadow, and
  efficiency comparisons must report it.

## Promotion

A candidate route is promoted only if BOTH:

- step drift: P(t) - P(t-1) >= -epsilon (per-change non-inferiority);
- origin drift: P(t) - P(0) >= -E (cumulative budget against the
  permanently pinned origin).

Without the second constraint, T sequential promotions each passing at
-epsilon can degrade a capability by up to T*epsilon with every individual
gate green.

## Protection tiers and the power budget

Tier A (critical): strict non-inferiority, family-wise error control,
permanently pinned origin. Tier B (important): non-inferiority under a less
conservative policy, scheduled full regression. Tier C (exploratory):
false-discovery-rate control; promotion only after sufficient evidence.

Tier A membership is bounded by statistics: with family-wise control,
per-test alpha shrinks roughly as 1/|A|, so |A|max is the largest k such
that power(delta, n, alpha/k) >= 1 - beta. Computed and published BEFORE the
first promotion. Designating more Tier A capabilities than |A|max means
running underpowered tests that cannot catch the regressions the tier
exists to detect.

## The interaction model bootstrap

The Mode B pair proposer has a cold-start circularity: synergy observations
come only from executing pairs, and the model exists to avoid executing
pairs. Protocol:

1. Offline seeding: a bounded random pair sample over a held-out probe set,
   once per admitted wing, at admission time.
2. Type-prior initialisation: capabilities producing complementary evidence
   classes over the same input get a higher synergy prior than
   capabilities producing the same evidence class.
3. Shadow harvesting: pair observations from live traffic, no production
   effect.
4. Exploration budget: a fixed fraction of shadow traffic executes pairs
   the model rates UNCERTAIN rather than PROMISING.
5. Falsification: after the seeding budget is exhausted, pair-discovery
   recall must exceed random pair sampling at equal cost, or Mode B is
   rejected for that task family and the finding is reported (F3).

## Safety interrupts

Absolute route freezing creates omission risk. Pre-authorized wings may
emit INTERRUPT.EMERGENCY, INTERRUPT.CONTRAINDICATION,
INTERRUPT.HARD_CONFLICT, INTERRUPT.SECURITY or INTERRUPT.LEGAL_POLICY. An
interrupt never silently replaces the protected route: it appends a guard
result, forces escalation, forces abstention, requests a secondary route,
or blocks an unsafe action. Interrupt paths are bindings like any other:
their false-interrupt rate and recall are separately calibrated and gated.

See flowcharts `flowcharts/07_upgrade_promotion.txt` and
`flowcharts/08_downgrade_rollback.txt`.