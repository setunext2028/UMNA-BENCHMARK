# Runtime physics

Developer: Koushik Das - Version 0.1.0

The system obeys two conservation laws that every number in the trace must
respect.

## Law 1 - wall clock is conserved

T_total = T_retrieve + T_plan + T_admit + T_load + T_transfer + T_wing +
T_fusion + T_risk

Nothing is created or destroyed by measurement. If the terms do not sum to
wall clock within tolerance, the measurement is broken, not the clock.
This is why a new kind of work may never be absorbed silently into an
existing term: work is mass, and hiding it violates conservation.

Planner time, admission time, cold-load time, transfer time and shadow
traffic are real mass and appear in every efficiency comparison. Reporting
Wing compute alone is dishonest measurement - it reports one term of a sum
that is dominated by other terms.

## Law 2 - routing information flows only one way

Every number that affects routing flows hostward: vendor to registry
(ceiling, types, dependencies) then registry to planner (measured values).
The reverse direction does not exist. A manifest field cannot push a number
into the planner because the manifest schema is an allow-list typed to
carry nothing of the kind. This is enforced at construction, not by trust.

Gradients are the one flow that is NOT blocked: during Wing training,
backpropagation traverses the frozen Core, and dL/d(core-param) is
genuinely non-zero. Only the WRITE is blocked. Claiming otherwise in a
comment or diagram is wrong and will be flagged in review.

## Law 3 - evidence is conserved, conflict is not averaged

Each claim keeps its polarity (support / oppose / unknown), source wing,
version, route and provenance. Contradictory evidence is not collapsed
into an average confidence; it is passed upward as a ConflictSet. A
conflict above the action threshold terminates in abstain-and-escalate
unless the task family has a calibrated, versioned override.