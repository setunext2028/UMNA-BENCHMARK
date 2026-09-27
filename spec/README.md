# UMNA specification

Developer: Koushik Das - Version 0.1.0

This directory documents how UMNA works, separate from the architectural
docs in `docs/`. The architecture docs say WHAT and WHY. This directory
says HOW: every algorithm, the mathematics it rests on, the resource
physics it obeys, and - file by file, function by function - what the code
does.

Read in this order:

| Folder | Answers |
|---|---|
| `algorithms/` | The full algorithms: retrieval, planning modes, admission, execution, risk, deployment |
| `mathematics/` | The equations: objective, cost identification, utility, conformal risk, drift, stability |
| `physics/` | The resource physics: how compute, latency and memory flow and where they are consumed |
| `code/` | One document per code file, with every function explained |
| `flowcharts/` | Small ASCII flowcharts, one per portion of the system |
| `lifecycle/` | Upgrade and downgrade operations, end to end |

The code has passed S1: contracts are typed; strict manifest validation,
registry loading/resolution/pinning, ceiling checks, typed graph building,
and constant stub Wings are implemented. Execution, tracing, calibration,
and planning remain staged work. Each `code/` document distinguishes what
is live from what is deferred.

The binding contract for this specification is `docs/05_SEAMS_AND_API.md`.
Where this directory and the seams disagree, the seams win and this
directory gets fixed.
