# Memory physics

Developer: Koushik Das - Version 0.1.0

## The quantity

M(S): peak memory of the graph, in MB. Peak, not sum: parallel branches
share the accelerator, and the peak over the execution window is what
must fit.

## Where memory is born

| Source | Term | Notes |
|---|---|---|
| Core resident | always resident | one Core epoch per deployment; a Core cannot be vendored twice |
| Wing parameters | cold-load cost | load per graph; warm wings cost near zero |
| Latent packets in flight | transfer term | bounded by graph size and parallelism |
| Trace buffers | measurement cost | recorded, counted |
| Shadow run | s * M_shadow | amortised, same rule as compute |

## The ceiling and the promise

A manifest declares max_memory_mb as a CEILING: a promise the host can
enforce and falsify cheaply. The host measures real peak memory and keeps
it out of the manifest. A ceiling is a promise about the vendor's artifact;
a measurement is a claim about runtime behavior - the host must not take
the second from a party competing for selection.

## The budget

M(S) <= B_M is a constraint, not a penalty. The planner may reject a
graph whose wings would fit individually but not together: peak over the
execution window decides.

## The fault-cost trade

Strong fault isolation duplicates memory (process isolation, host/device
transfer, lost fusion opportunities). Containment tiers F0/F1/F2 publish
(isolation probability, latency cost, memory cost) per fault class rather
than claiming containment as a binary property.