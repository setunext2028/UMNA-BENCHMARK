# Build order

Developer: Koushik Das · Version 0.1.0

Stages are sequential. Each has an exit gate. Do not start a stage until the
previous gate is green — the ordering exists because each stage is the
measuring instrument for the next.

---

## S0 · Contracts on paper
Write the manifest schema, the Wing trait, the LatentPacket schema, the trace
schema, and the budget types. No behaviour yet.

**Gate:** every seam in `05_SEAMS_AND_API.md` has a concrete type. A reviewer
can tell, from types alone, where any given piece of logic belongs.

---

## S1 · Registry and ABI, with stub Wings
Load manifests, validate them, reject host-owned fields, resolve versions,
pin the Core epoch, enforce ceilings. Wings return constants.

**Gate:** a manifest declaring `reliability:` is rejected with a clear error.
An incompatible Core epoch is rejected. A type-invalid graph fails to build.

> Stub Wings first is not laziness. It separates control-plane bugs from model
> bugs, and you will never get a cleaner chance to do so.

---

## S2 · Execution and trace
Run a graph. Produce a full trace with decomposed latency, including
`T_plan` and `T_admit`. No planner yet — graphs are hand-specified.

**Gate:** timings sum to wall clock within tolerance. A trace replays to the
same graph. Replay level is *recorded*, not assumed.

---

## S3 · Real Core and real Wings
Train the small Core once, freeze it, hash it. Train 4–6 Wings through it
with the write firewall on.

**Gate:** Core and old-Wing hashes are byte-identical after a new Wing trains.
An attempted unauthorised write raises. (CI assertion — not a result.)

---

## S4 · Risk layer and calibration classes
Split conformal on the task head. Register the hand-built classes. Wire the
admission check between planning and execution.

**Gate:** coverage holds on held-out data at the target level. A graph outside
every class is refused production and routed to shadow.

---

## S5 · NICE-R and the arms — **the flagship**
Utility estimator, then: top-k, independent gate, greedy, bounded-pair, beam,
restricted oracle. Equal tuning budget enforced by the harness.

**Gate:** the full comparison table, reproducible from one command, with
per-arm tuning budget printed in the output.

---

## S6 · Synergy probe
Build tasks where two Wings are worthless alone and strong together. Test
whether greedy misses them and whether the interaction model beats **random
pairing at equal budget**.

**Gate:** a clear verdict. "Mode B rejected for this family" is a valid,
publishable outcome.

---

## S7 · Scaling probe
Grow the registry with proxy and metadata-only entries. Measure retrieval
recall and latency as N grows.

**Gate:** recall@m and retrieval latency plotted against N.

---

## Later, in this order
Shadow routing and router learning → cumulative drift gating → safety
interrupts → Bridges → second modality → Rust control plane → fault
containment tiers → Core migration study.

Each of these is only meaningful once S5 has an answer. Building them early
means maintaining them through every S5 revision.
