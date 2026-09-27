# MVP scope

Developer: Koushik Das · Version 0.1.0

The MVP exists to answer **one question**:

> Given a fixed set of independently trained capability modules, can a
> host-controlled planner find small typed graphs that beat top-k,
> independent-gate, and greedy routing on quality / latency / coverage —
> under an equal tuning budget?

Everything below is chosen to answer that and nothing else.

## In scope

| Area | MVP content |
|---|---|
| Modality | **One.** Text or tabular — whichever is cheaper to source. Not both. |
| Core | One small frozen Core. It does not need to be good; it needs to be *fixed*. |
| Wings | **4 to 6.** Deliberately built with known overlap: two redundant, two complementary, one misleading, one cheap-but-weak. |
| Planner | top-k · independent gate · greedy · bounded-pair lookahead · beam · restricted oracle |
| Registry | In-process, file-backed. Typed retrieval, version pinning, ceiling enforcement. |
| ABI | Typed Wing trait, manifest schema, manifest validator that rejects host-owned fields. |
| Admission | Calibration-class registry with a handful of hand-registered classes. |
| Risk | Split conformal or selective classification on the task head. One method, done properly. |
| Trace | Full execution record with decomposed latency. Non-negotiable — it is the measurement instrument. |
| Harness | Benchmark runner enforcing equal tuning budget across arms. |

## Out of scope for MVP

Bridges · multimodal · marketplace · process/hardware fault isolation
(F1/F2) · Core migration · safety interrupts · regulated deployment ·
cumulative drift gating · Rust rewrite of the planner · GPU multi-tenancy ·
quantised Wing variants · any registry above ~50 entries.

Several of these are load-bearing for the *paper*. None is load-bearing for
the *first question*. They come later, in the order set by `03_BUILD_ORDER.md`.

## Deliberately small numbers

- Registry: **20** real + proxy entries is enough to exercise retrieval.
- Candidates after stage 1: **~8**.
- Planned graph: **1–3** Wings.
- Beam width: **4**.

Do not scale any of these until the mechanism works at this size. Scaling an
unproven mechanism only buys slower failure.

## What "MVP done" looks like

A single command runs every planner arm over the benchmark, under equal
tuning budget, and emits one table: task score, coverage, p50/p95 latency
with every term broken out, active FLOPs, graph size, class-admission rate,
and oracle regret against the declared restriction.

The MVP succeeds if that table is **trustworthy**, whichever way it falls.
A table showing NICE-R losing is a successful MVP.
