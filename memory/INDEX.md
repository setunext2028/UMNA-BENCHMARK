# Memory index

One line per memory. Content lives in the files, never here.
Format and discipline: `docs/08_MEMORY_GRAPH.md`.

- [Stage pointer](stage-pointer.md) — where work stopped, what resumes it. Update every session.
- [Why Python first](why-python-first.md) — Rust deferred to S6; what the deferral buys and what it costs.
- [Admission sits between plan and execute](admission-between-plan-and-execute.md) — the invariant most at risk from a tidy-up refactor.
- [Strict type unification in S1](strict-type-unification-in-s1.md) — exact family and arguments until real S3 latent semantics justify more.
- [Write firewall mechanism](write-firewall-mechanism.md) — how the parameter write firewall enforces Core immutability during Wing training.
- [S3 task family and roles](s3-task-family-and-roles.md) — the deterministic tabular task family and the six wing roles.
- [S3 core training](s3-core-training.md) — Core training procedure, hyperparameters, and probe accuracy results.
- [Split conformal calibration](split-conformal-calibration.md) — split conformal for binary heads, alpha=0.1, holdout coverage verified.
- [Calibration class registry](calibration-class-registry.md) — the ClassRegistry and admission integration.
- [Shadow routing and router learning](shadow-routing.md) — ShadowRouter, RouterModel, promotion gate, evidence collection.
- [Cumulative drift gating](cumulative-drift-gating.md) — DriftGate, step/origin constraints, Tier A/B/C, power budget, interference matrix.
- [Safety interrupts](safety-interrupts.md) — InterruptRouter, 5 types × 5 actions, pre-authorization, calibrated bindings, priority order.
- [Bridges](bridges.md) — GenericBridge, TrainedBridge, BridgeRegistry, O(1) generic fallback, lazy trained bridges.
- [Second modality](second-modality.md) — ImageEncoderWing, ImageClassifierWing, ModalityRegistry, Core epoch frozen.
- [Rust control plane](rust-control-plane.md) — 6-crate workspace (abi, registry, graph, scheduler, replay, core), typed capability contract, wave-parallel executor, trace replay.
