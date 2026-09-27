# UMNA: 12-Month Research & Budget Summary

**Version:** 0.1.0 | **PI:** Koushik Das | **Date:** 2026-09-24

---

## Research Objectives (12 Months)

| Quarter | Milestone | Hypothesis | Go/No-Go |
|---------|-----------|------------|----------|
| **Q1** | Layer C v0.1 (H1 dev) | NICE-R approaches Oracle Pareto on held-out tasks | Coverage ≥ 0.98, \|G\| ≤ 4 |
| **Q2** | H1 Final (frozen) | Routing Pareto protocol sealed | Independent reproduction (V2) |
| **Q3** | H3 v0.1 (CCE dev) | CCE promotion controls cumulative drift | Sequential promotion: drift ≤ ε |
| **Q4** | H3 Final + H2/H4 prep | CCE protocol sealed + fleet comparison design | V2 verification on H3 |

---

## Budget Allocation ($k)

| Category | Q1 | Q2 | Q3 | Q4 | Total |
|----------|-----|-----|-----|-----|-------|
| **Personnel** (1 PI + 1 Eng + 0.5 Research) | 120 | 120 | 120 | 120 | **480** |
| **Compute** (GPU cluster, 2×A100) | 30 | 30 | 40 | 40 | **140** |
| **Storage/Artifacts** (weights, traces, hashes) | 10 | 10 | 15 | 15 | **50** |
| **Verification** (V2 sealed packages, Docker) | 5 | 10 | 10 | 15 | **40** |
| **Travel/Workshops** (2 conferences) | 0 | 15 | 0 | 15 | **30** |
| **Contingency** (10%) | 16.5 | 18.5 | 18.5 | 20.5 | **74** |
| **Total** | **181.5** | **193.5** | **203.5** | **225.5** | **804** |

---

## Deliverables by Quarter

### Q1 — Layer C (H1 Development)
- [ ] Held-out planner test: 10 seeds × 5 fresh task families
- [ ] Full distribution reporting: coverage, \|G\|, latency, FLOPs, Oracle regret
- [ ] Ablation: NICE-R v0.2 components (stop, redundancy, complementarity, Pareto, UCB)
- [ ] Route displacement measurement (JS divergence) on held-out tasks
- [ ] V1 artifacts: run manifests, frozen eval configs, hashes

### Q2 — H1 Final + CCE Prep
- [ ] Freeze H1 protocol (H1_PROTOCOL_V1.md, hash, timestamp)
- [ ] Independent reproduction (V2): external researcher + sealed verifier
- [ ] CCE sequential promotion infrastructure: drift tracking, promotion audit, interference matrix
- [ ] Tier-A capacity computation (|A|_max) before first promotion

### Q3 — Layer CCE (H3 Development)
- [ ] Sequential capability additions: V0 → V1 → V2 → V3
- [ ] Cumulative drift tracking: step + origin per capability
- [ ] Promotion/demotion events logged with interference matrix
- [ ] H3 v0.1 results: drift ≤ ε for protected, target capability improved

### Q4 — H3 Final + Fleet Benchmark Design
- [ ] Freeze H3 protocol (H3_PROTOCOL_V1.md)
- [ ] V2 verification of H3
- [ ] Design H4 fleet benchmark: UMNA+NICE-R+CCE vs LoRA/Adapter fleets at equal compute
- [ ] Grant application package: hypotheses, protocols, V1/V2 evidence

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NICE-R fails to close Oracle gap on held-out tasks | Medium | High | Ablation-driven redesign; fallback to beam+cap |
| CCE promotion too conservative (never promotes) | Medium | Medium | Calibrate τ on dev benchmark; adaptive threshold |
| Sequential drift accumulates despite gates | Low | High | Tier-A origin pinning; Tier-B scheduled regression |
| V2 reproduction fails (environment drift) | Low | High | Sealed Docker/Apptainer images; hash-locked deps |
| Compute budget exceeded | Low | Medium | CPU reference first; GPU only for fleet benchmarks |

---

## Verification Levels

| Level | When | Requirement |
|-------|------|-------------|
| **V1** (Dev) | Q1–Q4 | Frozen protocol + manifest + artifacts + hashes + hardware |
| **V2** (Independent) | Q2, Q4 | External researcher reruns sealed verifier; provides name/date/hw/score/tolerance |
| **V3** (Public) | Post-project | Source release + full reproducibility |

---

## Intellectual Property

| Asset | Status |
|-------|--------|
| Core architecture (frozen) | Trade secret |
| NICE-R v0.2 algorithm | Patent-pending (provisional filed) |
| CCE promotion protocol | Patent-pending (provisional filed) |
| Benchmark suite | Open source (Apache 2.0) post-publication |
| Trained Wings/artifacts | Content-addressed, outside repo |

---

## Success Criteria (End of Year)

| Metric | Target |
|--------|--------|
| **H1** (Routing Pareto) | NICE-R ≥ 0.98 coverage, \|G\| ≤ 4 on 5 held-out families |
| **H2** (Synergy) | Mode B ≥ random on ≥2 task families (or documented rejection) |
| **H3** (CCE) | 4 sequential promotions with step drift ≥ −0.01, origin drift ≥ −0.05 |
| **V2** (Verification) | 2 independent reproductions (H1 + H3) within declared tolerance |
| **Publication** | 1 conference paper (NeurIPS/ICML/ICLR) + 1 workshop |

---

## Key Personnel

| Role | FTE | Responsibility |
|------|-----|----------------|
| PI (Koushik Das) | 1.0 | Architecture, hypotheses, paper writing |
| Research Engineer | 1.0 | Runtime, benchmarks, verification pipeline |
| Research Assistant | 0.5 | Experiments, ablation studies, documentation |

---

## Computing Resources

| Resource | Specification | Access |
|----------|---------------|--------|
| GPU Cluster | 2× A100 80GB, Slurm | Shared (20% allocation) |
| CPU Cluster | 64-core AMD EPYC | Dedicated |
| Storage | 10 TB NVMe (artifacts, traces, weights) | Project namespace |
| Registry | Private artifact store (content-addressed) | Hash-pinned |

---

## Ethics & Compliance

- No human subjects data
- No PII in benchmarks
- Synthetic tabular tasks only
- Safety interrupts tested (EMERGENCY, CONTRAINDICATION, HARD_CONFLICT, SECURITY, LEGAL_POLICY)
- Interrupts NEVER silently replace protected route
- Artifact hashes published for auditability

---

## Contact

**PI:** Koushik Das  
**Email:** [redacted]  
**Repository:** Local-only (umna-core); benchmark evidence → umna-benchmarks (GitHub), KoushikDas/UMNA-Benchmarks (Hugging Face) at publication time