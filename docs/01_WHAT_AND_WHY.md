# What we are building

Developer: Koushik Das · Version 0.1.0

## In one paragraph

A runtime where specialised neural capabilities are **installed** rather than
trained-in. A frozen Core provides shared computation. Each capability is a
**Wing**: an independently versioned package with typed inputs and outputs,
its own parameters, and its own resource ceilings. For each input, a planner
called **NICE-R** picks a small typed graph of Wings to execute, under hard
budgets, and a risk layer decides whether the answer is good enough to serve
or whether the system abstains.

## The bet

Almost everything here except NICE-R already has strong prior art — frozen
bases, adapters, adapter fusion, sparse routing, module versioning. So the
project does not compete on those. It competes on whether a **host-controlled
planner** can choose better capability combinations than top-k or greedy
routing, at a cost that does not eat the benefit.

If a frozen base plus a served LoRA fleet with simple routing does as well,
this project is not justified. That is the honest baseline.

## What we are deliberately not building

- Not a better Transformer.
- Not a marketplace (not in MVP, possibly never for regulated use).
- Not a new tensor library or compiler — we sit above existing ones.
- Not a universal Core. One Core epoch, one closed compatibility domain.
- Not a system that claims frozen parameters mean frozen behaviour.

## Vocabulary you must use precisely

| Term | Means |
|---|---|
| **Core epoch** | A frozen Core plus the latent/type contract valid for it. One per deployment. |
| **Wing** | An installable capability package. Not an adapter layer. |
| **Bridge** | A trained fusion of two or more Wing outputs. Post-MVP. |
| **Graph / route** | The set and wiring of Wings executed for one input. |
| **Binding** | `(Core epoch, graph, head, distribution, hardware profile)`. Calibration attaches here, never to a Wing. |
| **Calibration class** | A pre-registered family of graphs sharing a valid calibrator. Only these may serve. |
| **Shadow** | Executed, logged, never served. |
| **Promotion** | The audited act of making a new route live. |

Do not use "confidence ceiling", "gradient firewall", or "interference tensor".
All three were removed for cause. See `docs/12_TRAPS.md`.
