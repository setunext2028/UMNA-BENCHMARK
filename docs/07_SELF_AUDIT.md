# Self-audit

Developer: Koushik Das · Version 0.1.0

Run this before saying done. "It compiles" and "tests pass" are the start.

## A · Correctness sweep

1. **Trace the happy path and at least one failure path** through every new
   branch. Say which failure path you traced.
2. **Read the actual file**, not your memory of it. Struct fields, key names,
   and dispatch order are authoritative on disk.
3. **Check both directions of every reference.** A name written in a manifest
   must be readable by the loader, and the loader's expected key must exist in
   the manifest. One-way checking is how dead configuration survives.
4. **Find the consumer.** If you added a construct nothing reads, you added
   dead weight. Name the function that consumes it.

## B · The UMNA-specific checks

5. **Does anything read a routing-relevant number from a manifest?**
   `grep` the manifest keys against the planner's inputs. Any overlap is a bug.
6. **Can a graph reach execution without passing admission?**
   Every call path into `execute.run` must come through `admit`.
7. **Does the trace still sum?** New work means a new latency term or an
   existing one grows. Silent absorption into `T_wing` destroys the experiment.
8. **Did you add a free hyperparameter?** If yes, it must be either measured,
   or tuned within the declared budget for *every* arm. Say which.
9. **Is the new assertion capable of failing?** If it is true by construction,
   move it to CI and do not report it as a finding.
10. **Does any comparison on the control path use a bare float?**

## C · The claim check

Before writing any result, classify it:

- **Construction-enforced** — true because of how the code is written. Goes in
  CI. Never a finding.
- **Measured** — a number from a run. State the conditions and the seeds.
- **Hypothesis** — not yet answered. Say so plainly.

Mixing these up is the failure mode this project was reorganised to avoid.

## D · Adversarial pass

Ask, and answer in writing:

- What would make this result wrong but still look right?
- If a reviewer said "this cannot fail", what would they be pointing at?
- What is the cheapest baseline that would beat this, and has it been run?

## E · Tooling

```bash
python tools/audit.py          # A and B, mechanically
python tools/lint_human.py     # docs/09 style checks
```

Both must exit zero. If a check is wrong, fix the check and say why — do not
add an exception for your file.
