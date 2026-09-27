# Definition of done

Developer: Koushik Das · Version 0.1.0

A change is done when **all** of these hold. Not most.

## Every change

1. `python tools/audit.py` exits zero.
2. `python tools/lint_human.py` exits zero.
3. Tests pass, and at least one new test exercises a failure path, not only
   the happy path.
4. You traced the happy path and one failure path by hand, and can say which.
5. Nothing reads a routing-relevant number from a manifest.
6. No path reaches `execute.run` without passing `admit`.
7. The trace still decomposes — no new work hidden inside an existing term.
8. Any new free hyperparameter is either measured, or tuned within the
   declared budget for every arm.
9. Version strings still read 0.1.0.
10. `memory/` updated if a decision, invariant, trap, or result came out of
    this work. The stage pointer is current.

## Every stage gate

11. The gate written in `03_BUILD_ORDER.md` for that stage is demonstrated,
    not asserted. Show the command and its output.
12. Construction-enforced properties are in CI, not in the results.
13. Measured numbers state seeds, hardware, and conditions.

## Before reporting any result

14. Every arm ran under the same tuning budget, and the budget is printed.
15. Latency is reported per term, including `T_plan`, `T_admit`, and
    amortised shadow cost.
16. Oracle regret names its declared restriction.
17. Replay claims name their level.
18. You answered, in writing: *what would make this result wrong but still
    look right?*

## What does not count as done

- "It compiles."
- "Tests pass" with only happy-path tests.
- A green gate that could not have gone red.
- A number without its conditions.
- A finding that is true by construction.

## Reporting

Say plainly what works, what does not, and what you skipped. If a gate failed,
show the output. If you left part of the scope undone, name it and say why.
A partially finished task reported accurately is more useful than a finished
one reported vaguely.
