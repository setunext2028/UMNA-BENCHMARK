# Memory graph

Developer: Koushik Das · Version 0.1.0

Sessions end. The reasoning behind a decision is worth more than the decision,
and it is the first thing lost. `memory/` is where it survives.

## Shape

One fact per file. `memory/INDEX.md` holds one line per file and nothing else.

```markdown
---
name: <short-kebab-slug>
type: decision | invariant | seam | trap | result | open
stage: S0..S7
---

<the fact, stated plainly>

**Why:** <the reasoning, including what was rejected>
**How to apply:** <what someone should do differently because of this>

Related: [[other-slug]] [[another-slug]]
```

Link freely with `[[slug]]`. A link to a file that does not exist yet is
fine — it marks something worth writing, not an error.

## The six types

| Type | Write one when | Example |
|---|---|---|
| `decision` | You chose between real alternatives | Python-first for MVP, Rust deferred to S6 |
| `invariant` | Something must stay true and is easy to break | Admission sits between plan and execute |
| `seam` | An interface's real shape differed from the doc | `WingOutput` needed provenance for the trace |
| `trap` | You lost time to something non-obvious | A trap graduates into `docs/12_TRAPS.md` |
| `result` | A number came out of a run | Seeds, conditions, and which way it fell |
| `open` | A question you could not close | Blocks a stage, or just noted in passing |

## What not to write

Do not record what the repo already says. Code structure, function
signatures, what a file contains, what you did this session — all of that is
recoverable by reading. Memory is for what is **not** recoverable: why an
obvious approach was rejected, what a measurement actually showed, which trap
cost an afternoon.

If you catch yourself writing "implemented X in Y" — that is a commit
message, not a memory.

## Discipline

- Check `INDEX.md` before deciding anything that smells previously decided.
- Update the existing file rather than adding a near-duplicate.
- Delete memories that turn out to be wrong. A stale invariant is worse than
  no invariant.
- A `result` written before the run finished is a lie. Write it after.

## Seeded now

- `stage-pointer` — where work stopped and what resumes it. Update every session.
- `why-python-first` — the S6 Rust deferral and its reasoning.
- `admission-between-plan-and-execute` — the invariant most likely to be
  refactored away by someone tidying the call path.
