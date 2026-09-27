# Code must read as human-written

Developer: Koushik Das · Version 0.1.0

This is a standing requirement, not a preference. Code, comments, docs, and
commit messages all count.

## Never

- No AI attribution anywhere. No `Co-Authored-By` trailer, no "Generated with"
  line, no tool name in a commit or a file header.
- No emoji, in code, comments, commits, or docs.
- No changelog blocks at the top of files. Git holds the history.
- No decorative section banners (`# ===== SECTION =====`).
- No comment that restates the line below it.

## Comment discipline

Comments explain **why**, and only where the why is not obvious.

```python
# Bad — restates the code
# increment the counter
count += 1

# Bad — the tell. Nobody writes this.
# Note that we carefully handle the edge case here to ensure robustness
if not candidates:
    return None

# Good — explains a decision a reader would otherwise undo
# Falling back to the nearest admissible graph rather than abstaining:
# an abstain here would hide the class gap from the coverage metric.
graph = nearest_admissible(graph, classes)
```

Match the comment density of the surrounding file. A function with a comment
on every line reads as generated regardless of content.

## Vocabulary to avoid

These words are rare in real engineering prose and common in generated prose:
*comprehensive, robust, seamlessly, leverage, utilise, delve, crucial, vital,
ensure that, it is important to note, powerful, cutting-edge, best-in-class.*

Prefer the plain verb. "We check X" beats "we ensure that X is validated".

## Naming

Names come from the domain, not from tutorials. `candidates`, `graph`,
`binding`, `admission` — not `data`, `result`, `helper`, `process_item`,
`my_list`. If a name could belong to any program, it is the wrong name.

## Structure

- Do not wrap everything in `try/except`. Catch what you can act on.
- Do not add a config option for something with one caller.
- Do not write an abstract base class with one implementation.
- Leave a rough edge where the domain is rough. Over-uniform code reads as
  generated because real constraints are not uniform.

## Commits

Imperative subject under ~60 characters. Body explains why, not what.

```
Reject manifests declaring host-owned metrics

A vendor-supplied reliability field would feed the planner directly,
so the validator now works from an allow-list. A deny-list loses this
game as soon as someone picks a new field name.
```

No trailers. No emoji. No "this commit adds".

## Check

```bash
python tools/lint_human.py
```

Word-boundary matching on the vocabulary list, plus checks for emoji,
attribution trailers, changelog headers, and banner comments. Exits non-zero.

This file will trip its own checker, because it quotes every forbidden
pattern. Exempt `docs/09_HUMAN_CODE.md` by path when you write the tool, and
exempt nothing else. An exemption list that grows past one entry means the
checker is wrong or the code is.
