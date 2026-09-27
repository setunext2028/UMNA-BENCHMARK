# Version and git

Developer: Koushik Das · Version 0.1.0

## The version is 0.1.0 and it does not move

Every file, manifest, document, and package declares **0.1.0**. The code may
change a hundred times; the version string does not.

- No bumping. Not for a fix, not for a feature, not for a milestone.
- No `vN.N` in filenames. No `_v2`, no `_final`, no `_new`.
- No changelog files, no changelog blocks.
- Do not ask whether to bump. The answer is no.

Progress is expressed with **status words**, not numbers:

```
DRAFT · PROVISIONAL · GATED · SEALED · DEFERRED · REJECTED
```

A stage that passed its gate is `GATED`, not `0.2.0`. A sealed result is
`SEALED`, not `1.0`.

**Why:** a version number implies a compatibility promise to someone outside
the project. There is nobody outside the project. Bumping would encode a
release process that does not exist and would make the manuscript, the
manifests, and the artifacts drift apart. One pinned string keeps them
aligned by construction.

The exception is **Wing versions**, which are real semver and do move — a
Wing is exactly the thing that has an external compatibility contract, with
the Core epoch. Do not confuse the two. Project version: pinned at 0.1.0.
Wing version: semver, meaningful, enforced by the resolver.

## Git

**Two-repo model. Source stays local; benchmark evidence publishes.**

### umna-core (private, local only)
- Never `git push`. Never `git remote add`. Never open a pull request.
- If a command would contact a remote, do not run it.
- Commits are GPG-signed: `git commit -S`
- The owner commits. Assistant stages named paths and proposes message.
- Never `git add -A`. Stage specific files only.
- Author: Koushik Das. No co-author trailer.

### umna-benchmarks (public, owner pushes)
- Separate repository for benchmark evidence only.
- Contains: protocols, configs, results, reports, hashes — **no source code**.
- Owner pushes manually from their machine using their credentials.
- Assistant never touches remotes, tokens, or external services.

### Hugging Face: KoushikDas/UMNA-Benchmarks (public, owner pushes)
- Benchmark artifacts only: manifests, metrics, hashes, allowed sample outputs.
- No source code, no private weights, no credentials.
- Owner uploads manually using their HF token (stored locally, never in code).

**Commits are signed and owned.**

- `git commit -S` — GPG-signed, always.
- The owner commits. You stage the change and propose the message; you do not
  commit unless asked.
- Never `git add -A`. Stage named paths. Build artifacts and caches are easy
  to sweep in by accident and hard to remove afterwards.
- Author is Koushik Das. No co-author trailer of any kind (see `09_HUMAN_CODE.md`).

## Artifacts

Trained Core and Wing artifacts are content-addressed by hash and kept out of
git. The manifest records the hash; the bytes live outside. A repo that
carries weights stops being reviewable.
