# bench/ and core/

Developer: Koushik Das - Version 0.1.0

## bench/harness.py — run_all()
- **Contract:** runs every planner arm over the benchmark under equal
  tuning budget; prints the per-arm tuning budget in the output.
- **How it works (S5):** for each arm: tune within its declared budget on
  the held-out split, then run over the benchmark, collecting per-input
  traces. The harness is the only code allowed to import anything.
- **Failure modes:** an arm that receives a larger tuning budget than
  another invalidates the comparison (Protocol 9.2) — the harness asserts
  budget equality and prints it.

## bench/report.py
- **Contract:** emits the one table: task score, coverage, p50/p95
  latency with every term broken out, active FLOPs, graph size,
  class-admission rate, oracle regret against the declared restriction.
- **How it works (S5):** aggregates traces; every latency term has its own
  column; amortised shadow cost included; per-arm tuning budget printed.

## core/
- **Contract:** the frozen Core: definition, training, artifact + hash.
- **How it works (S3):** one small Core trained once, frozen, hashed.
  Core and old-Wing hashes byte-identical after a new Wing trains (CI
  assertion). One epoch, one compatibility domain — not universal.

Tool internals are documented in `spec/code/tools.md`.
