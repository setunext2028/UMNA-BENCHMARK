# Downgrade operations

Developer: Koushik Das - Version 0.1.0

A downgrade is a controlled return to a previously pinned state. It is
not the inverse of an upgrade: the origin baseline never moves, and a
downgrade that crosses Core epochs is a migration, not a rollback.

## 1. When to downgrade

- Step drift alarm: P(t) - P(t-1) < -epsilon.
- Origin drift alarm: P(t) - P(0) < -E (the cumulative budget broke).
- Measured behavior diverges from the admission profile (a drift alarm on
  production telemetry vs admission-time estimates).
- A wing's measured failure rate or latency moves outside its envelope.
- A decision to demote: the failing component leaves routing eligibility.

## 2. The rollback procedure (same Core epoch)

1. **Resolve.** The registry selects the last pinned graph for the task
   family. Every pinned route is kept; "the previous version" is an exact
   WingRef set, not a fuzzy memory.
2. **Restore artifacts.** The prior manifest names the exact versions and
   artifact hashes. Artifact bytes are content-addressed and stored out
   of git; restore verifies the hash before loading.
3. **Re-admit.** The restored graph goes through admission like any other
   planned graph: it must belong to a calibration class with a currently
   valid calibrator. If the class lapsed in the meantime, the restored
   route is shadow-only until the calibrator is re-fitted.
4. **DO NOT restore the origin baseline.** The origin is permanently
   pinned; that immutability is what makes cumulative drift detectable.
   Rollback restores the previous deployed state, not the origin.
5. **Demote.** The failing wing version is flagged and removed from
   routing eligibility. It returns only through the full evaluation
   path: re-load, re-shadow, re-pass gates.
6. **Record.** The rollback is a promotion event in reverse: it changes
   which graph serves, so the promotion interference matrix is measured
   on the affected routes, and the audit log records why the rollback
   happened.

## 3. Rolling back a bad Wing version

- A v2.1 that regressed a protected task: restore the graph to v2.0 per
  the procedure above. v2.1 stays installed in shadow (its traces are
  evidence), but is demoted from eligibility. If v2.1's failure was
  discovered during calibration-class admission (the new graph never
  admitted), no rollback is needed — production never left the pinned
  route. This is the strongest property of the architecture: nothing
  serves until admission says it may.

## 4. Downgrades that are really migrations

- A rollback to a Wing trained on a different Core epoch is NOT a
  rollback. Because a Wing is numerically entangled with its epoch
  (W_i,e != W_i,e+1 unless compatibility is demonstrated), crossing an
  epoch means: compatibility analysis, retraining or verification,
  revalidation, recalibration, benchmark regeneration, and a separate
  deployment epoch. Run the migration procedure, not the rollback
  procedure.

## 5. Operational downgrade checklist

- [ ] exact prior graph identified (WingRef set), not "latest"
- [ ] artifact hashes verified on restore
- [ ] restored graph re-admitted (class + valid calibrator)
- [ ] origin baseline untouched
- [ ] failing version demoted from routing eligibility, flagged
- [ ] interference matrix entry recorded on affected routes
- [ ] audit log entry states the trigger and the evidence
- [ ] nothing pushed; commits proposed, owner commits (GPG-signed)

Flowchart: flowcharts/08_downgrade_rollback.txt.
Missing fallback at any point terminates in ABSTAIN, never in a guess.