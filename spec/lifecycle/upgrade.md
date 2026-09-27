# Upgrade operations

Developer: Koushik Das - Version 0.1.0

An upgrade is any change to an installed Wing version, a graph, a
calibration class, or the Core. The smallest upgrade that changes a
served graph is a promotion event and follows the promotion protocol.

## 1. Wing version upgrade (same Core epoch)

Steps:

1. **Package.** The new artifact ships with a manifest declaring the same
   core_epoch, a new semver, and an artifact_hash of its own bytes.
2. **Load and validate.** The registry parses the manifest against the
   allow-list. Rejected at load: any host-owned field, an epoch mismatch,
   an unsatisfiable dependency diamond, a hash mismatch.
3. **Install side-by-side.** The new version is installed without touching
   the pinned live route. Version resolution returns exact versions;
   nothing auto-resolves to "latest".
4. **Shadow.** Over traffic fraction s the new wing runs in shadow graphs;
   outputs are logged counterfactually, never served.
5. **Evaluate.** Collected traces feed the estimators (U_hat, A_hat), the
   metrics, and the promotion test.
6. **Promote only if BOTH gates hold:**
   - step drift:   P(t) - P(t-1) >= -epsilon
   - origin drift: P(t) - P(0)   >= -E
   The promotion is a promotion event: the route now executes a different
   graph, so behavior is allowed to change, and the promotion interference
   matrix is measured on exactly those routes.
7. **Re-admit.** The new route's graph must belong to a registered
   calibration class with a valid calibrator. Outside every class: the
   promoted graph is shadow-only, which means the promotion did not
   actually happen — production stayed on the pinned route.
8. **Audit.** The promotion, its evidence, and the interference matrix are
   recorded; the wing version in the live binding is updated in the
   trace schema.

Version discipline: Wing versions are real semver and move. The PROJECT
version is 0.1.0 and never moves — an upgrade of the codebase is not a
project version bump.

## 2. Calibration class registration (a new graph family is served)

1. Shadow graphs accumulate evidence for a proposed class: a family of
   graphs sharing one calibrator.
2. The calibrator is fitted on held-out data; coverage must hold at the
   target level under SELECTION (the routing procedure is part of what is
   validated).
3. Registration is a promotion event with its own audit record.
4. Class coverage is a first-class metric: if the planner proposes graphs
   that rarely fall in registered classes, the system is under-registered
   or over-free.

## 3. Core migration (epoch e -> e+1)

NOT an upgrade in the Wing sense — a migration event:

1. Compatibility analysis of every Wing, Bridge, Head, route, calibrator,
   calibration class and benchmark snapshot against the new latent/type
   contract.
2. Wing retraining or verification through the new Core.
3. Bridge revalidation; route revalidation; calibration renewal;
   benchmark regeneration.
4. Rollout as a separate deployment epoch. One deployment, one Core
   major version — version diamonds on a large Core are
   hard-unsatisfiable, not merely expensive.
5. MigrationCost = w1*N_retrain + w2*N_revalidate + w3*N_recalibrate +
   w4*T_engineering is measured and reported as a limitation (Experiment
   J), not hidden.

## 4. Planner / estimator upgrade

- Any change to U_hat or A_hat is a change to a research component:
  evaluated as an estimator (calibration of the predictor) on held-out
  data with the same protocol as every arm, under the equal-tuning-budget
  rule. A better estimator must be shown better on the same budget, not
  just tuned longer.

## 5. Operational upgrade checklist (pre-flight)

- [ ] manifests re-validated against the allow-list
- [ ] epoch pinned, diamonds resolved
- [ ] new wheels load and pass test vectors
- [ ] shadow budget s accounted in efficiency numbers
- [ ] promotion gates (step AND origin) evaluated, not assumed
- [ ] |A|max recomputed if n or delta changed
- [ ] calibrators re-fitted where the binding changed
- [ ] trace schema covers every new latency term
- [ ] nothing pushed; commits proposed, owner commits (GPG-signed)

The stage gates stay in docs/03_BUILD_ORDER.md: each stage has an exit
gate, and no stage starts before the previous gate is green.