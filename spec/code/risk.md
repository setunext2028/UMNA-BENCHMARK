# risk/classes.py and risk/conformal.py

Developer: Koushik Das - Version 0.1.0

The risk package. Imports abi and graph. Imported by admission (S4) and by
nothing in runtime — runtime never imports risk.

## risk/classes.py

### CalibrationClass
- **Contract:** one registered family of graphs sharing a valid
  calibrator, at a target level alpha.
- **How it works:** class id, the binding it calibrates, the calibrator
  artifact (typed in S4), the target coverage/risk level, and a validity
  flag. Membership of the class was established empirically at admission
  time, never asserted.
- **Failure mode:** a class whose calibrator lapses is invalid: exact
  graphs stop being admitted until the calibrator is re-fitted.

### ClassRegistry
- **Contract:** the registry admit consults: one entry per registered
  class.
- **How it works:** dict of class id to CalibrationClass; contains(graph)
  is true when the graph matches a registered, currently valid class. A
  graph outside every class is shadow-only — that is the design, not a
  registry bug.

### register(cls)
- **Contract:** add a class after empirical validation.
- **How it works (S4):** validation precedes registration: held-out
  coverage at the target level must hold; a class is only registered with
  its evidence recorded. Registration is the promotion event for graph
  families.

## risk/conformal.py

### assess(output, binding)
- **Contract:** the post-execution risk test: ACCEPT | ABSTAIN | EXPAND.
- **How it works (S4):** split-conformal or selective classification on
  the task head, fitted per binding. The output's score is tested against
  the binding's calibrated threshold with hysteresis; the decision is
  returned. EXPAND returns to ADMISSION, never to execution — a grown
  graph is a different binding and must be re-admitted.
- **Failure modes:**
  - risk not met, budget remains: EXPAND (bounded, one round);
  - risk not met, no budget: ABSTAIN;
  - conflicting evidence above the action threshold: abstain and escalate;
  - a bare float compared against the threshold without hysteresis or a
    margin: a bug (audit check B10).
- **Boundary:** assess is authoritative; the planning-time abstention
  predictor A_hat is a soft efficiency term and never overrides it.