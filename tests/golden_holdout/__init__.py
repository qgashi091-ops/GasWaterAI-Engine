"""GOLDEN HOLDOUT -- W-001..W-010, STRICT EVALUATION-ONLY.

Mirrors the isolation discipline already established on the Base44 side
(qgashi091-ops/gaswaterai-base44-app, base44/shared/goldenTestHarness.ts):
this package is the ONLY place in GasWaterAI-Engine that may read golden
plan bytes or golden ground truth, and an automated test
(tests/test_golden_holdout_isolation.py) fails the suite the moment any
`app/` module or training script references it.

NEVER import this package from `app/` (PlanFacts, component evidence,
Detector v1 training/inference, canonical inventory, the rule engine, or
the vision fallback) or from any script that prepares training/tuning data.
The only legitimate callers are `scripts/run_golden_evaluation.py` and this
package's own tests.

ORDERING DISCIPLINE (enforced by convention here, by the caller in
practice): a caller must run the frozen engine and obtain its output FIRST,
and only call `load_ground_truth()`/`load_raw_findings()` AFTER -- never
before, and never pass ground truth into any engine code path.
"""
