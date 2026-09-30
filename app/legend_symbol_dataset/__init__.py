"""Legend-symbol-only dataset pipeline (v0.4 Phase 7, "ACTIVE ANNOTATION v2").

Replaces the candidate-based active annotation workflow: the only
authoritative source of a new training example is now a plan's OWN symbol
legend, never an arbitrary v0.1-detected drawing candidate. This package
extracts legend entries for all 20 real dev plans, keeps only the ones a
conservative, text-based classifier can positively establish as a potable-
water apparatus/fixture, and proposes a component name directly from the
legend's own text -- for a human expert to confirm or correct.

The prior 837-candidate / 387-annotation dataset (`data/dev_plans_v04/
annotation_dataset/`, `app/dataset_pipeline/`) is untouched by this
package: it remains on disk and in the hosted tool's `annotations`
database collection as historical/audit data, per the domain expert's
explicit instruction, but is no longer fed by any new extraction here.
"""
