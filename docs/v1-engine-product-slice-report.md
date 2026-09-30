# GasWaterAI Engine v1.0 — End-to-End Product Slice Report

Scope: `PDF -> PlanFacts -> Components -> Canonical Inventory -> deterministic
professional checks -> structured result`, as a first running product slice,
not a publication-quality ML benchmark. Base44 integration is explicitly out
of scope for this epic.

## 0. What was frozen, what was added

**Frozen, unchanged** (per the epic's own instruction): the v0.1 PDF parser,
deterministic vector graph, `plan_facts.py` topology derivation, text
extraction/association (`text.py`/`association.py`), the privacy scanner
(`app/dataset_pipeline/privacy.py`), and every existing human annotation.
None of these files were edited. `/analyze`'s request/response contract is
unchanged (verified below). The full existing test suite -- 116 tests,
covering every frozen module plus the new vision-fallback fixture tests --
passes unchanged: `116 passed in 243.77s`.

**Added, strictly additive**:
- `app/detector/` — Detector v1 (feature extraction + training + inference).
- `app/plan_analysis/component_evidence.py` — the canonical component-
  resolution fusion layer.
- `app/vision_fallback/` — the controlled vision-fallback interface +
  Anthropic direct-API implementation.
- `app/plan_analysis/canonical_inventory.py` — the canonical inventory
  builder.
- `app/rules/` — the deterministic rule engine + initial rule set.
- `POST /check` in `app/main.py` — a new, separate endpoint; `/analyze` is
  untouched.
- `scripts/train_detector_v1.py`, `scripts/reproducibility_check.py`,
  `scripts/run_golden_evaluation.py`.

## 1. Detector v1 — measured results

Trained only on the 2 classes the earlier Detector POC (Phase 0, see
`docs/v04-detector-poc-phase0-report.md`) found to have adequate multi-
plan/multi-style-family evidence: **kueche** (küche) and **wc_up** (wc up).
Every other class (dusche, secomat, up verteiler unter wt, and every
OTHER_RELEVANT_SYMBOL sub-type) remains **unsupported by Detector v1** — not
trained, not evaluated, reported as such rather than silently dropped.

**What "detector" means here**: a classifier over v0.1's own already-
localized candidate regions (the deterministic vector-graph/symbol pipeline
already produces a bbox; what was missing was TYPE identification for these
2 classes). Model: `RandomForestClassifier(n_estimators=200, max_depth=12,
class_weight="balanced", random_state=42, n_jobs=1)` over a hand-rolled
HOG-style gradient-orientation descriptor + a coarse intensity grid (see
`app/detector/features.py` for why hand-rolled: this environment's
`opencv-python-headless` build does not expose `cv2.HOGDescriptor`). Pinned
seed, model config, and dataset version are stored in
`data/dev_plans_v04/detector_v1/training_config.json`.

**Split**: family-level, via the existing, unmodified
`app.dataset_pipeline.splits.design_split`. Because the 3 families that
carry ALL of this dataset's kueche/wc_up evidence (FAM-4, STANDALONE-1,
STANDALONE-3) are also the largest families in plan-count terms, one global
split call routes all of them into "train" (see `scripts/train_detector_v1.py`'s
own comment) — so the split was computed in two calls (once over only the
label-bearing families, once over the background-only families), forcing a
genuine held-out family for each class while using the identical,
unmodified split function both times.

| Split | Plans/families | kueche | wc_up | background |
|---|---|---|---|---|
| train | FAM-4, STANDALONE-1 | 16 | 9 | 295 |
| validation | STANDALONE-17 (negligible, n=1 wc_up) | 0 | 1 | 2 |
| test (held out) | STANDALONE-3 (DEV-03) | 12 | 11 | 28 |

**Test-set (held-out family) result — the frozen, final number**:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| kueche | 0.0 | 0.0 | 0.0 | 12 |
| wc_up | 0.0 | 0.0 | 0.0 | 11 |
| background | 0.55 | 1.0 | 0.71 | 28 |

Confusion matrix (test): every one of the 51 held-out items was predicted
`background`. Train-set self-fit accuracy was 100% (complete memorization of
25 positive training examples). `class_weight="balanced"` was tried once, as
a principled fix to the training set's own known 295:16:9 imbalance
(decided before looking at any held-out result) — it did not change the
outcome; the result above is frozen, no further tuning was performed.

**Interpretation**: Detector v1, in its current form, **does not generalize
across style families** at this data volume (9–16 positive training
examples per class). This is a genuine, measured negative result, not a
bug — a classifier this small, trained on this few examples, learning shape
statistics that must transfer to a different drafting company's rendering
convention, is exactly the failure mode a data-scarce POC is supposed to
surface honestly rather than hide behind a self-fit number. **Neither
kueche nor wc_up is currently a reliably supported Detector v1 output** —
component identity for these two classes still routes primarily through the
existing text/legend evidence layer, with the detector's opinion recorded
in provenance but never trusted alone (see `component_evidence.py`'s fusion
rule: a lone `trained_detector` proposal with no corroboration is at most
`COMPONENT_SUPPORTED`, never `COMPONENT_FACT`).

## 2. Supported component classes

None of Detector v1's own 2 target classes currently resolve reliably in
isolation (see above). The engine's actual `COMPONENT_FACT`/
`COMPONENT_SUPPORTED` output on any given plan still depends primarily on
the existing (frozen, unchanged) generic-shape and plan-specific-legend
evidence from `component_facts.py`/`legend_intelligence.py`, now merged
with the detector's corroborating opinion where available.
On the W-003 reference fixture (the only reference plan available in this
session; see section 6), 0/68 candidates reached `COMPONENT_FACT` or
`COMPONENT_SUPPORTED` — a plan whose components this engine's whole existing
symbol-recognition stack was already known (from earlier project work) to
resolve weakly.

## 3. Canonical inventory example

```json
{
  "inventory_id": "HC31b974a9cbcd963715f8",
  "component_type": null,
  "resolution": "COMPONENT_UNRESOLVED",
  "page": 1,
  "bbox": [941.76, 1877.76, 1021.56, 1877.76],
  "graph_node_ids": [],
  "graph_edge_ids": ["e170"],
  "medium": "Zirkulation",
  "dimension": null,
  "topology_status": null,
  "cycle_or_dead_end": null,
  "reserve_evidence": false,
  "safety_device_evidence": false,
  "provenance": [
    {"source": "trained_detector", "value": null, "note": "predicted background"}
  ],
  "unresolved_fields": ["component_type", "dimension", "cycle_or_dead_end"]
}
```

`medium` was correctly recovered from nearby plan text ("Zirkulation") even
though component identity itself stayed unresolved on this item — the two
are independent fields, exactly as designed, each carrying its own
`unresolved_fields` entry rather than one item-level pass/fail flag.

## 4. Executable deterministic Water rules

**4 rules**, registered in `app/rules/water_rules.py`, covering the
capability families section 7 of the brief prioritizes (explicit-text-
based, topology/cycle, reserve-text/topology, component-identity):

1. `RULE-001-unintended-loop` — a drawn cycle on a medium NOT labeled
   Zirkulation is `NON_COMPLIANT` (stagnation-risk configuration); a
   Zirkulation-labeled cycle is `COMPLIANT`; unresolved medium is
   `NOT_ASSESSABLE`.
2. `RULE-002-dead-end-stagnation-length` — always `NOT_ASSESSABLE`: this
   engine has no real-world scale calibration (drawn length is in PDF
   points, never converted to metres anywhere in this codebase), so any
   length-dependent stagnation check is honestly unsupported rather than
   silently skipped.
3. `RULE-003-reserve-connection` — a plan-text-named reserve connection
   corroborated by a proven dead-end topology is `COMPLIANT`; without that
   topological corroboration, `NOT_ASSESSABLE`.
4. `RULE-004-safety-device-identity` — a plan-text-named safety device
   (Sicherheitsventil, Rückflussverhinderer, etc.) with a confidently
   resolved (`COMPONENT_FACT`) component identity is `COMPLIANT`; otherwise
   `NOT_ASSESSABLE`.

**Honesty note on scope**: the original GasWaterAI Water rules/check types
and reference cases (`generatePruefstellen`, `vektorGraphMapper.ts`) live on
the Base44 platform itself. Both repositories available in this session
(`qgashi091-ops/GasWaterAI-Engine`, this repo, and `qgashi091-ops/gaswaterai`,
checked explicitly for this epic) were searched and neither contains that
code — `qgashi091-ops/gaswaterai`'s `backend/` is effectively an earlier
mirror of this same Python engine, and its `frontend/` is a generic
scaffold, not the Base44 application. These 4 rules are therefore a fresh,
conservative implementation, not a port of existing rule source code — each
one's `source_reference` says so explicitly rather than citing an invented
SVGW clause number. No professional requirement was invented: every
COMPLIANT/NON_COMPLIANT branch is defensible from either this engine's own
already-documented PlanFacts semantics or a general, named-as-such SVGW
hygiene principle.

## 5. `/check` API example

```
POST /check
Content-Type: multipart/form-data
file=<plan.pdf>

200 OK
{
  "engine_version": "1.0.0",
  "document_fingerprint": "...",
  "inventory": [ ... canonical inventory items, see section 3 ... ],
  "checks": [
    {
      "check_id": "CK2f6ae0b88fbadc3f10cd",
      "rule_id": "RULE-002-dead-end-stagnation-length",
      "result": "NOT_ASSESSABLE",
      "facts_used": [],
      "missing_facts": ["real_world_length", "scale_calibration"],
      "evidence": {"cycle_or_dead_end": "dead_end"},
      "source_reference": "This engine computes topology in PDF/vector coordinate space only -- ...",
      "inventory_id": "HCe62c2123423d345a9330"
    }
  ],
  "diagnostics": { "inventory_stats": {...}, "check_stats": {...}, "timing_ms": {...} }
}
```

`/analyze` is byte-for-byte unchanged in behavior (same code path, verified
by direct call in this session's testing).

## 6. W-001..W-010 Golden benchmark result

**BLOCKED — not fabricated.** The 10 real reference-plan PDFs and their
38-golden-finding ground truth are not present anywhere in this session:
- This repo's `tests/fixtures/` holds only `W-003_Referenzfall.Plan.pdf`;
  `git log --all --diff-filter=A --name-only` confirms no W-001, W-002,
  W-004..W-010 file was ever committed here.
- `qgashi091-ops/gaswaterai`'s own `backend/tests/fixtures/reference_plans/`
  directory is explicitly `.gitignore`'d for the same reason (real client
  deliverables) and contains only a README saying so.
- No copy exists elsewhere on this container's filesystem.

`scripts/run_golden_evaluation.py` is written and ready: point it at a
directory holding the 10 real PDFs plus a `golden_findings.json` (which also
does not currently exist anywhere accessible) and it will run the frozen
engine and report TP/partial/FN, unsupported FP, weighted recall,
precision, NOT_ASSESSABLE, and per-rule-family results against the stated
baseline (weighted TP 5.5/38, recall 14.5%, precision 23.9%, 13 unsupported).
Running it now would require either fabricating a result or fabricating the
missing inputs — neither was done.

## 7. Reproducibility result

10/10 independent runs of the full `/check` pipeline (parse -> PlanFacts ->
legend intelligence -> component evidence fusion (detector only, no vision
call) -> canonical inventory -> rule checks) against the same PDF bytes
(W-003) produced **byte-for-byte identical** serialized output (564,055
bytes every time; see `scripts/reproducibility_check.py`). External-model
(vision) observations are the one explicitly-excepted non-deterministic
source and were not exercised in this test (no credential in this
environment — see section 9).

## 8. Runtime / RAM

Measured on W-003 (68 symbol candidates, 1 page): **~15.2s wall-clock**,
**~267MB peak RSS**, for the full `/check` pipeline (parse + PlanFacts +
legend intelligence + component evidence fusion, including one Detector v1
inference call per candidate + generic/legend template matching + rule
checks). Detector v1's own serialized model is **407KB**; a single
inference call is sub-millisecond once the model is loaded (loaded once,
cached process-wide — see `app/detector/infer.py`). No GPU used or needed
anywhere in `/check`.

## 9. Concrete remaining blockers to higher recall

1. **Detector v1 cross-style generalization (the biggest blocker)**: 0%
   held-out recall on both trained classes. Needs either (a) substantially
   more labeled instances per class across more style families (the
   earlier Detector POC's own "targeted annotation" recommendations still
   apply), or (b) a fundamentally different, less shape-fragile feature
   representation — a hand-rolled HOG-style descriptor evidently does not
   transfer across CAD drafting conventions at this data volume. A
   CNN/transfer-learning approach might do better but was not attempted:
   with only 9-16 positive training examples per class it would almost
   certainly overfit even harder than the RandomForest already did.
2. **No real-world scale calibration**: every length/distance-dependent
   professional check (dead-end stagnation length foremost) is
   unconditionally `NOT_ASSESSABLE`. This blocks an entire rule family
   regardless of any other improvement.
3. **No access to the original SVGW/Base44 rule corpus**: the 4 rules
   implemented here are a conservative, fresh implementation, not a
   verified port. Real coverage expansion requires either restoring access
   to that corpus (a repository/platform export not available in this
   session) or an explicit, reviewed set of new rule specifications from a
   domain expert.
4. **Golden evaluation is entirely blocked**: without the W-001..W-010
   files and the 38-golden-finding ground truth, this epic's own stated
   purpose ("measure the first external-engine product baseline") cannot
   be measured at all yet. This is the single most consequential blocker
   for judging whether the whole v1.0 slice is actually competitive with
   the stated baseline.
5. **Vision fallback is untested against a live endpoint**: implemented
   and unit-tested against fixtures only; needs a real
   `GASWATERAI_VISION_API_KEY` in a deployment environment before its
   real-world behavior (latency, accuracy, cost) is known at all.

## Verdict

**`ENGINE v1 DOES NOT YET OUTPERFORM BASELINE`**

The end-to-end slice runs, is deterministic, and is honestly measured
end-to-end — but two of its three headline capabilities cannot yet be shown
to work: Detector v1 has a measured 0% held-out cross-style recall (section
1), and the Golden benchmark comparison against the stated baseline could
not be run at all in this session because the required real reference data
does not exist here (section 6). The rule engine itself is real and
executes correctly, but only 4 conservative rules exist and they were not
ported from a verified source corpus (section 4). Concluding
`ENGINE v1 PRODUCT SLICE WORKS` or `WORKS BUT COVERAGE TOO LOW` would both
overstate what was actually measured; concluding this instead reflects
that the biggest open question this epic was meant to answer (is this
better than the 14.5%/23.9% baseline?) remains genuinely unanswered, not
just weak.
