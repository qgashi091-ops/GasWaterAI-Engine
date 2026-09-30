# Component Detector v1 — POC Phase 0 Report

Goal of this POC: determine whether a small trained detector can reliably
recognize a limited set of potable-water components on previously unseen
real plans. This report covers Phase 0 only (dataset audit) and Phase 1
(class selection), per the task's own stopping rule: training is not
attempted unless at least 3 classes clear the eligibility bar.

Context: handcrafted symbol recognition for this project is finished and
failed at every attempt tried (generic SVGW raster templates, plan-specific
raster templates, structural legend/vector matching, same-plan vector-graph
re-identification — see `data/dev_plans_v04/symbol_reid_experiment/report.md`
and `docs/v04-legend-structural-benchmark-report.md`). This POC does not
build another handcrafted matcher; it asks a different, narrower question
about a trained detector.

## Method

`scripts/detector_poc_phase0_audit.py` reads ONLY:
- `data/dev_plans_v04/annotation_dataset/manifest.json` — 837 v0.1-generated
  candidates (geometry + provenance only; `class`/`verification_status` in
  this file are always the pre-annotation seed, never ground truth)
- `data/dev_plans_v04/legend_symbol_dataset/legend_symbols.json` — 341
  extracted legend icons (geometry + provenance only, same caveat)
- a fresh dump of the live `annotations` collection (426 human-labeled
  documents) and the live `legend_symbol_reviews` collection (150 human
  reviews), both from `https://claude.ai/artifact/NVn2g1hNY48sn291JzUAaW`

No automatically-proposed label (`engine_suggestions`, `proposed_component_name`,
`relevance_reason`) is ever counted as ground truth — those fields are only
read for context. Only a human `class` (annotations) or `decision` +
`confirmed_name` (legend reviews) counts as truth. `AMBIGUOUS` is excluded
from truth entirely. `NOT_A_COMPONENT` is negative/background evidence only,
never a positive class.

## Phase 0 — dataset audit

### Pool A: `annotations` (426 labeled, real in-situ candidate instances)

| Class | Count |
|---|---|
| NOT_A_COMPONENT (negative evidence) | 325 |
| küche | 28 |
| OTHER_RELEVANT_SYMBOL (heterogeneous, see below) | 27 |
| wc up | 21 |
| dusche | 10 |
| up verteiler unter wt | 9 |
| AMBIGUOUS (excluded from truth entirely) | 5 |
| secomat | 1 |

Positive classes, with unique plans and unique style families (each
instance is an independently-generated, geometrically distinct candidate —
the project's existing IoU/graph-id duplicate audit already found **zero**
duplicate physical instances across all 41,309 same-page candidate pairs in
this exact 837-candidate set, `docs/v04-dataset-cleanup-report.md` Step 2;
not re-derived here):

| Class | Instances | Plans | Style families | Per-family breakdown |
|---|---|---|---|---|
| küche | 28 | 4 | 3 | FAM-4: 9 (DEV-04: 2, DEV-05: 7) · STANDALONE-1: 7 (DEV-01) · STANDALONE-3: 12 (DEV-03) |
| wc up | 21 | 4 | 3 | FAM-4: 9 (DEV-04: 5, DEV-05: 4) · STANDALONE-3: 11 (DEV-03) · STANDALONE-17: 1 (DEV-20) |
| up verteiler unter wt | 9 | 3 | **1** | FAM-4 only: DEV-04: 5, DEV-05: 3, DEV-06: 1 |
| dusche | 10 | **1** | **1** | STANDALONE-3 only: DEV-03: 10 |
| secomat | 1 | 1 | 1 | STANDALONE-1: DEV-01: 1 |

`OTHER_RELEVANT_SYMBOL` (27 instances) is **not one visual class** — it is
a catch-all bucket. Breaking it down by the annotator's own free-text note
(never guessed, never inferred beyond what the human wrote):

| Real component (from annotator note) | Instances | Plans | Families |
|---|---|---|---|
| Verteilbatterie | 3 | 3 | 2 |
| Waschmaschine | 3 | 2 | 2 |
| Badewannenmischer (incl. 1 typo "Badewannenmischrf", normalized) | 3 | 2 | 1 |
| Zirkulationsventile | 3 | 1 | 1 |
| Gartenventil Frostsicher | 2 | 2 | 2 |
| Sicherheitsventil | 1 | 1 | 1 |
| Duschenmischer | 1 | 1 | 1 |
| Waschtrog | 1 | 1 | 1 |
| Waschtisch | 1 | 1 | 1 |
| Beschriftung Verteilung (a label, not a component — excluded) | 1 | 1 | 1 |
| (no note, unidentified) | 7 | — | — |

Every sub-type tops out at 3 instances and at most 2 style families — none
is close to the bar clarified below.

### Pool B: `legend_symbol_reviews` (150 reviewed, one crop per legend row)

| Decision | Count |
|---|---|
| unclear (never training truth) | 134 (89%) |
| correct | 12 |
| corrected | 4 |

This pool is dominated by noise (89% unclear) — consistent with the earlier
domain-expert finding that triggered the (separately parked) legend-gate
work. The 16 confirmed identifications (`correct` + `corrected`) are:

| Confirmed name | Instances | Plans |
|---|---|---|
| Absperrklappe, Motor. Absperrklappe, KW: 2 X VPE Ø16, Filter, Wasserzähler M 3, Pex-Verteiler 2, WW: 1 X VPE Ø20, Sicherheitsventil, LEGENDE SANITÄR (not a component), Rückschlagklappe, Rückschlagventil, KW: 3 | 1 each | 1 each |
| Waschtisch (2x, corrected from unrelated OCR guesses) | 2 | 1 (DEV-07) |
| Absperrventile (corrected) | 1 | 1 (DEV-07) |
| Abwasser (corrected, wastewater — out of scope) | 1 | 1 (DEV-07) |

Every confirmed legend identification is a **single instance in a single
plan** — none provides multi-plan/multi-style evidence on its own. Note
Sicherheitsventil is confirmed once here (DEV-03) and once more as an
`OTHER_RELEVANT_SYMBOL` annotation instance, also in DEV-03 — i.e. still
only one plan's worth of independent evidence, not two.

## Phase 1 — class eligibility

Eligibility bar applied (not merely "some data exists," but "enough
independent evidence to support a meaningful POC," per the task's own
wording): a class needs (a) enough positive instances to be worth training
on, AND (b) evidence from more than one plan, AND (c) evidence from more
than one style family — because Phase 2 requires splitting by
project/style family and evaluating on a family the detector never trained
on. A class with only one style family cannot have both a non-trivial
train set and a non-trivial *cross-style* held-out test set for that class
at the same time; that is a structural blocker, not just thin data.

| Class | Instances | Plans | Families | Verdict |
|---|---|---|---|---|
| **küche** | 28 | 4 | 3 | **Eligible.** All 3 families have substantial counts (7/9/12) — any one can be held out as test while training on the other two. |
| **wc up** | 21 | 4 | 3 | **Eligible**, with a caveat: family diversity is really FAM-4 (9) + STANDALONE-3 (11) plus a negligible STANDALONE-17 (n=1). Hold out FAM-4 or STANDALONE-3 as the untouched test family, not STANDALONE-17. |
| up verteiler unter wt | 9 | 3 | 1 | **Not eligible.** All 3 plans are the same style family (FAM-4) — no way to build a cross-style test set for this class at all. |
| dusche | 10 | 1 | 1 | **Not eligible.** Single plan, single family — cannot be split at all. |
| secomat | 1 | 1 | 1 | **Not eligible.** One instance total. |
| OTHER_RELEVANT_SYMBOL (as a bucket) | 27 | 8 | 6 | **Not eligible as a single class** — it is not one visual object category (see breakdown above); training a detector on it would ask the model to learn one bounding-box class covering ~9 visually unrelated fixtures. |
| Verteilbatterie / Waschmaschine / Badewannenmischer / Zirkulationsventile / Gartenventil / Duschenmischer / Waschtrog / Waschtisch (sub-types) | 1–3 each | 1–3 each | 1–2 each | **Not eligible.** Every sub-type tops out at 3 instances. |
| Wasserzähler / Absperrklappe / Filter / Sicherheitsventil / Rückschlagventil / Rückschlagklappe / Pex-Verteiler / Motor. Absperrklappe (legend-confirmed) | 1 each | 1 each | 1 each | **Not eligible**, and not even in the current annotation taxonomy as candidate classes yet — no drawn-occurrence candidates have been generated or annotated for these names at all (Sicherheitsventil is the closest, with 1 legend + 1 drawn instance, both in DEV-03). |

**Only 2 classes clear the bar: küche and wc up.** The task requires a
minimum of 3. Per the task's own instruction:

> If fewer than 3 classes have adequate data: STOP BEFORE TRAINING.

No training run was attempted. No detector was built. Phases 2–7 (split,
data prep, model choice, training, evaluation, PlanFacts integration) were
not started, per this stopping rule.

## What would be needed to reach 3 eligible classes

Ranked by how close each candidate already is:

1. **up verteiler unter wt** — has 9 instances across 3 plans already,
   blocked purely on style-family diversity (all FAM-4). Needs roughly
   **10+ more confirmed positive instances drawn from at least one
   different style family** (ideally 2, to match küche/wc up's profile).
   This is a targeted-annotation task, not a re-processing task: candidates
   for this class likely already exist unlabeled in plans outside FAM-4 and
   need review.
2. **dusche** — has 10 instances but all in one plan/family (DEV-03 /
   STANDALONE-3). Needs **roughly 10+ more confirmed instances from at
   least 2 additional plans in at least 2 different style families**.
3. **Gartenventil Frostsicher** (currently folded into
   OTHER_RELEVANT_SYMBOL) — already has the best diversity-per-instance
   ratio of any sub-type (2 instances, 2 plans, 2 families already). Give
   it its own explicit taxonomy class and target **8–12 more confirmed
   instances** across those or additional families.
4. **Verteilbatterie / Waschmaschine** — each already has 2–3 families;
   promoting either to an explicit class and collecting **7–12 more
   instances each** would make it competitive.
5. **Wasserzähler, Sicherheitsventil** (legend-confirmed, real, standard
   fixtures, currently absent from the candidate taxonomy) — collecting
   these would require first generating candidates for these specific named
   components across the other ~19 plans (none exist yet), then annotating
   **~20+ instances across ≥3 plans / ≥2 style families each** to match the
   küche/wc up bar. This is the most work of the options above but targets
   symbols the earlier legend work already independently confirmed as real
   and clearly drawn.

None of this ranking is a decision to select these classes now — per the
task's instruction, no class is selected merely because it is
professionally important; this is only a report of where the fastest,
best-evidenced path to a 3rd eligible class lies.

## Compliance checklist

- No detector was trained.
- No automatically-proposed label was treated as ground truth (enforced in
  code: only `class`/`verification_status=LABELED` and
  `decision ∈ {correct, corrected}` count).
- `AMBIGUOUS` was never used as training truth.
- `NOT_A_COMPONENT` was used only as negative/background evidence, never a
  positive class.
- No existing annotation was relabeled, corrected, or silently changed.
- No PDF was reprocessed; no new candidate was generated.
- Base44 was not touched. W-001…W-010 were not touched. No SVGW rule-engine
  work was done. No new handcrafted symbol matcher was built.
- No deployment.

## Verdict

**`INSUFFICIENT POSITIVE DATA — TARGETED ANNOTATION REQUIRED`**
