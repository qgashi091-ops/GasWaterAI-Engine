# Legend + Vector-Geometry Structural Symbol Benchmark — Report

## Goal

Before building a trained symbol detector or annotating hundreds more
candidates by hand, this benchmark tested a specific, narrower hypothesis:
that a plan's own legend, combined with v0.1's existing vector geometry and
pipe graph, can identify components reliably — without any raster template
matching (v0.2/v0.3 already showed that alone doesn't generalize well),
without an LLM for classification, and without training anything.

No detector was trained. No LLM classified a component. No new candidates
were annotated. No plan outside the 5 selected below, and no plan among
W-001..W-010 (a separate holdout in a different repository, never touched),
was processed.

## 1. Selected plans and selection reason

**Selected: DEV-01, DEV-03, DEV-04, DEV-05, DEV-06.**

Selection criteria, applied in this order (`app/legend_structural_benchmark/select_plans.py`):
1. The plan must have at least one candidate with an EXISTING human
   annotation in the hosted tool's database — a presence check only
   (candidate_ids, never label content) — otherwise Phase C would have
   nothing to score.
2. Maximum feasible `style_family` diversity among plans passing (1).
3. A usable symbol legend, confirmed by actually running
   `legend_detection.detect_legend_candidates` +
   `legend_entries.classify_candidate_rows` against the real PDF.

**Honest limitation on family diversity.** Only 5 of the 14 manifested DEV
plans have any human annotation yet — review has so far reached DEV-01,
DEV-03, DEV-04, DEV-05, DEV-06 (in manifest order) — and those 5 span only
3 distinct `style_family` values (`STANDALONE-1`, `STANDALONE-3`, `FAM-4`
×3), not 5. Every other manifested plan has zero annotations and could not
have contributed anything to Phase C regardless of its legend quality or
family, so real scorable evidence was prioritized over family diversity.
Full counts are in `data/dev_plans_v04/legend_structural_benchmark/selected_plans.json`.

| Plan | style_family | total candidates | already annotated | usable legend confirmed |
|---|---|---|---|---|
| DEV-01 | STANDALONE-1 | 185 | 159 | yes (41 entries, 14 usable) |
| DEV-03 | STANDALONE-3 | 68 | 67 | yes (38 entries, 35 usable) |
| DEV-04 | FAM-4 | 65 | 63 | yes (20 entries, 14 usable) |
| DEV-05 | FAM-4 | 92 | 90 | yes (22 entries, 4 usable) |
| DEV-06 | FAM-4 | 112 | 8 | yes (22 entries, 4 usable) |

## 2. Holdout: how it was technically enforced

Phase A (`scripts/legend_benchmark_phaseA_predict.py`) and its whole import
graph (`app/legend_structural_benchmark/*`) read only two local inputs: raw
plan PDFs and `manifest.json` (for plan selection and, later, for Phase C's
bbox join back to `candidate_id` — never for its own `class` field, which
is `null` for all 837 records anyway; the 387 real annotations live only in
the hosted tool's separate database). Nothing in this import graph imports
`ArtifactData` or any hosted-database client.

Phase B (`scripts/legend_benchmark_phaseB_freeze.py`) hashes `predictions.json`
(SHA-256), and — before writing that hash — regex-audits every Phase A
source file for actual executable patterns that would read a human label
(an `ArtifactData` import/call, a `["class"]`/`.get("class")` access, the
same for `verification_status`/`annotator_note`, or an `open()` of an
annotations-dump directory). All five patterns came back clean; see
`data/dev_plans_v04/legend_structural_benchmark/freeze_manifest.json`
for the literal audit result.

Only Phase C (`scripts/legend_benchmark_phaseC_score.py`) reads human
labels, and only after re-verifying `predictions.json`'s SHA-256 still
matches the frozen value — an integrity check that would abort the script
if predictions had been touched after freezing.

## 3. Structural-fingerprint method

New code (`app/legend_structural_benchmark/vector_features.py`), not a
reuse of v0.1's `symbols.py` (which deliberately stops at
bbox/curve-presence/primitive-count) or of v0.2/v0.3's raster template
matching (`symbol_templates.py`, `legend_intelligence.py`'s
`cv2.matchTemplate` stage — never imported here). For any bbox (a legend
icon zone or a schema-area candidate), a `PageVectorIndex` built once per
page from `page.get_drawings()` computes:

- line/curve/rect/quad item counts, via **exact segment-vs-bbox clipping**
  for line-like items (not an area-overlap ratio — a real bug this
  benchmark found and fixed: axis-aligned lines have a zero-area bounding
  box, so an area-ratio test always discarded them; see the fix's
  docstring in `vector_features.py`)
- a length-weighted, 12-bin (15°) undirected angle histogram, **circularly
  shifted** so its heaviest bin sits at index 0 — a real but limited
  rotation-invariance (cancels rotations that are roughly a multiple of one
  15° bin, not every continuous angle; stated plainly, not overclaimed)
- bounding-box aspect ratio (`max(w,h)/min(w,h)`)
- a true line-segment crossing count (proper interior intersections only)
- "curviness" (fraction of items that are Bezier curves) as an honestly-named
  proxy for circular/arc content — not a circle fit

`structural_similarity()` combines six sub-scores into one `structural_score`
(0..1) with **fixed weights chosen before Phase A ever ran on real data**
(angle 0.30, composition 0.20, size 0.15, count 0.15, crossing 0.10,
curviness 0.10) and never adjusted afterward. Text and graph evidence are
kept as **separate** corroboration signals (`text_support`, `graph_support`)
— per the task's own instruction, graph position may support a match but
must never turn a weak structural score into a confident one.

Decision thresholds (also frozen before Phase A ran, by analogy to v0.2/v0.3's
own `PLAN_FACT_SCORE_MIN=0.90`/`PLAN_FACT_MARGIN_MIN=0.12`, adapted to this
benchmark's different score scale): `COMPONENT_FACT` requires
`structural_score>=0.75`, `margin>=0.12` **and** at least one corroborating
signal (`text_support>=0.5` or `graph_support` in `{at_endpoint, at_branch}`);
`COMPONENT_CANDIDATE` requires only `structural_score>=0.60`; everything
else is `UNRESOLVED`. None of these numbers were touched after seeing
Phase C's scores.

## 4. Frozen predictions

648 prediction records across the 5 plans, frozen by Phase B:

- **SHA-256: `107ac9961c54b700bda0f6cc1fbb8d274618641179c4e347dacad79722a39d8f`**
- File: `data/dev_plans_v04/legend_structural_benchmark/predictions.json` (829,981 bytes)
- Frozen at: 2026-09-29T19:24:45Z, at git commit `45cd8d6...`
- Per-plan status counts:

| Plan | COMPONENT_FACT | COMPONENT_CANDIDATE | UNRESOLVED | EXCLUDED_LEGEND_REGION | EXCLUDED_TITLEBLOCK_OR_TABLE | EXCLUDED_WASTEWATER |
|---|---|---|---|---|---|---|
| DEV-01 | 1 | 177 | 0 | 3 | 19 | 28 |
| DEV-03 | 11 | 57 | 0 | 4 | 1 | 1 |
| DEV-04 | 0 | 93 | 0 | 12 | 3 | 0 |
| DEV-05 | 0 | 95 | 22 | 0 | 5 | 0 |
| DEV-06 | 0 | 94 | 19 | 0 | 3 | 0 |

Note this table reflects a corrected re-freeze: the first Phase A run had a
real coordinate-space bug (see section 9) that made DEV-03's catalog
entirely empty; fixing it changed DEV-03's SHA-256 and status counts before
any human label was ever read — the freeze/re-freeze cycle itself never
touched Phase C.

## 5. Human ground-truth scope

387 human annotations exist in total, and (per section 1) all 387 belong to
these same 5 plans — every one of them was matched back to its original
`candidate_id` by exact bbox join (100% match rate after fixing the
rotation-space bug in section 9; 0 candidates were "labeled but not
reproduced by the fresh v0.1 rerun").

## 6. Overall precision / recall / F1

| Metric | Value |
|---|---|
| Overall accuracy (exact class match) | 8.27% (32/387) |
| Macro precision | 26.4% |
| Macro recall | 11.3% |
| Macro F1 | 9.6% |

These numbers are computed strictly (`scripts/legend_benchmark_phaseC_score.py`):
every one of the 387 labeled candidates in the 5 plans is scored, none
excluded for being a hard case.

## 7. Per-class results

| Class | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| NOT_A_COMPONENT | 296 | 13 | 0 | 283 | 1.00 | 0.044 | 0.084 |
| OTHER_RELEVANT_SYMBOL | 22 | 19 | 312 | 3 | 0.057 | 0.864 | 0.108 |
| küche | 26 | 0 | 0 | 26 | — | 0.0 | — |
| wc up | 20 | 0 | 0 | 20 | — | 0.0 | — |
| dusche | 10 | 0 | 0 | 10 | — | 0.0 | — |
| up verteiler unter wt | 9 | 0 | 0 | 9 | — | 0.0 | — |
| AMBIGUOUS | 3 | 0 | 0 | 3 | — | 0.0 | — |
| secomat | 1 | 0 | 0 | 1 | — | 0.0 | — |

**Not one of the five named component classes (küche, wc up, dusche, up
verteiler unter wt, secomat — 66 ground-truth instances total) was EVER
predicted by name.** This traces to a real, specific cause, not a matcher
weakness alone: in 4 of the 5 plans' own legends (DEV-01, DEV-04, DEV-05,
DEV-06), the usable legend entries are pipe-**fitting**/material codes
("Cr 28", "PE-S 110", "FL 2" — Chromstahl fittings, PE-S pipe, flanges),
never fixture symbols. Only DEV-03's legend actually names plumbing
components ("Motor. Dreiwegventil", "Wasserzähler m³"). A plan's legend
depicting the *pipe network's own parts* rather than the *fixtures a
reviewer labels* means there is often nothing in the catalog for a
küche/dusche/wc-up ground-truth instance to match against at all — every
one of those 66 instances ends up `UNRESOLVED` or mis-bucketed as generic
`OTHER_RELEVANT_SYMBOL`, confirmed in the confusion matrix below.

Confusion matrix (ground truth → predicted, counts):

```
NOT_A_COMPONENT        -> OTHER_RELEVANT_SYMBOL:254  EXCLUDED:27  NOT_A_COMPONENT:13  UNRESOLVED:2
OTHER_RELEVANT_SYMBOL  -> OTHER_RELEVANT_SYMBOL:19  UNRESOLVED:3
küche                  -> OTHER_RELEVANT_SYMBOL:19  UNRESOLVED:7
wc up                  -> OTHER_RELEVANT_SYMBOL:20
dusche                 -> OTHER_RELEVANT_SYMBOL:10
up verteiler unter wt  -> OTHER_RELEVANT_SYMBOL:5  UNRESOLVED:4
AMBIGUOUS              -> OTHER_RELEVANT_SYMBOL:3
secomat                -> OTHER_RELEVANT_SYMBOL:1
```

## 8. Comparison with v0.2 / v0.3 (and this benchmark's own vector approach)

Run against the same 5 plans, unmodified, using this benchmark's own
Phase A parse (`app.plan_analysis.component_facts.build_component_facts`
for v0.2, `app.plan_analysis.legend_intelligence.build_legend_intelligence`
for v0.3):

| Approach | COMPONENT_FACT | COMPONENT_CANDIDATE | UNRESOLVED | Evaluated |
|---|---|---|---|---|
| v0.2 generic raster templates | 0 | 0 | 648 | 648 |
| v0.3 plan-specific raster templates | 0 | 0 | 598 | 598* |
| This benchmark (structural legend matcher) | 12 | 516 | 41 | 648 |

\* v0.3 excludes each plan's own legend-region symbols from evaluation
before matching (this benchmark's own `EXCLUDED_LEGEND_REGION` does the
same, separately), which is why its evaluated count is 50 lower — not a
missing result.

**v0.2 and v0.3, run unmodified against these same 5 plans, produced
ABSOLUTE ZERO `COMPONENT_FACT` or `COMPONENT_CANDIDATE` results — every
single evaluated position came back `UNRESOLVED`.** v0.3 built real
plan-specific templates in every plan (13-44 templates per plan, from
63-152 legend entries per plan), so this is not a legend-detection failure
on v0.3's part either — its own raster template match scores against those
templates simply never cleared its precision-first bar
(`PLAN_FACT_SCORE_MIN=0.90`, `PLAN_FACT_MARGIN_MIN=0.12`) anywhere in this
sample. This benchmark's structural matcher is far more willing to claim a
match (only 41/648 `UNRESOLVED`), but section 7 shows that willingness is
poorly calibrated: it trades v0.2/v0.3's total silence for 516 candidate-tier
claims of which the human labels confirm only 19 were ever a real
`OTHER_RELEVANT_SYMBOL` and zero were ever a correctly-named specific
component — precisely the failure mode this benchmark's own frozen
thresholds (including the corroboration requirement for the top tier) were
meant to guard against, and did not fully prevent.

## 9. False-positive analysis

**312 false positives on `OTHER_RELEVANT_SYMBOL` alone** (precision 5.7%),
overwhelmingly explained by one mechanism: most of these plans' usable
legend "components" are generic multi-line pipe-fitting icons (couplings,
flanges, reducers), which are **not structurally distinctive** from the
countless ordinary pipe junctions, elbows, and crossings scattered across
the schema area. A junction of 3-4 straight lines at roughly similar
angles scores well against a generic fitting-icon fingerprint essentially
everywhere it occurs on the page — this is a genuine limitation of
line-count/angle/aspect-ratio features against non-distinctive symbols, not
an implementation bug.

A second, smaller contributor: two of the five plans' legend candidates
(DEV-05, DEV-06) picked up **project title-block text** ("Plan-Name:
13926-S-SS-M", "- 1:50 Massstab: Rev. A:") as spurious `USABLE_TEMPLATE`
legend entries — `legend_entries.py`'s ink-presence check found *some*
nearby ink (likely a title-block rule or logo fragment) next to that text
and mis-classified it as an icon. These entries then acted as extra,
meaningless catalog fingerprints, adding to the false-positive pool.

Two real, non-label-dependent bugs were found and fixed *during* Phase A
development (both **before** Phase B ever froze a hash, and independently
re-verified deterministic afterward):

1. **Zero-area line bbox bug**: an area-overlap-ratio test on an
   axis-aligned line's bounding box is always 0/near-0, since such a line's
   bbox has zero width or height — silently discarding nearly all line
   evidence. Fixed by exact segment-vs-bbox (Liang-Barsky) clipping.
2. **Display-vs-raw coordinate space bug**: `legend_detection.py`/
   `legend_entries.py` work in **display space** (correct, since that's how
   a human reads a rotated page); this benchmark's own vector index is
   built from `page.get_drawings()`, which PyMuPDF always returns in the
   PDF's **raw**, pre-rotation space. On DEV-03 (rotation 90°) this silently
   fingerprinted the wrong region for every legend entry, leaving its
   catalog empty (0 usable entries) until fixed by converting legend/
   title-block bboxes to raw space via the inverse rotation matrix before
   any vector lookup.

Both fixes changed *how a bbox is looked up*, never the identity or
decision thresholds themselves, and both were made before Phase B's freeze
— the frozen SHA-256 in section 4 is the POST-fix, final result.

## 10. 10-run reproducibility

DEV-04 (unrotated) run 10 times end-to-end: identical SHA-256 of the full
result (`29d1aa3e...c3f23f`) and identical `stable_match_id` sets on every
run. DEV-03 (rotated 90°, the plan most exposed to the coordinate-space fix
above) additionally run 3 times: identical SHA-256
(`68578f10...ee27497`) every time. No LLM was involved anywhere in this
pipeline.

## 11. Performance / RAM

Full 5-plan Phase A run: **86.9 seconds total, peak RSS 338.5 MiB**.
Dominated by DEV-01 (69.7s of the 86.9s) — a single, very dense page
(5612×1672pt, ~101,355 raw vector drawings, all pure line strokes) whose
v0.1 graph-building and this benchmark's own crossing-count computation are
the main cost; the other 4 plans each complete in 3-6 seconds.

## 12. Verdict

**TRAINED DETECTOR REQUIRED.**

Per the task's own decision gate: precision is not high (`OTHER_RELEVANT_SYMBOL`
precision 5.7%; only `NOT_A_COMPONENT`, at 100% precision but 4.4% recall,
is precise — and only because it almost never fires), so this does not meet
"LEGEND-STRUCTURE USEFUL AS CORROBORATION ONLY" (which requires high
precision, low recall). Recall for every one of the five actually-relevant
component classes is exactly 0%. This benchmark's own structural matcher —
built specifically to go beyond v0.2/v0.3's raster template matching using
richer vector geometry — still cannot reliably distinguish real components
from ordinary pipe geometry, and the underlying data problem (these plans'
own legends describe pipe *fittings*, not the *fixtures* a reviewer
actually labels) means more engineering on this same approach would not by
itself fix the recall gap. A trained detector, or a fundamentally different
source of fixture-identity evidence than these plans' own legends, is
required.

## Reproducing this benchmark

```
python3 scripts/legend_benchmark_phaseA_predict.py   # Phase A + B combined entry is separate:
python3 scripts/legend_benchmark_phaseB_freeze.py
python3 scripts/legend_benchmark_phaseC_score.py <fresh_annotation_dump_dir>
```

`predictions.json` and the raw annotation dump are intentionally excluded
from the repository (see `.gitignore` addition) — they contain per-candidate
bbox/page detail that adds no value outside this report and, for the
annotation dump, would duplicate human-authored content that already lives
in the hosted tool's own database. `freeze_manifest.json` (holding the
SHA-256 and holdout audit) and `phaseC_score_report.json` (the full,
non-hashed scoring detail) are small enough to keep and are committed.
