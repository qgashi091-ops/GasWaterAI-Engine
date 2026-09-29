# v0.4 Annotation Dataset Cleanup — Report

Triggered by the domain expert's manual review of >400 of the 837 candidates
in the hosted annotation tool, which surfaced two suspected systematic
dataset-quality problems: (1) an impression of being asked about the same
physical object repeatedly, and (2) many candidates concerning
wastewater/drainage (Schmutzwasser/Abwasser/Regenwasser/Entwässerung)
content, out of scope for GasWaterAI's drinking-water/hygiene checker.

This cleanup did not reprocess any PDF, regenerate any candidate, retrain
anything, or touch a single existing human annotation. It added two purely
derived, auditable fields to the existing 837 candidate records —
`scope_status` and `dedup_status` — computed once by
`scripts/apply_dataset_cleanup.py` from `manifest.json` alone, and used them
to define which unlabeled candidates the hosted tool still queues for
review. Full methodology and the exact wastewater-label list are documented
in that script's module docstring.

## Step 1 — measurements (before any change)

- Original candidates: **837**
- Human annotations so far: **387** (387 labeled, 0 skipped), verified as
  genuine documents (real timestamps) in the live Artifact database
- Unlabeled: 450
- Class distribution of the 387 labels: NOT_A_COMPONENT 296 (76.5%),
  küche 26, OTHER_RELEVANT_SYMBOL 22 (5.7%), wc up 20, dusche 10,
  up verteiler unter wt 9, AMBIGUOUS 3 (0.8%), secomat 1
- Candidates per plan: DEV-01 185, DEV-06 112, DEV-05 92, DEV-15 89,
  DEV-12 81, DEV-03 68, DEV-04 65, DEV-20 55, DEV-10 32, DEV-13 18,
  DEV-11 17, DEV-17 12, DEV-14 9, DEV-07 2

## Step 2 — physical-instance deduplication

**Methodology:** candidates were grouped by `(plan_id_pseudonymous, page)`;
every same-page pair's `bbox_page_space` was compared by IoU and centroid
distance, plus `graph_association` node/edge-id overlap. A pair counts as
the same physical instance only when `IoU>=0.5`, OR
(`IoU>=0.15` AND a shared graph id AND `centroid_dist<20pt`) — geometric
overlap corroborated by graph position, **never mere visual resemblance**.

**Result: zero duplicate physical-instance candidates found.** 41,309
same-page pairs were checked; the single highest IoU observed anywhere in
the entire dataset was 0.118 (two candidates on DEV-03 p1, sharing no graph
id) — far below either threshold. All 837 candidates are `dedup_status:
CANONICAL`.

This is a measured null result, not an assumption: per the domain expert's
own instruction to "never deduplicate merely because two symbols look
alike," no candidate was merged or removed on this basis. The most likely
explanation for the "same object repeatedly" impression is the wastewater
suggestion-label repeats in Step 3 below — visually similar generic
pipe/insulation crops, each at a genuinely different physical location, not
literal duplicate candidates.

## Step 3 — wastewater/drainage scope filtering

**Methodology:** exact-match against 5 suggestion-label strings confirmed
present verbatim in `manifest.json`'s `engine_suggestions`
(`plan_specific.label` / `generic.label`), all of which describe
wastewater/drainage pipe-**network** material or medium — never a fixture:

- "13 mm armaflex schmutzwasser entlüftungen" (15 candidates)
- "dämmschlauch schmutzwasser sammelleitungen" (21 candidates)
- "geberit isol schmutzwasser fallstränge" (5 candidates)
- "schmutzabwasser" (1 candidate)
- "tropfwasser kondensabwasser" (1 candidate)

No WC/Dusche/Waschtisch/Küche-suggested candidate is ever matched by this
list — verified, none of those labels overlap it. Per the domain expert's
explicit instruction, a fixture that also has a wastewater connection keeps
its own suggestion label and is never excluded by this rule; the exclusion
targets the wastewater/drainage system itself.

**Result: 43 candidates marked `OUT_OF_SCOPE_WASTEWATER`** (794 remain
`IN_SCOPE`).
- 21 already have a preserved human label (18 NOT_A_COMPONENT, 2
  OTHER_RELEVANT_SYMBOL, 1 küche) — untouched, still shown in the tool.
- 22 are still unlabeled — removed from the *future* active queue only; the
  source records are never deleted.

## Step 4 — candidate-generation learning (forward-looking only)

Per the instruction to use the >400 annotations for dataset *selection*,
not model training, the following suggestion labels correlate ≥90% with
NOT_A_COMPONENT (n≥3 labeled each) and are candidates for tightening in a
**future** candidate-generation batch — no retroactive exclusion was applied
for these in this cleanup, since the task scope was limited to duplicates
and wastewater:

cns28 (27 lu) (98%), warmwasserleitung von haus 7 cr 35 (92%), plus several
generic plan-annotation/legend-boilerplate labels ("a", "(bauseits)",
"wasserversorgung", "b", scale/revision boilerplate, "pe-s 110",
"messfühler niveau") each at or near 100%.

## Step 5 — active queue rebuild

No PDF was regenerated, no candidate ID changed, and no existing annotation
progress was affected. The hosted tool (`app.html`) now computes an active
queue on load: an already-annotated candidate is **always** kept (regardless
of scope/dedup status); an unannotated candidate joins the queue only if
`scope_status == IN_SCOPE` and `dedup_status == CANONICAL`. On reopening,
the tool jumps straight to the first unlabeled candidate in that queue, so
the expert is never asked to newly review a known duplicate or an unlabeled
wastewater candidate. Republished to
`https://claude.ai/artifact/NVn2g1hNY48sn291JzUAaW`; the live `annotations`
database was verified byte-for-byte identical (all 387 documents, version 1,
zero diffs) both before and after the republish.

## Step 6 — final counts

| Metric | Value |
|---|---|
| Original candidates | 837 |
| Human annotations so far | 387 (all labeled, 0 skipped) |
| Duplicate physical instances filtered | 0 (measured null result, see Step 2) |
| Wastewater/out-of-scope filtered | 43 (21 already labeled & preserved, 22 removed from future queue) |
| Other low-value candidates filtered | 0 (forward-looking learnings only, see Step 4) |
| Remaining active queue (total) | 815 |
| Remaining active queue (still unlabeled) | 428 |
| Existing labels preserved | 387 / 387 (verified byte-for-byte intact) |
| Estimated additional annotations still needed | 428 |
| Class distribution (unchanged) | NOT_A_COMPONENT 296, küche 26, OTHER_RELEVANT_SYMBOL 22, wc up 20, dusche 10, up verteiler unter wt 9, AMBIGUOUS 3, secomat 1 |

No detector was trained, no new plans were processed, and Base44 was not
modified.
