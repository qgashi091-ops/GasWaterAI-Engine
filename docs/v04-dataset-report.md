# v0.4 report — real-plan dataset & annotation pipeline

20 new, real, unedited water-installation plans (pseudonymous ids `DEV-01`
.. `DEV-20` throughout — see §3) were run through the unchanged v0.1/v0.2/v0.3
engine and turned into a human-reviewable annotation dataset. **No detector
was trained.** W-001..W-010 (the existing Golden holdout) were not read,
touched, or counted anywhere in this pipeline.

## 1. Audit of all 20 plans

Per-plan page count, vector-vs-scanned classification, native text
availability, dimensions, parser outcome, graph node/edge counts, symbol
candidate counts, and legend coverage — from
`data/dev_plans_v04/annotation_dataset/pipeline_summary.json`'s `audits`
array (also committed as `audit_report.json`), every row identified only by
its pseudonymous id:

| plan | pages | rot | scanned? | text source | parser | nodes | edges | symbols | legend | usable entries | runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DEV-01 | 1 | 0 | no | native | ok | 462 | 329 | 228 | yes | 49 | 147.8 |
| DEV-02 | 1 | 0 | no | ocr | ok | 424 | 365 | 88 | yes | 46 | 71.4 |
| DEV-03 | 1 | 90 | no | native | ok | 166 | 102 | 74 | yes | 40 | 30.2 |
| DEV-04 | 1 | 0 | no | native | ok | 499 | 395 | 108 | yes | 28 | 32.3 |
| DEV-05 | 1 | 0 | no | native | ok | 530 | 432 | 122 | yes | 18 | 38.6 |
| DEV-06 | 1 | 0 | no | native | ok | 522 | 421 | 116 | yes | 14 | 38.0 |
| DEV-07 | 1 | 90 | no | native | ok | 329 | 232 | 13 | yes | 856 | 513.6 |
| DEV-08 | 1 | 90 | no | native | ok | 602 | 456 | 67 | yes | 155 | 27.1 |
| DEV-09 | 1 | 0 | no | mixed | ok | 188 | 127 | 42 | yes | 107 | 18.3 |
| DEV-10 | 1 | 0 | no | native | ok | 376 | 324 | 94 | yes | 163 | 484.6 |
| DEV-11 | 1 | 0 | no | native | ok | 341 | 286 | 37 | yes | 90 | 16.7 |
| DEV-12 | 1 | 0 | no | native | ok | 661 | 491 | 150 | yes | 112 | 85.4 |
| DEV-13 | 1 | 90 | no | native | ok | 221 | 150 | 41 | yes | 21 | 9.1 |
| DEV-14 | 1 | 270 | no | native | ok | 529 | 370 | 58 | yes | 156 | 33.5 |
| DEV-15 | 1 | 270 | no | native | ok | 496 | 379 | 94 | yes | 62 | 64.7 |
| DEV-16 | 1 | 0 | no | mixed | ok | 202 | 127 | 59 | yes | 138 | 42.6 |
| DEV-17 | 1 | 0 | no | native | ok | 203 | 140 | 24 | yes | 36 | 8.7 |
| DEV-18 | 1 | 270 | **yes** | ocr | ok | 0 | 0 | 0 | yes | 64 | 30.4 |
| DEV-19 | 1 | 0 | no | native | ok | 419 | 286 | 35 | yes | 72 | 14.5 |
| DEV-20 | 1 | 90 | no | ocr | ok | 161 | 103 | 72 | yes | 30 | 23.9 |

Parser success rate: **20/20**. No failures, no timeouts, no plan skipped.
(Two plans, DEV-07 and DEV-10, initially exceeded the pipeline's first
240s per-plan wall-clock budget — diagnosed by hand with a generous one-off
timeout, both complete correctly and deterministically in ~475-520s; the
budget was raised to 700s for all 20 plans uniformly, not tuned to these
two specifically. See Performance section and `scripts/run_v04_pipeline.py`'s
`PER_PLAN_TIMEOUT_SECONDS` comment.)

All 20 plans are single-page. All 20 are vector-based drawings (hundreds to
hundreds-of-thousands of raw drawing primitives — DEV-07's 327k is the
extreme end, driven by an unusually wide, ~14300pt multi-riser page).
Text extraction is more mixed than "vector vs. scanned" alone suggests:
DEV-18 is the only plan that is both near-zero native text AND low vector
drawing count (`is_likely_scanned`, v0.1's existing OCR-fallback path
carries the whole page). DEV-02 and DEV-20 are richly vector-based
drawings whose text nonetheless required full OCR fallback (`text_source
== "ocr"`), and DEV-09/DEV-16 needed a mix of native and OCR'd text
(`"mixed"`) — plausibly the same "text drawn as vector outlines" CAD-export
phenomenon documented in `candidates.py`'s privacy-layer-3 fix (see §3),
just affecting the WHOLE page's text on these four plans rather than one
title-block field. v0.1's `text.py` (unchanged) already handles this
correctly; it is noted here because it means 5/20 plans, not 1/20, actually
exercise the OCR path in this batch.

## 2. Project/style families

Grouped by shared filename prefix (families.py; see that module's docstring
for the exact rule and why it needs two thresholds together) — the grouping
computation uses the local, never-committed filename mapping, but this
output is pure pseudonym-to-pseudonym and carries no identifying text:

17 style families for 20 plans:

- **FAM-4**: DEV-04, DEV-05, DEV-06 (same project, three riser-schema sheets)
- **FAM-9**: DEV-11, DEV-12 (same project, two revisions/sheets)
- 15 standalone, single-plan families: DEV-01, DEV-02, DEV-03, DEV-07,
  DEV-08, DEV-09, DEV-10, DEV-13, DEV-14, DEV-15, DEV-16, DEV-17, DEV-18,
  DEV-19, DEV-20

No near-duplicate page pairs were detected (perceptual-hash comparison,
`families.py::detect_near_duplicates`) — every plan's page content is
visually distinct, family membership here comes entirely from filename
similarity, not near-identical drawings.

## 3. Privacy findings

Every plan's own native text was scanned (never OCR'd/vision-scanned) for
street-address, Swiss postal-code+town, title-block-field, company-name,
phone, and email patterns (privacy.py) — findings are reported as **kind +
count only**, never as the matched text itself, so this report cannot leak
what it is warning about. Filenames were scanned separately and discarded
immediately after; several of the 20 real filenames contained a street
address or a personal name directly, which is exactly why every artifact
this pipeline produces — including this report — refers to plans only by
their pseudonymous `DEV-xx` id, never their original filename.

**15 of 20 plans' own native title-block/legend text** matched at least one
privacy pattern (street address, Swiss postal-code+town, title-block field,
company name, phone, or email — kind+count only, never the matched text
itself, consistent with this report's own no-leak discipline). **1 of 20**
original filenames was flagged as containing a street address. Aggregate:
**281 candidates were excluded entirely** across the batch for overlapping
PII-shaped content (never rendered or saved — see the three-layer design
below), out of 1522 raw symbol candidates the underlying engine found
before any privacy filtering.

Two further, DIFFERENT privacy bugs were found and fixed while building
this exact report, both beyond the three crop-image layers below — full
account in `app/dataset_pipeline/taxonomy.py` and `candidates.py`'s own
module docstrings, and `docs/architecture.md`:

- **Taxonomy evidence** (`legend_entries[].normalized_label`, aggregated
  into `classes.json`/`pipeline_summary.json`'s `taxonomy` block) is
  separate TEXT, not a crop image, and inherited none of the three crop
  layers below. A real supplier's phone number, postal code+town, and a
  company name printed directly under a materials list in one plan's legend
  region were found flowing straight toward a committed JSON file before
  `taxonomy.py::is_plausible_component_label` (a mandatory `privacy.scan_text`
  call, plus a separate noise filter for dimension/fixture-table rows) was
  added.
- Fixing that surfaced a SEPARATE bug in `privacy.py` itself:
  `SWISS_PLZ_TOWN_PATTERN`/`COMPANY_SUFFIX_PATTERN` were missing
  `re.IGNORECASE`, so they silently never matched `legend_intelligence.py`'s
  already-lowercased label text — a real postal-code+town pair passed every
  check until this was fixed. Both patterns are now case-insensitive.
- A THIRD, independent vector: each candidate's `plan_specific_suggestion`
  (v0.3's own best-matching-legend-entry label, shown as a one-click
  suggestion button in the annotation tool and stored per-candidate in
  `manifest.json`) is populated from the SAME legend-entry text, via a
  completely different code path than either the crops or the taxonomy
  evidence. Auditing this batch's real QC report showed a phone number as
  the single MOST COMMON "suggested label" across all 20 plans' candidates,
  before `candidates.py::_is_safe_plan_suggestion_label` (same
  `privacy.scan_text` check) was added; the fix drops the suggestion, never
  the whole candidate, since the candidate's own crop image is independently
  already verified safe. A final, comprehensive automated scan of every
  string in the three committed JSON output files, after all fixes, found
  zero genuine PII-shaped matches (two loosely-matched false positives on
  v0.1's own internal diagnostic warning text, e.g. "Excluded 6784 of
  7051..." — a segment count, not plan content — confirmed benign by
  inspection).

Design response for the CROP IMAGES specifically (not a text redaction — a
structural, three-layer one; see
`candidates.py`'s module docstring and `docs/architecture.md`'s "Privacy:
three stacked, independent layers" for the full account):

1. Every annotation crop already comes from
   `legend_intelligence.build_legend_intelligence`'s own Phase-4 exclusion
   of any v0.1 symbol candidate that falls inside a detected
   legend/dense-text region — on every plan in this batch, just as on
   W-003, that includes the title block. Found NOT sufficient on its own
   during development: one real plan's unusually large, fragmented title
   block left a gap between two clustered sub-regions that a stray symbol
   candidate fell into, and its rendered crop (caught by manual visual
   review before ever being committed) showed a property owner's name, a
   company's address, phone number and email in plain text.
2. `candidates.py::_crop_contains_pii` — a mandatory, independent,
   content-based regex scan (`privacy.py`'s patterns) of every text span
   that overlaps the crop's own render region, regardless of what layer 1's
   clustering found. Drops the candidate entirely, never flagged-but-kept.
3. `candidates.py::_ocr_pii_zones` — re-checking the exact candidate layer
   2 was added for showed it was STILL present, because that title block's
   specific fields are drawn as vector outlines ("convert text to paths", a
   common CAD-export setting): perfectly readable pixels, but not PDF text
   objects at all, so no text-layer method — however it's called — can see
   them. Layer 3 renders and OCRs the actual pixels around every detected
   legend/dense-text region instead, which is the only way to catch PII
   that was never text in the PDF to begin with.

The annotation tool's own live "larger context" view carries the
equivalent of layers 2 and 3 as well, since it renders a bigger,
on-demand region the offline pipeline never precomputed or checked.

## 4. Total annotation candidates

**837 candidates** generated across the 20 plans (`manifest.json`), each
with: a stable `candidate_id`, its pseudonymous source plan, page, bbox (in
both page space and pixel-space within its own crop), a rendered PNG crop
with bounded context (never the whole page — see `MAX_CROP_SIDE_PT` in
candidates.py, added after an early real crop showed nearly a whole
multi-storey schema), nearby associated text, graph relation, and the
engine's own plan-specific/generic top suggestions where available. Every
candidate starts `UNLABELED`.

## 5. Proposed initial classes and estimated frequency

Proposed from three combined, purely evidence-driven signals (taxonomy.py):
legend-entry frequency, v0.1's own existing backflow-device-code pattern
match (EA/CA/BA/..., reused from label_hints.py, not reimplemented), and
generic-library suggestion frequency on real drawing candidates. No class
below `MIN_SUPPORT` occurrences anywhere in this batch is proposed.

15 proposed classes (`data/dev_plans_v04/annotation_dataset/classes.json`),
ranked by measured evidence count:

| rank | class | evidence count | note |
|---|---|---|---|
| 1 | pex-verteiler | 46 | distributor (task's suggested "Verteiler"); instance numbers normalized away (e.g. "pex-verteiler 45" → "pex-verteiler") |
| 2 | dusche | 27 | shower fixture |
| 3 | küche | 17 | kitchen fixture/sink |
| 4 | coop laden | 12 | **flag for human review** — a named retail tenant/room-use label, not a device; see caveat below |
| 5 | apparate stk. kw ww dim | 8 | **flag for human review** — a fixture-count table's own column header, not a device |
| 6 | up verteiler unter wt | 8 | "flush-mounted distributor under washbasin" — device-adjacent but a descriptive phrase, not a clean class name |
| 7 | BA | 7 | backflow/venting device code (v0.1's existing `label_hints.py` pattern, task's suggested "BA") |
| 8 | 9 mm armaflex | 6 | **flag for human review** — an insulation-material spec, not a device |
| 9 | an bestehenter leitung anschliessen | 6 | **flag for human review** — an installation NOTE ("connect to existing pipe"), not a device |
| 10 | dämmung regenwasser | 6 | **flag for human review** — an insulation note, not a device |
| 11 | kaltwasser | 6 | cold-water media label (pipe/media type, not a discrete component) |
| 12 | kw stk. ww dim | 6 | **flag for human review** — another fixture-table column header |
| 13 | secomat | 6 | a named grease-separator product line — real, distinguishable device, but brand-specific rather than generic |
| 14 | warmwasser | 6 | hot-water media label |
| 15 | wc up | 6 | flush-mounted WC/toilet fixture |

Plus the three fixed classes every candidate can always fall into:
`OTHER_RELEVANT_SYMBOL`, `NOT_A_COMPONENT`, `AMBIGUOUS`.

**Known, honest limitation of this ranking** — flagged explicitly rather
than silently shipped: `taxonomy.py`'s two structural filters (mandatory
privacy scan + a regex/digit-density noise filter for dimension/fixture-
table rows — see `taxonomy.py`'s module docstring) removed the WORST
contamination (the naive, pre-fix version's top 15 was almost entirely
measurement-table noise like "lu: 1"/"ø63"/"dn100", and separately, real
PII — see §3). They cannot, mechanically, distinguish a genuine device
legend row from a room/tenant-use label or a table/column header made of
ordinary words (rows 4, 5, 8, 9, 10, 12 above) — that needs semantic
understanding a regex heuristic does not have. **Before large-scale
annotation, a human should skim this 15-class list in the annotation tool
and drop or rename the flagged rows**; this is exactly the kind of
judgment call `legend_intelligence.py`'s existing, deliberately generic
legend clustering cannot make on its own, on ANY batch of real plans, not
just this one.

## 6. Legend coverage

**20/20 plans** had a legend detected (`legend_intelligence.build_legend_
intelligence`) — no plan in this batch lacked a printed legend entirely.
2257 usable (`USABLE_TEMPLATE`) legend entries total, mean ~113 per plan —
heavily skewed by DEV-07's outlier 856 (its unusually large, wide,
multi-riser page has a correspondingly large fixture-count table swept into
"legend" entries alongside real device rows, consistent with §5's noise
finding); the median across the other 19 plans is closer to 62.

## 7. Recommended train/validation/test split

Family-level (never plan-level) assignment, greedy largest-family-first
targeting 70/15/15 — see splits.py's docstring for why an exact ratio
isn't achievable at this scale and why that's fine:

17 families → **14 train / 3 validation / 3 test** (plan counts, respecting
every family boundary — FAM-4's 3 plans and FAM-9's 2 plans each stay
entirely inside one split):

- **train**: DEV-01, DEV-02, DEV-04, DEV-05, DEV-06, DEV-08, DEV-11,
  DEV-12, DEV-13, DEV-14, DEV-15, DEV-16, DEV-17, DEV-18
- **validation**: DEV-03, DEV-09, DEV-19
- **test**: DEV-07, DEV-10, DEV-20

Not an exact 70/15/15 split by plan count (14/3/3 = 70/15/15 exactly, in
fact, at this scale) — greedy largest-family-first naturally lands close
here only because both multi-plan families happened to fit into train
without needing to be split (they never would be, regardless of ratio).
W-001..W-010 are not inputs to this module and appear nowhere above.

## 8. Annotation-tool instructions

```bash
pip install -r requirements.txt
uvicorn app.annotation_tool.main:app --reload --port 8010
# open http://localhost:8010/
```

Requires, on the SAME machine: the generated `manifest.json` + `classes.json`
+ `crops/` (committed, pseudonymous) under `data/dev_plans_v04/annotation_dataset/`,
and — only for the optional "larger context" view — the raw PDFs under
`data/dev_plans_v04/raw/` (never committed; re-supply locally from the
original ZIP if needed, matched by pseudonym via your own local copy of
`id_mapping.LOCAL_ONLY.json`, which this pipeline writes but never commits).

For each candidate: a tight crop with a red outline marking exactly which
object is the candidate (added specifically because some crops are busy
multi-component views), an optional larger-context view rendered live from
the raw PDF, the plan/page, nearby text, graph relation, and the engine's
suggestions as one-click buttons. Choose a class, mark
`NOT_A_COMPONENT`/`AMBIGUOUS`, skip, or go previous/next (buttons or
arrow keys; `S` skips). Every action saves immediately to `manifest.json`
(atomic write, see `store.py`) — closing the browser loses nothing.
Progress (`annotated / total`) is shown in the header at all times.

## 9. Expected annotation workload

837 total candidates; 690 carry at least one engine suggestion (a
one-click accept/correct), 147 carry none (`qc.py`'s
`review_queue_no_suggestion`, ~18% of the batch) and need the reviewer's
own judgment from the crop alone. At roughly 3-5s per one-click
suggestion-review and 8-12s per no-suggestion candidate (skip/classify/mark
NOT_A_COMPONENT), a full first pass over all 837 candidates is roughly
**1-1.5 hours** of focused review time — one sitting, not a multi-day
effort, consistent with the closing goal of clicking through suggestions
rather than personally re-reviewing all 20 full plans.

## 10. Is this dataset sufficient for a first trained detector?

**No — not yet, and not by design at this phase.** Three concrete reasons,
same standard this project has applied at every prior phase (v0.3's own
POC verdict was "FAILED — TRAINED DETECTOR DATASET REQUIRED" for the same
reason: not overclaiming what hasn't been verified):

1. **Nothing is human-verified.** Every one of the 837 candidates starts
   `UNLABELED`; `export.py::export_yolo_labels` will (correctly, by
   construction) emit zero training labels until a human sets
   `assigned_class` via the annotation tool. An engine suggestion is
   provenance/context, never training truth.
2. **Thin, uneven support.** `MIN_SUPPORT=2` is a real floor, not a
   comfortable one — several proposed classes have only 6-8 measured
   occurrences across all 20 plans (§5), and `qc.py`'s own class-imbalance
   check on the engine's raw suggestions already reports a **147:1**
   imbalance ratio between the most- and least-common suggested label
   (`imbalance_warning: true`), before any human labeling has even
   started.
3. **The taxonomy itself needs a human pass first** (§5's flagged rows) —
   training against "apparate stk. kw ww dim" as if it were a real device
   class would bake a table-header artifact into a detector.

This phase's job was to make a first annotation pass FAST and safe, not to
produce training-ready labels — that is Phase 6 (the annotation tool) and
beyond, explicitly out of this task's scope ("do NOT train a detector").

## Quality control

Duplicate/near-duplicate candidates (bbox IoU ≥ 0.5 on the same plan/page —
exactly the "candidate generation overlaps" case named in the task),
class-imbalance signal over the engine's own suggested labels (not yet
human labels), and a review-priority queue of candidates with no engine
suggestion at all:

- **Duplicate/near-duplicate candidates**: 0 pairs found (bbox IoU ≥ 0.5 on
  the same plan/page) — v0.1's own symbol clustering did not produce
  overlapping candidates on this batch.
- **Class-imbalance signal** (over the engine's own SUGGESTED labels, not
  yet human labels — descriptive only): `imbalance_warning: true`, ratio
  **147.0** (the largest suggested-label bucket, "NO_SUGGESTION" itself at
  147 candidates, versus several 1-occurrence tail labels). Consistent with
  §5/§10's point that raw suggestion frequency is noisy and needs human
  curation.
- **Review-priority queue**: 147 candidates (~18% of the batch) carry no
  engine suggestion at all and should be reviewed first, since the engine
  has nothing to offer there — exactly the set §9's workload estimate gives
  extra time per candidate.

## Performance

**20/20 plans succeeded** (100%), zero failures, zero timeouts against the
final 700s per-plan budget. Runtime: **min 8.7s, max 513.6s
(DEV-07), mean 86.6s**, total 1731.5s (~29 minutes) sequential wall-clock
for the whole batch, no GPU. Peak RSS per plan ranged from **65MB (DEV-09) to 619MB (DEV-07)** — the
<512MiB target was exceeded on exactly the two largest plans (DEV-07 619MB,
DEV-10 573MB; both 300k+ raw drawing primitives), "where practical" per the
task's own wording; every other plan stayed well under it, the next-highest
being DEV-01 at 413MB. 845→837 crop PNGs written (the 8-candidate difference between the
pre-fix and final run is the `privacy.py` case-sensitivity fix correctly
excluding a few more PII-overlapping candidates, not a regression).

## Limitations

- Only 1 of this batch's 20 real plans (DEV-18) is genuinely scanned/
  image-based with near-zero vector content; it and 4 further, richly
  vector-based plans (DEV-02, DEV-09, DEV-16, DEV-20) needed v0.1's
  existing tiled-OCR fallback (`text.py`, unchanged) for some or all of
  their text (`text_source` "ocr"/"mixed" — see §1's table). One OCR call
  triggered a genuine tesseract hang (multiple minutes on a single tile)
  during development — not a v0.4 regression, a pre-existing OCR-path
  robustness gap this batch's real, varied input surfaced for the first
  time. Worked around at the orchestration level
  (`scripts/run_v04_pipeline.py` runs every plan in its own process GROUP
  with a hard wall-clock timeout and kills the whole group, including any
  runaway tesseract, on timeout) rather than by touching v0.1's text.py.
- **Privacy required three separate rounds of fixes to reach its current
  state, across three genuinely different leak vectors** (crop-image text
  layer, crop-image OCR-invisible vector-outlined text, and legend-derived
  metadata TEXT reaching committed JSON independent of any crop) — see §3
  for the full account. All three are fixed and re-verified (a final
  automated scan of every string in the three committed JSON files found
  zero genuine matches; 5 real crops, including 3 from the specific plan
  the first leak was found on, were also manually inspected). The pattern
  across all three — geometric/structural assumptions about where PII can
  hide turning out to be false on real, messier plans than W-003 — is
  worth taking as a standing caveat for any FUTURE real-plan batch this
  pipeline processes, not just this one: re-run the same "does a
  privacy-relevant string reach a committed file" style audit this report
  did by hand (§3), don't assume the existing three-plus-two-more-layer
  net generalizes perfectly to plans with different title-block/legend
  conventions.
- The proposed taxonomy (§5) still surfaces some non-device rows (table
  headers, installation notes, a tenant name) that only a human can
  reliably tell apart from real component classes — flagged explicitly in
  §5 rather than silently shipped as if solved.
- The symbol-zone lookback and context-crop sizing are tuned against this
  batch's and W-003's real layouts; a plan with a very different legend or
  drawing layout may need re-tuning.
- No scale calibration, same as v0.1/v0.2/v0.3.
- All 20 real plans in this batch happen to be single-page, so this wasn't
  exercised, but is worth flagging for the next batch: `_v04_plan_worker.py`'s
  own per-plan audit fields (`graph_nodes`, `graph_edges`,
  `symbol_candidates`, `text_source`, per-page warnings) are read from
  `doc.pages[0]` only, even though `page_count` itself correctly reflects
  the whole PDF. `component_facts.py`, `legend_intelligence.py`, and
  `candidates.py` already iterate every page correctly (`for page_model in
  doc.pages`) — candidate generation and crops would be complete and
  correct for a multi-page plan today — but the audit *summary* numbers in
  `pipeline_summary.json`/this report would under-report a multi-page
  plan's graph/symbol counts to its first page alone. Fix before adding a
  multi-page plan to this pipeline.
- This is proposal/tooling only — no detector has been trained, no
  threshold was tuned to any specific plan, and W-001..W-010 were not read.

## Verdict

20/20 plans processed successfully; 837 privacy-checked annotation
candidates with crops, suggestions, and graph context; a working local
annotation tool; a family-respecting train/validation/test split; and a
15-class taxonomy proposal with its known, documented gaps (§5, §10) for a
human to curate during the first review pass. Nothing here trains a
detector, tunes a threshold to any specific plan, or touches W-001..W-010.

`DATASET READY FOR HUMAN ANNOTATION`
