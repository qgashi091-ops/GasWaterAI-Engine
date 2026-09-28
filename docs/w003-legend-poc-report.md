# v0.3 report — plan-specific legend intelligence

## 1. Legend detection result

`legend_detection.py::detect_legend_candidates()` runs on W-003 with **no
hardcoded coordinates and no hardcoded filename** — it only uses: (a) a
spatial density test reusing v0.1's own `geometry.SpatialBBoxClusterer` on
text-row bounding boxes (rows only merge when actually close in both X and
Y, exactly like v0.1 clusters nearby vector primitives into a symbol — never
"same height anywhere on the page", which was an early, rejected design
that bridged unrelated content into a nonsense near-full-page-width block),
and (b) a nearby "Legende/Symbole/Zeichenerklärung" heading match as
corroborating, not required, evidence. Every text-span bbox is transformed
through the page's own `rotation_matrix` before any clustering — W-003 is
rotated 90°, and skipping this step was the first, diagnosed-and-fixed bug
during development (see §5).

Result on the real W-003 PDF: **4 candidates**, ranked by confidence,
ambiguity preserved (all 4 returned, not just the top one):

| legend_id | confidence | heading | rows | bbox (display space) |
|---|---|---|---|---|
| LG1_4 | **1.00** | `LEGENDE SANITÄR` | 40 | (42.5, 268.5)–(524.5, 502.3) |
| LG1_2 | 0.60 | *(none)* | 44 | (34.1, 535.7)–(637.8, 832.0) — "Dämmungslegende" text/notes block |
| LG1_1 | 0.55 | *(none)* | 15 | (34.4, 983.8)–(478.3, 1094.7) — "Legende Verteilbatterie" |
| LG1_3 | 0.43 | *(none)* | 11 | (36.7, 47.6)–(582.8, 235.4) — title block/letterhead |

The real, usable symbol legend (`LEGENDE SANITÄR`) is correctly identified
as the top-confidence candidate, with a bbox matching its true visual
extent (verified against a rendered crop of the page). The two "lookalike"
blocks the task explicitly anticipated — a bordered/coded lookup table
("Legende Verteilbatterie") and a pure text/notes block ("Dämmungslegende")
— are both detected as dense text blocks (correctly, since geometrically
they are) but ranked well below the heading-matched real legend by
construction (confidence bands `[0, 0.6)` for headingless candidates and
`[0.6, 1.0]` for heading-matched ones cannot overlap). Verified 10/10
identical across repeated runs (`tests/test_legend_detection.py`).

## 2. Number and quality of extracted entries

`legend_entries.py::classify_candidate_rows()` applied to all 4 candidates
yields **110 total legend entries**. For the real symbol legend (LG1_4,
40 rows):

| classification | count | notes |
|---|---|---|
| `USABLE_TEMPLATE` (real icon) | 29 | Ventil, Wasserzähler, Pumpe, Manometer, Thermometer, Filter, Sicherheitsventil, Rückschlagklappe, Fliessrichtung, ... |
| `USABLE_TEMPLATE` (line-style swatch, excluded from search) | 8 | Kaltwasser/Warmwasser/Zirkulation/Enthärtet-N°fH pipe-type dash/dot samples |
| `TEXT_ONLY` | 1 | empty/no-ink row |
| `REJECTED` | 2 | single-letter "M"/"T" marker rows whose lookback zone picked up a neighboring icon's ink (border-shaped) |

For the two lookalikes: `LG1_1` ("Legende Verteilbatterie") is **rejected
as a whole** — its rows match a `<code> = <description>` pattern (`a =
Schrägsitzventil 5/4"`, `b = Wasserzähler (Lieferung EWL)`, ...) at a 100%
ratio, structurally identifying it as a coded lookup table with no symbol
column, exactly as the task anticipated it as the intended trap case.
`LG1_2` ("Dämmungslegende") is likewise rejected as a whole: its usable-icon
ratio (ink found next to a real label) is far below the 15% floor — it is
almost entirely prose about insulation material choices.

Quality, visually verified (§4): every inspected `USABLE_TEMPLATE` crop is
a clean, correctly-isolated icon or line-style sample with no label text or
table border bleeding in — see §5 for the two contamination failure modes
that were found and fixed during development before this state was
reached.

## 3. v0.2 vs. v0.3 component-recognition comparison

| | v0.2 (generic 58-symbol library) | v0.3 (plan-specific legend templates, hybrid) |
|---|---|---|
| Symbol candidates evaluated | 76 (every v0.1 candidate, including ones inside the legend) | 68 (legend-interior candidates correctly excluded — see §"legend self-match exclusion" in §7) |
| `COMPONENT_FACT` | 0 | 0 |
| `COMPONENT_CANDIDATE` | 0 | 0 |
| `UNRESOLVED` | 76 | 68 |
| Best real match score | 0.581 (below 0.75 candidate floor) | 0.591 (`ventil`, still below 0.75) |
| Duplicates | 0 (de-duplicated by bbox key) | 0 (same de-duplication, reused) |

The plan-specific legend hypothesis produced templates in the plan's own
CAD style (closing the exact style gap v0.2 diagnosed), and the best real
score did move — marginally — from 0.581 to 0.591. That is not a
meaningful improvement, and it does **not** clear `CANDIDATE_SCORE_MIN`
(0.75), so v0.3's own precision-first gate correctly reports it as
`UNRESOLVED`, identically to v0.2. See §5 for why (root-cause finding:
that top match's own source crop is a near-empty v0.1 `SymbolCandidate`,
not a real icon silhouette).

## 4. Representative verified matches

Three real crops were rendered from the PDF and visually inspected (not
assumed correct from a score alone):

- **`Ventil` legend template** vs. its real drawing counterpart: the legend
  crop is a clean bowtie-valve icon on a line; visual inspection confirmed
  correct isolation (no label text, no table border).
- **`Sicherheitsventil` legend template**: a triangle+circle+line
  safety-valve icon, correctly isolated.
- **Top real drawing match ("ventil", score 0.591)**: rendering the exact
  matched bbox from the drawing shows an almost-blank crop — a single thin
  horizontal line fragment, not a valve icon. This is the direct evidence
  behind §5's root-cause finding: the v0.1 `SymbolCandidate` at that bbox is
  degenerate (near-zero visual content), so no template — plan-specific or
  generic — could have matched it correctly. The classification
  (`UNRESOLVED`) is the right outcome for this specific instance.

Per the task's explicit instruction, this candidate is reported honestly as
what it is (a bad `SymbolCandidate`, not a wrongly-classified match), not
silently treated as "the algorithm failed here."

## 5. False-positive / ambiguity analysis

**A real false positive was found and fixed during development.** Before
the line-style-swatch detector was strengthened, one v0.1 drawing candidate
matched the `Schmutzabwasser`/`Regenabwasser` legend templates (dash-dot
pipe-type line samples) at score 0.75 — exactly clearing
`CANDIDATE_SCORE_MIN` and producing a spurious `COMPONENT_CANDIDATE`.
Rendering both the template and the matched drawing crop showed the
template was a dashed line sample, and the "match" was an unrelated straight
pipe segment elsewhere in the drawing that happens to look similar as a
raw shape. This is precisely the risk this task's design reasoning
anticipated ("a line-style swatch would be dangerous... would match nearly
every drawn pipe of that type"). Fix: `legend_entries.py`'s swatch detector
was extended from a simple "ink height ≤ 4px" rule (which caught 8 of 10
line-style entries) to also flag entries whose ink forms ≥6 small,
disconnected islands along a thin band (dash/dot patterns whose marks poke
slightly taller than a solid line) — this caught the remaining 2
(`Schmutzabwasser`, `Regenabwasser`) without over-triggering on genuine
multi-stroke icons (e.g. `Filter`, which also has several strokes but a
much less extreme height/width ratio). After the fix, the spurious
candidate disappeared and the templates-built count dropped from 31 to 29,
exactly the two swatches. **Two false-positive-prone rows were also
rejected within the real legend itself** (the single-letter "M"/"T" marker
rows, §2) — their lookback zones picked up border-shaped ink from a
neighboring row, and the border-shape heuristic correctly rejected them.

**Ambiguity preservation**: all 4 legend candidates are returned, not just
the best guess (§1); every legend entry's classification is explicit, never
a silent drop (§2); `hybrid_conflict` is recorded (never hidden) whenever
the plan-specific and generic recognition paths would disagree on identity
— though on the real W-003 run, 0 conflicts occurred, since no candidate
cleared the plan-specific threshold at all.

## 6. 10-run reproducibility

Verified for every new stage, on the real, committed W-003 fixture:

- Legend detection: 10/10 identical (`tests/test_legend_detection.py::test_w003_legend_detection_is_reproducible_across_ten_runs`).
- Full legend-intelligence pipeline (legend entries, templates, hybrid
  component facts, stable IDs): 10/10 bit-identical canonical-JSON hash
  (`tests/test_legend_intelligence.py::test_w003_legend_intelligence_is_reproducible_across_ten_runs`).
- v0.1 topology: unchanged and unaffected by running v0.3 afterward,
  verified by an explicit before/after hash comparison
  (`test_v01_topology_is_unaffected_by_running_v03_afterward`), in addition
  to v0.1's own pre-existing 10-run reproducibility test still passing
  unmodified.

## 7. Runtime / RSS

Measured with `scripts/measure_performance.py` (5 runs, real W-003 PDF):

| stage | min | max | avg |
|---|---|---|---|
| Parse (`pipeline.analyze_pdf_bytes`) | 1.18s | 1.70s | 1.33s |
| PlanFacts (v0.1) | 0.016s | 0.023s | 0.018s |
| Component recognition (v0.2) | 6.61s | 6.80s | 6.69s |
| **Legend intelligence (v0.3)** | **10.94s** | **11.37s** | **11.11s** |
| Total | 18.75s | 19.71s | 19.15s |

**Peak RSS across 5 cumulative runs: 227.6 MiB** — well under the 512 MiB
hard requirement, with substantial headroom (< 45% of the budget), matching
v0.2's own measured footprint (raster rendering at 300 DPI dominates memory
use in both, and v0.3 reuses the exact same rendering path).

Legend self-match exclusion (Phase 4 requirement) reduces v0.3's own
evaluated-candidate count below v0.2's (68 vs. 76, §3) — the 8 fewer
candidates are exactly the ones whose bbox falls inside a detected legend
region.

## 8. Dataset-export readiness

Every extracted `LegendEntry` already carries what a future training
pipeline would need: a normalized crop (`symbol_canonical`, a 128×128
ink-mask, scale/rotation-ready), a label (`normalized_label`), a bbox, and
a `legend_id`/`page` (from which a plan/document identifier is trivially
derivable). What is **not** built (correctly, per the task's explicit
boundary): no automatic promotion of an unverified match into "training
truth" — every entry's `classification` and every hybrid component fact's
`kind` stay exactly what the evidence supports (`USABLE_TEMPLATE` is
document-local matching evidence, not a verified label; a real training
export would need an explicit human verification step per example, not
implemented here). No training code, no YOLO/PyTorch dependency, exists
anywhere in this version.

## 9. Limitations

See `docs/architecture.md`'s "Known v0.3 limitations" section for the full
list; in summary:
- Legend detection/extraction (Phases 1–3) work very well on W-003, but
  Phase 4 (searching the drawing) found zero components above threshold —
  and the one near-threshold match's own source crop was degenerate,
  pointing at v0.1's symbol-candidate clustering quality as at least as
  large a limiting factor as any remaining style gap.
- The symbol-zone lookback (fixed 58pt window left of a row's label) is
  tuned to W-003's own legend layout; a plan with a differently-laid-out
  legend would need a different or adaptive rule.
- The coded-lookup-table and line-style-swatch detectors are real,
  validated heuristics but were only ever tuned against this one plan.
- No scale calibration, same as v0.1/v0.2.

## 10. Verdict

**POC FAILED — TRAINED DETECTOR DATASET REQUIRED.**

Rationale: the core hypothesis under test — that a plan's own legend,
rendered in the plan's own CAD style, would close the domain/style gap that
kept every v0.2 generic-library match below threshold — was implemented
correctly and evaluated honestly (Phases 1–3 succeeded excellently; Phase 4
found the templates but the drawing-side matches still don't clear the
precision-first bar). The measured outcome is that plan-specific legend
matching, on its own, **does not demonstrably close the recognition gap**
on the one real plan available: the best real score barely moved (0.581 →
0.591) and the one near-threshold candidate was traced to a degenerate
underlying `SymbolCandidate`, not a genuine near-miss shape comparison.
That is exactly the condition this task named for this verdict. The
success-gate criterion requiring "at least some real drawing components
classified with stronger evidence than v0.2" is not met, and — per this
task's explicit instruction — thresholds were never lowered to manufacture
a passing result; doing so would have produced exactly the false-positive
this report instead caught and fixed (§5).

This is not a wasted result: Phases 1–3 (legend detection, entry
extraction, plan-specific template construction) are solid, reusable,
reproducible infrastructure — and, per §8, already shaped to become the
labelled-example source for a trained detector, which this verdict
identifies as the right next investment rather than further classical-CV
threshold tuning.
