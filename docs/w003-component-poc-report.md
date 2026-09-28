# W-003 Component-Recognition PoC Report (v0.2)

## Scope recap

Extends v0.1 (deterministic topology, unchanged and re-verified below) with
component recognition: *what component is located at or near this proven
graph position?* Evaluated on the real, committed W-003 fixture only.
W-001…W-010 remain untouched holdout data — neither the symbol-library data
audit nor the recognition thresholds below were built or tuned against them.

## Data audit summary

Full detail in `docs/symbol-audit.md`; headline numbers: of the 58 SVGW
reference symbols, **43 are eligible for shape-based matching**; 5
(SYM-001…005) are pipe/line notation already covered by v0.1's topology and
text layer, not physical components; 7 ambiguous families and 12
low-distinctiveness symbols were identified and excluded or capped below
`COMPONENT_FACT`; 9 require a text/legend code before any specific type
claim is a fact. **A critical, audit-driven fact**: the 58 reference images
are themselves scanned raster legend graphics, not vector paths — this
forced a raster/classical-CV recognition strategy on principle, not
convenience.

## v0.1 regression check (must stay unchanged)

`pytest tests/test_reproducibility.py::test_w003_ten_repeated_runs_produce_semantically_identical_plan_facts`
— **still 10/10 identical**, same 1,468-fact count, same canonical hash
behavior as the v0.1 report. v0.1's code (`plan_facts.py`, `pipeline.py`,
and all ten reused parser modules) was not modified in v0.2.

## Component recognition results on real W-003

| Metric | Value |
|---|---|
| Symbol candidates evaluated (v0.1's own `detect_symbols()` output) | 76 |
| `COMPONENT_FACT` | 0 |
| `COMPONENT_CANDIDATE` | 0 |
| `UNRESOLVED` | 76 |
| Highest confidence score reached, any candidate | 0.581 (below the 0.75 `COMPONENT_CANDIDATE` floor) |

**Every one of the 76 real symbol candidates on W-003 came back
`UNRESOLVED`.** This is reported exactly as measured — the precision-first
thresholds (`FACT_SCORE_MIN=0.90` + `FACT_MARGIN_MIN=0.12`,
`CANDIDATE_SCORE_MIN=0.75`) were not lowered to manufacture recall.

### This is a real finding, not a bug — how it was verified

Before accepting "zero recognitions" as a real result, the matching
*pipeline itself* was validated against clean, controlled inputs
(`tests/test_symbol_library.py`, all passing):
- An exact copy of a template scores > 0.99 against itself.
- A 90°/180°/270°-rotated copy still scores > 0.95 for the correct symbol.
- A 0.6×–1.6× rescaled copy still scores > 0.85 for the correct symbol.

So the matcher correctly recognizes anything drawn in a style consistent
with the reference legend. Visually inspecting real W-003 crops
(`scratch_audit/w003_crops_sheet.png`, not shipped — one-off diagnostic)
showed the actual failure mode: many of v0.1's 76 "symbol" clusters are
**not SVGW plumbing icons at all** (drafting tick marks, numbered
revision-callout bubbles, a prohibition/no-smoking-style icon) — correctly
`UNRESOLVED` — and the ones that plausibly *are* plumbing components (e.g.
a mixing-valve assembly) are drawn in a denser, more detailed CAD style
than the clean scanned legend reference, with `detect_symbols()`'s
proximity clustering sometimes merging several nearby annotations into one
oversized bounding box rather than one clean icon. **The gap is a genuine
style/data mismatch between the reference legend and this specific plan's
drawing convention, not a defect in the matching code.**

### Graph association (always derived, never invented)

Even though no component reached `COMPONENT_FACT`, every one of the 76
candidates got an honest graph-association classification purely from
v0.1's own already-proven evidence (symbol ports / edge proximity, see
`component_facts.py::_graph_association`, tested directly in
`tests/test_component_facts.py`) — never invented from proximity alone:

| Relation | Count |
|---|---|
| `at_branch` (≥2 proven ports converge here) | 34 |
| `on_edge` (no proven port, but geometrically on a drawn pipe run) | 23 |
| `at_endpoint` (exactly 1 proven dangling-end port) | 17 |
| `near_not_connected` (neither holds — honestly reported, not forced) | 2 |

### False/duplicate detections vs. available ground truth

No independently-labeled ground truth for W-003's component types exists in
this environment (per the task's own instruction, W-001…W-010 are
holdout-only, and no separate answer key was supplied) — so a
precision/recall number against real labels cannot be honestly computed
here. What *was* verified: (a) the duplicate-suppression logic collapses an
injected duplicate symbol entry into exactly one component fact
(`test_w003_duplicate_symbol_entry_is_never_reported_twice`), and (b) the
real 76-candidate run produced 76 distinct `component_fact_id`s with no
natural duplicates.

## Legend investigation (light, per the task's explicit boundary)

W-003 **does** contain a plan-specific legend block (`"LEGENDE SANITÄR"`,
with a `"Dämmungslegende Sanitäranlagen"` sub-heading), found by the same
kind of text-vocabulary search used in v0.1's garden-valve probe. Its ~15
entries are mostly pipe-*type* labels (`Kaltwasser Netzdruck`,
`Kaltwasser red. Druck`, `Zirkulation (RAR)`, `Warmwasser`,
`Schmutzabwasser`, `Regenabwasser`) that correspond to *line color/style*
conventions already handled by `label_hints.py`'s pipe-code patterns, plus a
few entries that do name specific SVGW components (`Wasserzähler`,
`Druckreduzierventil`, `Fliessrichtung`). **Feasibility verdict**: a
plan-specific legend parser is plausible — it would need to associate each
legend text label with a small drawn swatch/icon to its own left (extending
`association.py`'s existing text-to-geometry matching to a new target,
legend rows) and fuzzy-match the label text against the 58-item catalog's
`name` field — but per the task's explicit instruction, **it was not built
in this POC**; this section is the required light investigation only.

## Performance (real W-003, `scripts/measure_performance.py`, 5 runs)

| Metric | Value |
|---|---|
| Parse (`pipeline.analyze_pdf_bytes`, v0.1, unchanged) | avg 1.23s |
| PlanFacts (`plan_facts.build_document_facts`, v0.1, unchanged) | avg 0.018s |
| **Component recognition (`component_facts.build_component_facts`, new)** | avg 6.88s |
| **Total** | avg 8.13s |
| **Peak process RSS (cumulative, 5 runs, one process)** | **~227 MiB** |

Comfortably below the 512 MiB target. No GPU, no neural network. The ~6.9s
component-recognition cost is 76 crops × 43 eligible templates × 4
rotations (~13,000 `cv2.matchTemplate` calls on 128×128 arrays) plus 76
PyMuPDF pixmap renders at 300 DPI — real, but a synchronous cost most
"modest cloud service" deployments can absorb; not optimized further in
this POC (candidate future work: lower render DPI, batch/vectorize the
template comparisons, cache rotated templates across requests within a
worker process rather than only within one).

## 10-run reproducibility (both v0.1 and v0.2 facts)

`pytest tests/test_reproducibility.py -v` — **10/10 identical** for both:
- v0.1's `plan_facts` (unchanged from the v0.1 report).
- v0.2's `component_facts`: identical canonical hash and identical
  `component_fact_id` sets across all 10 runs (all 76 `UNRESOLVED`, every
  run, byte-for-byte).

## Tests

32/32 pass (`pytest tests/ -v`): the 16 from v0.1 (all still green,
confirming zero regression) plus 16 new — 6 on `symbol_library.py` (exact,
rotated ×3, scaled ×2, ambiguous-family self-consistency, excluded-symbols
guard, near-blank crop), 9 on `component_facts.py` (endpoint/branch/on-edge/
near-not-connected graph association, ambiguous-recognition-stays-
unresolved, unrecognizable-crop-is-unresolved, a genuine clean self-match
reaching `COMPONENT_FACT`, duplicate suppression, graph-order independence
— all on real W-003 data where a real PDF was needed), and 1 reproducibility
test.

## Limitations

- **Zero components reached `COMPONENT_FACT` or `COMPONENT_CANDIDATE` on
  the one real plan evaluated.** The recognition pipeline is verified
  correct on clean inputs; the gap is the reference-vs.-plan style mismatch
  described above, which a single-plan POC cannot rule out as
  plan-specific. More real reference plans, ideally spanning several
  drafting styles/offices, are needed to know whether this generalizes.
- `detect_symbols()`'s clustering (unchanged, v0.1) sometimes merges several
  nearby annotations into one oversized symbol candidate on dense areas of
  a plan — this directly hurts single-icon template matching and was not
  addressed here (would require touching v0.1's `symbols.py`, out of scope).
- No ground-truth component labels for W-003 exist in this environment, so
  precision/recall against real answers could not be computed, only
  measured against the engine's own internal consistency.
- Text/legend corroboration is investigated but not implemented as a full
  pipeline stage (per the task's explicit boundary).

## Is the classical-CV approach sufficient, or is a learned detector needed?

**Not yet decidable from one plan.** The matching *algorithm* is
demonstrably correct (100% recognition on exact/rotated/scaled reference
copies). The *reference data* (58 clean scanned legend icons) may simply be
too far, stylistically, from how symbols are actually hand/CAD-drawn across
real engineering offices — the honest next step is evaluating against
several more real plans (still never W-001…W-010) before concluding
classical methods are insufficient and reaching for a learned detector, per
the task's own instruction not to add one "merely because it is possible."

## Verdict

**POC PARTIAL — COMPONENT DATASET MUST BE EXPANDED**
