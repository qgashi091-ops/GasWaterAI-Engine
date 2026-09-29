# Architecture

```
PDF bytes
  -> pipeline.analyze_pdf_bytes()          (reused, unchanged)
       -> vectors.classify_page_vectors()   (reused, unchanged)
       -> graph.build_pipe_graph()          (reused, unchanged)
       -> symbols.detect_symbols()          (reused, unchanged)
       -> bridging.bridge_graph()           (reused, unchanged)
       -> text.extract_native_text_spans() / ocr_page()  (reused, unchanged)
       -> label_hints.guess_label_hint()    (reused, unchanged)
       -> association.associate_text_spans()(reused, unchanged)
     -> schema.DocumentAnalysis             (reused, unchanged — the JSON contract)
  -> plan_facts.build_document_facts()      (v0.1 — topology, unchanged in v0.2)
  -> component_facts.build_component_facts() (v0.2 NEW — component recognition)
       -> symbol_library.SymbolLibrary       (v0.2 NEW — the 58-symbol reference set)
  -> app/main.py: POST /analyze             (thin FastAPI wrapper, extended additively for v0.2)
```

## What was reused, and why unchanged

The existing `GasWaterAI` Python parser repository already does deterministic
PDF→graph extraction well: it classifies vector drawings into pipe segments /
text glyphs / symbol candidates, snaps segment endpoints into a graph within
a page-relative tolerance, conservatively bridges a small set of genuine
junction gaps (three independent, gated mechanisms — never a blanket
tolerance increase), extracts text natively or via tiled OCR, and associates
text to the nearest symbol or edge. None of this needed to change for this
PoC, and rewriting working, already-tested geometry/graph code would have
been pure risk for no benefit. Copied verbatim into `app/plan_analysis/`:

| Module | Role |
|---|---|
| `geometry.py` | distance/bbox helpers, `UnionFind`, `SpatialBBoxClusterer`, `PointSnapper` |
| `vectors.py` | classifies raw vector drawings into line/glyph/symbol primitives; background-layer detection |
| `graph.py` | snaps segment endpoints into a `BuiltGraph` (nodes + edges) |
| `symbols.py` | clusters leftover primitives into symbol candidates; matches dangling ends to symbol "ports" |
| `bridging.py` | three conservative, gated mechanisms (collinear gap / symbol junction / corner convergence) that reconstruct genuine junction gaps, always tagged `is_bridge=True` |
| `text.py` | native text extraction with a tiled-OCR fallback for vectorized/garbled/scanned pages |
| `association.py` | nearest-symbol-then-nearest-edge text association, grid-indexed |
| `label_hints.py` | non-authoritative SVGW W3 Anhang 4 pattern matching (pipe codes, DN sizes, backflow-device codes) |
| `schema.py` | the Pydantic JSON contract (`DocumentAnalysis`/`PageAnalysis`/`GraphNode`/`GraphEdge`/...) |
| `pipeline.py` | orchestrates the above into one `DocumentAnalysis` |

**Zero lines of these ten modules were changed.** This is the strongest
possible form of "reuse where appropriate" — no adaptation was needed because
the existing contract (`schema.PageAnalysis`) already exposes everything the
new topology layer needs: the graph, which edges are bridges and why, symbol
ports, and text-to-edge associations.

## What is new: `plan_facts.py`

The original repository's `pipeline.py` stops at graph/symbol/text
extraction — it has never computed a cycle, a dead-end, or any other
topological classification (confirmed by direct inspection of the module: no
cycle-detection or connected-component-classification code exists anywhere
in the original repo). `plan_facts.py` is new work that answers exactly this
gap, and only this gap — no scale calibration, no professional rules.

### The FACT / DERIVED_FACT / UNRESOLVED ontology

- **FACT** — base geometric or textual evidence that exists independent of
  interpretation: a directly-drawn pipe segment exists; a piece of text was
  extracted and, through direct association (`association.py`), sits on a
  specific edge.
- **DERIVED_FACT** — a claim computed *from* FACTs through pure graph
  algorithms: node degree, connected-component membership, branch
  classification, a proven terminal endpoint, cycle membership, a confirmed
  dead-end path. Deterministic and reproducible, but one step removed from
  raw geometry — exactly the "two proven segments share a proven node"
  category from the prior Base44 architecture review.
- **UNRESOLVED** — a first-class, terminal answer whenever a bridge
  (reconstructed) edge or an unclassifiable endpoint prevents proof. Never a
  fallback error, never a guess.

A bridge edge is always recorded (as a `FACT`-tier `pipe_segment` entry with
`kind="UNRESOLVED"`, `evidence_status="reconstructed"`) but can never
contribute to a `cycle_membership` or `dead_end_path` DERIVED_FACT — see the
module docstring in `plan_facts.py` for the full semantics, ported directly
from the equivalent (and independently validated) TypeScript module in the
Base44 application's PlanFacts Phase 1 work.

### Algorithm

Per page, restricted to the directly-drawn (non-bridge) subgraph:

1. Recompute node degree from scratch, counting only direct edges — the
   original pipeline's own `GraphNode.degree` field **includes** bridge
   edges after `bridging.bridge_graph()` runs (see `graph.py`/`bridging.py`:
   a bridged node's degree is explicitly incremented), so trusting it here
   would let a reconstructed connection manufacture a branch or hide one.
2. Union-Find connected components over direct edges only.
3. Per component: a deterministic BFS spanning tree (root = lexicographically
   smallest node id; neighbors visited in sorted edge-id order) — any direct
   edge not in the tree is a back edge that closes exactly one fundamental
   cycle (its tree path to the least-common-ancestor, plus itself). This
   correctly marks every edge/node that participates in *any* cycle, not
   just one arbitrarily chosen loop.
4. Dead-end/stub paths: walk outward from every proven leaf (degree 1, not a
   cycle member) through non-cycle edges until reaching a branch or the edge
   of a cycle.
5. Terminal classification (`PHYSICAL_END` / `DEVICE_BOUNDARY` /
   `DISTRIBUTOR_BOUNDARY` / `UNRESOLVED`) reuses the *existing* symbol-port
   and distributor-vocabulary evidence already in `schema.PageAnalysis` — it
   is the same design as the Base44 application's `vektorGraphMapper.ts`
   `classifyEndpoints()`, reimplemented natively in Python (not imported: this
   is a standalone repository with no dependency on the TypeScript
   application) with one deliberate strengthening: **any** bridge touching a
   degree-1 node forces `UNRESOLVED`, regardless of what the parser's own
   `degree` field or `dangling_pipe_ends` list would otherwise suggest.

O(V+E) per page; no raster processing, no ML model, no new dependency beyond
what the original parser already required.

## v0.2: component recognition (`symbol_library.py`, `component_facts.py`)

New modules, built strictly on top of v0.1's unchanged output. Full data
audit in `docs/symbol-audit.md`; full evaluation in
`docs/w003-component-poc-report.md`. Summary:

- **`symbol_library.py`** loads the 58 SVGW reference images (committed at
  `app/data/symbol_library/`, extracted once from the source PDF — itself a
  scanned raster legend, not vector data, which is why matching is
  raster/classical-CV rather than vector-to-vector) plus the offline data
  audit (`audit.json`), and exposes rotation-invariant (0/90/180/270°),
  scale-invariant (via ink-bbox-crop-and-resize canonicalization) normalized
  cross-correlation matching. The audit's `not_a_component` and
  `low_distinctiveness` symbols are excluded from the matching pool
  entirely; `ambiguous_families` and `text_legend_required` symbols can
  still be top matches but are capped below `COMPONENT_FACT`.
- **`component_facts.py`** re-renders a raster crop at each of v0.1's own
  already-detected `SymbolCandidate` bboxes (the ONLY new raster-processing
  step in this engine — v0.1 remains raster-free by design), matches it
  against the library, and classifies `COMPONENT_FACT` / `COMPONENT_CANDIDATE`
  / `UNRESOLVED` using precision-first thresholds plus the audit's
  ambiguous-family/text-required flags. Graph association
  (`at_endpoint`/`at_branch`/`on_edge`/`near_not_connected`) reuses v0.1's
  own proven `port_node_ids` and edge polylines — no new connectivity is
  invented from proximity alone.

**A real coordinate-frame bug was found and fixed while building this**,
entirely within v0.2's own code, without touching any v0.1 file: for a
rotated PDF page (`page.rotation != 0`, true for W-003 — it's rotated 90°),
PyMuPDF's `get_drawings()`/`get_text()` (which `vectors.py`/`graph.py`/
`symbols.py`/`text.py` all build every bbox from) report coordinates in the
page's *unrotated* frame, while `page.rect` and `get_pixmap()`'s `clip`
argument are in the *rotated/display* frame — meaning `schema.PageAnalysis`'s
own `width`/`height` (taken from `page.rect`) already silently disagreed
with the frame every bbox inside the same object is expressed in. This was
completely invisible in v0.1, whose topology algorithm never renders a
pixel or compares a bbox against `page.rect`. `component_facts.py` is the
first code to do so, and works around it locally by mapping each bbox
through `page.rotation_matrix` (identity when rotation is 0) before
building a render clip rect — see the comment in
`component_facts.py::_render_symbol_crop`.

## v0.3: plan-specific legend intelligence

New modules, built strictly on top of v0.1 and v0.2's unchanged output.
Full evaluation in `docs/w003-legend-poc-report.md`. Motivation: v0.2 found
that the 58 generic reference symbols (scanned from an SVGW W3 legend) are
drawn in a visibly different CAD style than W-003's own drawing, and that
gap kept every real match below the precision-first threshold. v0.3 tests
whether a plan's **own** legend can supply document-native templates
instead, closing that specific style gap.

```
legend_detection.py    Phase 1 -- detect_legend_candidates(): finds dense,
                        table/legend-shaped text-row clusters via v0.1's own
                        SpatialBBoxClusterer (reused on text-row bboxes
                        instead of vector primitives), corroborated by a
                        nearby "Legende/Symbole/Zeichenerklärung" heading
                        match. Ambiguity is preserved: every qualifying
                        block is returned, ranked by confidence, not just
                        the best guess.
legend_entries.py       Phase 2 -- classify_candidate_rows(): splits a
                        candidate into rows, looks for a symbol graphic
                        immediately left of each row's own label text
                        (the same rotation-aware raster rendering v0.2's
                        component_facts.py introduced), and classifies each
                        USABLE_TEMPLATE / TEXT_ONLY / AMBIGUOUS / REJECTED.
                        A whole candidate is rejected outright if it reads
                        as a coded lookup table ("a = ...", "b = ...") or
                        has too low a usable-icon ratio -- catching W-003's
                        own "Legende Verteilbatterie" (a lookalike, symbol-
                        less numeric table) and "Dämmungslegende" (a pure
                        text/notes block). A pipe-type line-style swatch
                        (e.g. "Warmwasser"'s dashed line sample) is kept as
                        USABLE_TEMPLATE evidence but flagged
                        `is_line_style_swatch` -- see symbol_templates.py.
symbol_templates.py     Phase 3 -- build_templates(): turns USABLE_TEMPLATE,
                        non-swatch entries into document-local match
                        templates, reusing symbol_library.canonicalize()
                        directly (same ink-crop/scale-normalize function the
                        58 generic templates use) and pre-rotating at
                        0/90/180/270° exactly like SymbolLibrary. Never
                        touches or overwrites the 58 generic templates.
legend_intelligence.py  Phases 4-6 + orchestration --
                        build_legend_intelligence(): searches every v0.1
                        symbol candidate OUTSIDE every detected legend
                        region (never a legend row against itself) against
                        the plan-specific templates, using the SAME
                        precision-first thresholds as v0.2's
                        component_facts.py (copied verbatim, not loosened).
                        Combines plan-specific + generic-library + graph
                        evidence: an unambiguous plan-specific match is
                        never overridden by a weaker generic one, but any
                        disagreement is recorded explicitly
                        (`hybrid_conflict`), never silently resolved.
```

A concrete false positive this discipline caught during development:
matching a pipe-type line-style swatch ("Schmutzabwasser"'s dash-dot
pattern) against an unrelated straight pipe segment elsewhere in the
drawing produced a spurious 0.75-confidence match. Rather than accept it,
`legend_entries.py`'s swatch detector was strengthened (ink-height *and*
connected-component-count checks) until it excluded that swatch from
becoming a search template, confirmed by visual inspection of both the
template and the match crop — see `docs/w003-legend-poc-report.md` for the
full account.

## API

`POST /analyze` (multipart PDF upload) → `{engine_version, document_fingerprint,
pages, plan_facts, component_facts, legend_intelligence, diagnostics}`.
`GET /health`. No database, no auth — out of scope per the PoC's hard
boundary. `component_facts` and `legend_intelligence` are purely additive:
every v0.1/v0.2 response field is unchanged, and a failure inside either is
caught in `app/main.py` and reported as an empty result rather than ever
taking down the (unrelated) topology or component-recognition response.

## v0.4: real-plan dataset & annotation pipeline

New, standalone modules (`app/dataset_pipeline/`, `app/annotation_tool/`) —
not part of the `POST /analyze` request path at all. Turns a batch of real,
unseen plans into a human-reviewable annotation dataset for a *future*
trained detector; trains nothing itself. Full results in
`docs/v04-dataset-report.md`.

```
app/dataset_pipeline/
  audit.py       Phase 1 -- runs the UNCHANGED v0.1/v0.2/v0.3 engine per
                  plan and reports page/vector/text/graph/legend stats plus
                  parser success/failure -- never silently skips a plan the
                  engine can't parse.
  privacy.py     Phase 2 -- regex-only PII heuristics over a plan's own
                  native text (never OCR/vision) and its original filename;
                  reports kind+count only, never the matched text itself.
  families.py    Project/style-family grouping (shared filename-stem
                  prefix, computed from the LOCAL, never-committed filename
                  mapping) + perceptual-hash near-duplicate detection --
                  the unit every later split decision respects.
  candidates.py  Phase 4 -- turns each v0.3 hybrid component fact into a
                  human-reviewable candidate: a context-bounded crop
                  (capped in absolute size, see MAX_CROP_SIDE_PT below),
                  nearby text, graph relation, and top suggestions. Every
                  candidate already comes from
                  legend_intelligence.build_legend_intelligence, which only
                  evaluates symbols OUTSIDE every detected legend/dense-text
                  region, so a title block is excluded for most candidates
                  "for free" -- but not reliably enough on its own; see
                  "Privacy: three stacked, independent layers" below for why
                  two more, mandatory checks sit in front of every crop this
                  module renders.
  taxonomy.py    Phase 5 -- proposes ~10-15 classes purely from this
                  batch's own measured evidence (legend-entry frequency,
                  v0.1's existing backflow-device-code pattern match
                  reused from label_hints.py, and generic-library
                  drawing-side suggestion frequency) -- never invents an
                  unsupported class.
  splits.py      Phase 7 -- train/validation/test assignment at the
                  FAMILY level (never a plan alone), greedy largest-family-
                  first. W-001..W-010 are never inputs to this module.
  export.py      Phase 8 -- manifest + class-list writer, and a YOLO-label
                  converter that only ever emits a label for a row a human
                  has actually verified (`verification_status == "LABELED"`)
                  -- an engine suggestion alone is never exported as
                  training truth.
  qc.py          Phase 9 -- duplicate/near-duplicate candidate detection
                  (bbox IoU on the same plan/page), class-imbalance
                  reporting, and a no-suggestion review queue.
app/annotation_tool/
  main.py + store.py + static/index.html
                  Phase 6 -- a small local FastAPI app + one static page
                  (no build step, no Base44, no network dependency). A
                  JSON-file store with atomic, autosaving writes. Draws a
                  highlight box over exactly which object in a (possibly
                  busy) crop is the candidate being reviewed, and offers an
                  on-demand, live-rendered wider-context view.
scripts/
  _v04_plan_worker.py   the actual per-plan engine run, executed as its
                         OWN subprocess.
  run_v04_pipeline.py   orchestrator: runs every plan in its own process
                         GROUP with a hard wall-clock timeout, killing the
                         whole group (not just the direct child) on
                         timeout -- see "Known v0.4 limitations" below for
                         why that specific detail mattered in practice.
```

**Crop sizing** (`candidates.py::MAX_CROP_SIDE_PT`): an early real crop from
this batch showed nearly an entire multi-storey schema, because the
proportional context margin was applied to a v0.1 `SymbolCandidate` that
was itself unusually large (the same pre-existing over-clustering
limitation already listed under "Known v0.2 limitations" below). Fixed by
capping the crop's absolute rendered size, centered on the candidate's own
centroid, rather than only bounding the margin.

**Privacy: three stacked, independent layers** (`candidates.py`, plus the
annotation tool's own live wide-context endpoint). This exists because a
real title block leaked into a rendered crop TWICE during development, each
time past a check that had looked sufficient:

1. `legend_intelligence.py`'s own geometric exclusion (Phase 4's self-match
   check: a v0.1 symbol candidate inside a detected legend/dense-text
   region is never turned into a component fact, so it never reaches
   `candidates.py` at all). Sufficient on W-003's title block; NOT
   sufficient in general -- one real plan's unusually large, fragmented
   title block left gaps between `legend_detection.py`'s clustered
   sub-regions, and a stray symbol candidate landed in exactly such a gap.
   Its rendered crop, caught by manual visual review before being
   committed, showed a property owner's name, a company's address, phone
   number and email.
2. `_crop_contains_pii()`: a second, content-based check -- scans every
   text span (v0.1's own `page_model.text_spans`) that overlaps the crop's
   OWN render region against `privacy.py`'s regex patterns (street
   address, postal-code+town, title-block field labels, company suffix,
   phone, email), independent of whatever `legend_detection.py`'s
   clustering geometry happened to find. Drops the candidate entirely --
   never renders or saves it -- rather than merely flagging it.
3. `_ocr_pii_zones()` / `_ocr_scan_rect_for_pii()`: re-checking the exact
   candidate layer 2 was added for showed it was STILL present and STILL
   leaking, because v0.1's own native text extraction
   (`text.py::extract_native_text_spans`, via PyMuPDF's `get_text()`)
   returns NOTHING for that part of the page -- confirmed directly with
   `get_text("words")` and `get_text("dict")`, not just the convenience
   wrapper. Root cause: some of that plan's title-block fields are drawn as
   vector outlines/curves (a common CAD-export setting, "convert text to
   paths"), which render as perfectly readable letters but are not text
   objects in the PDF at all -- invisible to EVERY text-layer extraction
   method, however it's called. Layer 2 cannot see what was never text to
   begin with. Layer 3 renders and OCRs the actual pixels instead: once per
   PAGE (not once per candidate -- prohibitively slow across hundreds of
   candidates), every detected legend/dense-text region is grouped with its
   neighbors within `OCR_ZONE_MERGE_DISTANCE_PT` (deliberately looser than
   `legend_detection.py`'s own clustering distance, since this real title
   block's fragments sat farther apart than that), padded, and OCR'd; any
   zone whose OCR'd text matches `privacy.py`'s patterns becomes a hard
   exclusion region for every candidate on that page. Fails SAFE (treats a
   render/OCR exception as PII) rather than silently passing an unreadable
   region through.

The annotation tool's `/api/candidate/{id}/wide_crop` endpoint (a live,
larger render the offline pipeline never precomputed or checked) carries
the equivalent of layers 2 and 3 itself (`_ocr_scan_rect_for_pii`, reused
directly rather than re-implemented) rather than relying on the offline
pipeline having already checked a smaller region.

**Privacy beyond the crop image: two MORE, independent vectors found while
auditing this batch's real output.** The three layers above guard the
rendered PIXELS of a crop. Two other places carry the SAME underlying
legend-entry text as plain TEXT, independent of any crop, and needed their
own separate fix:

- `candidates.py::_is_safe_plan_suggestion_label` — each candidate's
  `plan_specific_suggestion` (v0.3's own best-matching-legend-entry label,
  shown as a one-click suggestion button in the annotation tool and stored
  per-candidate in `manifest.json`) comes from `fact["plan_specific_
  evidence"]["label"]`, populated by v0.3's existing legend-matching logic
  with no reason to suspect the "legend entry" it matched against might
  actually be a company's contact line rather than a device row (the same
  root cause as the crop-image gap, seen from a different angle: legend
  detection's clustering is over-inclusive). Auditing this batch's real
  `qc.py` class-imbalance report found a phone number as the single MOST
  COMMON "suggested label" across all 20 plans' candidates before this fix.
  The fix drops only the suggestion (`plan_specific_suggestion = None`),
  never the candidate itself, since its crop image is independently already
  verified safe by the three layers above; `generic_suggestion` needs no
  such check, since it is always a fixed library name, never text scraped
  from the plan.
- `taxonomy.py::is_plausible_component_label` — `legend_entries[].
  normalized_label` text is aggregated into `classes.json`/`pipeline_
  summary.json`'s `taxonomy` block, which are committed files, independent
  of any crop or candidate. The same real plan's legend carried a supplier's
  phone number, postal code+town, and company name directly in this path;
  fixing it required a mandatory `privacy.scan_text` call plus a separate,
  non-privacy noise filter (regex/digit-density checks rejecting dimension
  and fixture-table rows like "kw: 3"/"ø63"/"dn100", which had been
  dominating the naive frequency ranking — see `taxonomy.py`'s own
  docstring for the full rationale and `docs/v04-dataset-report.md`'s §5
  for the resulting class list and its own documented residual gaps).

Fixing the first of these two surfaced a THIRD, separate bug: `privacy.py`'s
`SWISS_PLZ_TOWN_PATTERN` and `COMPANY_SUFFIX_PATTERN` were missing
`re.IGNORECASE` (unlike `STREET_PATTERN`/`TITLE_BLOCK_FIELD_PATTERN`, which
already had it) — since `legend_intelligence.py` lowercases its
`normalized_label` text, a real postal-code+town pair and a real company
name silently passed every check until this was fixed. All of `privacy.py`'s
patterns are now case-insensitive, with a regression test for exactly this.
A final, comprehensive automated scan of every string in the three
committed JSON output files (after all fixes) found zero genuine PII
matches — the only two hits were v0.1's own internal diagnostic warning
text ("Excluded 6784 of 7051 candidate line segments...", a segment count,
not plan content) coincidentally matching the same loose 4-digit+word
shape, confirmed benign by inspection.

**Coordinate-frame care, again**: `component_facts.py`'s
`_render_symbol_crop` was factored to share its exact DISPLAY-space clip-
rect computation (`_display_clip_rect`) with `candidates.py`, because that
function adds its OWN extra margin on top of whatever bbox it's given —
computing where a candidate lands WITHIN its saved crop (for the annotation
tool's highlight box) needs the crop's *actual* rendered origin, not the
bbox passed in to request it. Reusing the same function instead of
re-deriving the margin math avoids exactly the kind of silent, hard-to-spot
misalignment this whole engine's rotation-frame history has already run
into twice.

## Known v0.4 limitations (by design/measurement, not oversight)

- Only 1 of this batch's 20 real plans (DEV-18) is genuinely scanned/
  image-based with near-zero vector content; 4 more richly vector-based
  plans (DEV-02, DEV-09, DEV-16, DEV-20) still needed some or all of their
  text via v0.1's existing tiled-OCR fallback (`text.py`, unchanged;
  `text_source` "ocr"/"mixed" -- see `docs/v04-dataset-report.md`'s §1
  table). One OCR call triggered a genuine multi-minute tesseract hang on a
  single OCR tile during development -- a real, previously-unobserved
  robustness gap in that OCR path, surfaced by this batch's more varied
  real-world input rather than by W-001..W-010 or W-003. Not fixed inside
  `text.py` (out of scope, and v0.1's ten modules stay unchanged by
  design); worked around at the orchestration layer instead
  (`run_v04_pipeline.py` runs each plan in its own process group and kills
  the whole group on a timeout, so one hung OCR call can't take the batch
  down or leak an orphaned process).
- The SAME tesseract pathology above was found a second time, live, in
  `candidates.py`'s own OCR-based privacy layer (`_ocr_scan_rect_for_pii`,
  used by both `_ocr_pii_zones` during candidate generation and the
  annotation tool's `/wide_crop` endpoint) -- with no batch-level process
  group around it to catch a hang, one real request hung for 6+ minutes.
  Fixed with pytesseract's own per-call `timeout=` kwarg
  (`OCR_TIMEOUT_SECONDS`), which raises on expiry and is caught by the
  existing fail-SAFE exception handling (treated as PII, same as any other
  OCR failure) -- no batch-level process-group machinery needed at this
  smaller scope.
- Family grouping (`families.py`) uses filename-stem similarity, which
  generalizes only as well as this batch's own naming conventions did; a
  future batch with less informative filenames would need the near-
  duplicate/content-similarity signal to carry more weight.
- The symbol-zone/context-crop sizing constants were tuned against this
  batch's and W-003's real layouts, not against a large, diverse corpus.
- No scale calibration; no LLM anywhere in this pipeline, same as v0.1-v0.3.

## Known v0.1 limitations (by design, not oversight)

- Dimension-to-edge evidence only covers **directly associated** text (one
  `association.py` hop). The Base44 application's `vektorGraphMapper.ts` has
  additional, more sophisticated symbol-port dimension propagation with
  explicit ambiguity handling for competing labels — not reimplemented here.
- No scale/real-world-length calibration exists anywhere in this pipeline
  (inherited limitation from the original parser) — `length` fields are in
  PDF points, not calibrated distance. 4×ID checks are explicitly out of
  scope for v0.1.
- Topology is computed strictly per page; no cross-page or cross-file
  continuity is inferred (a leaf whose true continuation is drawn on another
  sheet is correctly reported as a proven dead-end *on this sheet*, which is
  all the evidence supports).
- No safety-device or apparatus *shape* classification exists (inherited from
  the original parser, which explicitly defers this to Base44's symbol
  library and AI review).

## Known v0.2 limitations (by design/measurement, not oversight)

- On the one real plan evaluated (W-003), **zero** components reached
  `COMPONENT_FACT` or `COMPONENT_CANDIDATE` — see
  `docs/w003-component-poc-report.md` for the full analysis of why (a
  reference-vs-plan drawing-style gap, verified not to be a matching-code
  defect).
- `detect_symbols()`'s clustering (v0.1, unchanged) can merge several nearby
  annotations into one oversized symbol candidate in dense plan areas,
  which directly hurts single-icon template matching; fixing it would mean
  touching v0.1's `symbols.py`, out of scope for this version.
- A full plan-specific legend parser was investigated (W-003 does have a
  usable legend block) but not built, per the task's explicit boundary.
- No scale calibration; recognized components carry PDF-point bboxes only.

## Known v0.3 limitations (by design/measurement, not oversight)

- Legend detection and entry extraction work very well on W-003 (its real
  "LEGENDE SANITÄR" legend is found automatically with the correct extent,
  and 29 of its real icon entries -- plus 8 correctly-flagged line-style
  swatches -- are extracted as usable, document-native templates), but
  **zero** components reached `COMPONENT_CANDIDATE` or `COMPONENT_FACT`
  when those templates were searched against the actual drawing -- see
  `docs/w003-legend-poc-report.md` for the full analysis. The best real
  score (0.59, below the 0.75 candidate floor) came from a v0.1
  `SymbolCandidate` whose own rendered crop was essentially a bare line, not
  a real icon silhouette -- direct evidence that v0.1's `symbols.py`
  clustering (already flagged as a v0.2 limitation above) is at least as
  large a factor here as any remaining style gap.
- `legend_entries.py`'s symbol-zone lookback (a fixed 58pt window to the
  left of a row's label) assumes the plan's own legend places its icon
  there, matching every legend column observed on W-003; a plan whose
  legend places icons elsewhere (right of the label, above it, or in a
  separate dedicated column with a very different offset) would need a
  different (or adaptive) lookback rule -- not implemented, since no second
  real plan was in scope for this version.
- The coded-lookup-table and line-style-swatch detectors (`legend_entries.py`)
  are shape/pattern heuristics validated against W-003's own real
  "Legende Verteilbatterie" and pipe-type swatches; they are deliberately
  conservative (biased toward rejecting/excluding when uncertain) but were
  only ever tuned against this one plan's actual legend content.
- No scale calibration; plan-specific templates and matches carry PDF-point
  bboxes only, same as v0.1/v0.2.
