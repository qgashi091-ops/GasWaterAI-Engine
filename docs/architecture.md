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
  -> plan_facts.build_document_facts()      (NEW — this PoC's actual contribution)
  -> app/main.py: POST /analyze             (NEW — thin FastAPI wrapper)
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

## API

`POST /analyze` (multipart PDF upload) → `{engine_version, document_fingerprint,
pages, plan_facts, diagnostics}`. `GET /health`. No database, no auth — out of
scope per the PoC's hard boundary.

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
