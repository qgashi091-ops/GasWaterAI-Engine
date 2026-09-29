# GasWaterAI Engine v0.4 — external deterministic PoC

A small, standalone proof of concept answering four questions:

> Can a real water-installation plan PDF be converted into a reproducible technical `PlanFacts` model **without letting an LLM invent topology**? (v0.1)
>
> Can the engine also answer *what component is located at or near this proven graph position* — deterministically, without an LLM? (v0.2)
>
> Can a plan's **own legend** become a temporary, plan-specific symbol library that recognizes components more reliably than a generic scanned reference set — still without any LLM? (v0.3)
>
> Can a batch of real, unseen plans be turned into a privacy-safe, human-reviewable annotation dataset for a *future* trained detector — without training anything yet? (v0.4)

## Why this exists

The Base44 application's multimodal LLM review classified the *same* W-003
crop, on the *same* plan, in response to the *same* question, as `Dead-End` /
`Dead-End` / `Loop` across three otherwise-identical runs — each reported at
high confidence. That is proof that free-form LLM interpretation is not a
reliable source of truth for pipe topology. This engine answers the same
class of question (cycle vs. dead-end vs. unresolved) **deterministically**,
straight from the vector graph a PDF's drawn geometry already encodes — no
LLM call anywhere in this service, for topology or for component
recognition.

Base44 remains the application/frontend layer. This repository is the
external plan-analysis engine that would replace its embedded parser call if
this PoC succeeds — see `docs/architecture.md`, `docs/w003-poc-report.md`
(v0.1), `docs/w003-component-poc-report.md` (v0.2),
`docs/w003-legend-poc-report.md` (v0.3) and `docs/v04-dataset-report.md`
(v0.4).

## Scope

```
PDF -> deterministic vector extraction -> normalized graph -> PlanFacts JSON
    -> deterministic topology analysis (cycles / dead-ends / unresolved)      [v0.1]
    -> component recognition against the 58-symbol SVGW library              [v0.2]
       (classical CV template matching, rotation/scale-invariant,
        COMPONENT_FACT / COMPONENT_CANDIDATE / UNRESOLVED)
    -> plan-specific legend intelligence                                     [v0.3]
       (detect the plan's OWN legend -> extract usable icon entries ->
        build document-local match templates -> search the drawing,
        excluding the legend itself -> hybrid recognition combining
        plan-specific + generic + text + graph evidence)
    -> real-plan dataset & annotation pipeline                               [v0.4]
       (audit + privacy-scan a batch of new real plans -> generate
        privacy-safe annotation-candidate crops -> propose a taxonomy from
        measured evidence -> local human annotation tool -> family-aware
        train/val/test split design -> detector-ready export format;
        trains nothing)
```

Explicitly **not** in this version: Base44 integration, frontend, billing/auth,
SVGW compliance verdicts, the 150-case retrieval library, Visual-First, any
LLM topology or component inference, a trained/neural symbol detector, scale
calibration / 4×ID, cross-page topology or fusion, and (per v0.3's own
evidence-based verdict — see `docs/w003-legend-poc-report.md`) training a
neural detector, which the report finds is now the *right* next step rather
than further classical-CV tuning. See `docs/architecture.md` for the full
boundary and which parser modules were reused vs. newly written.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
curl -F "file=@tests/fixtures/W-003_Referenzfall.Plan.pdf" http://localhost:8000/analyze
curl http://localhost:8000/health
```

See `examples/w003_planfacts.json` for a real response captured from this
exact fixture (includes both `plan_facts` and `component_facts`; regenerate
it to also see `legend_intelligence` in the same response).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

110 tests, all real (no mocked PDF parsing): the real W-003 PDF is a
committed fixture (`tests/fixtures/`), and v0.1's topology, v0.2's
component recognition, and v0.3's legend intelligence each have dedicated
10-repeated-run reproducibility tests — the project's primary acceptance
criterion. v0.4's dataset-pipeline and annotation-tool tests are unit/
synthetic (the real 20-plan batch is not part of the committed fixtures or
the per-commit test budget — see `docs/v04-dataset-report.md`). See
`docs/w003-poc-report.md`, `docs/w003-component-poc-report.md`,
`docs/w003-legend-poc-report.md` and `docs/v04-dataset-report.md` for the
actual results, and `docs/symbol-audit.md` for the data audit behind the
58-symbol library.

## Performance

```bash
python3 scripts/measure_performance.py
```

## v0.4: real-plan dataset & annotation pipeline

```bash
# 1. Place your own batch of real plan PDFs at data/dev_plans_v04/raw/,
#    named by a pseudonymous id (e.g. DEV-01.pdf .. DEV-20.pdf) -- this
#    directory is gitignored and never committed; see docs/v04-dataset-report.md
#    for why (some real filenames/title blocks carry customer PII).
python3 scripts/run_v04_pipeline.py

# 2. Review the generated candidates locally:
uvicorn app.annotation_tool.main:app --reload --port 8010
# open http://localhost:8010/
```

Every annotation crop passes THREE independent, stacked exclusion checks
before it is ever rendered or saved: (1) `legend_intelligence.py`'s own
geometric exclusion of anything inside a detected legend/dense-text region
(which usually already covers a title block), (2) a mandatory,
content-based regex scan (`candidates.py::_crop_contains_pii`) of every
text span actually inside the crop's own render region, and (3) an
OCR-based scan (`candidates.py::_ocr_pii_zones`) of the rendered pixels
around every detected legend/dense-text region. Layer 3 exists because
layer 2 was found insufficient on a real plan during development: some of
its title-block fields are drawn as vector-outlined text/paths (a CAD
"convert text to paths" export setting), which are perfectly readable as
pixels but are not text objects in the PDF at all — invisible to every
PDF text-layer extraction method, no matter how it's called. Only
rendering and OCRing the actual pixels catches that case (see
`docs/v04-dataset-report.md`'s privacy section for the full account). A
candidate is dropped entirely, never flagged-but-kept, if any check fires.
The annotation tool's own live "larger context" view carries the
equivalent of layers 2 and 3 as well, since it renders a bigger region the
offline pipeline never checked at all.

Crop IMAGES were only one of three places raw plan text was found leaking
during development, each a genuinely different mechanism (full account in
`docs/v04-dataset-report.md`'s privacy section): each candidate's
`plan_specific_suggestion` (a one-click suggestion button in the annotation
tool, stored per-candidate in `manifest.json`) and the proposed taxonomy's
own class names/evidence counts (`classes.json`, `pipeline_summary.json`)
are BOTH derived from the same legend-entry text independent of the crop
image, and both needed their own, separate mandatory privacy check
(`candidates.py::_is_safe_plan_suggestion_label`,
`taxonomy.py::is_plausible_component_label`) before this batch's real data
could be committed safely. `privacy.py`'s own regex patterns also had a
case-sensitivity bug (two of six were missing `re.IGNORECASE`) that let a
real postal-code+town pair and a company name pass every check once
lowercased — fixed, and now covered by a regression test.

## Every fact carries provenance

PlanFacts (v0.1, topology):

```json
{
  "fact_id": "PFxxxxxxxxxxxxxxxxxxxx",
  "fact_type": "cycle_membership",
  "kind": "DERIVED_FACT",
  "value": true,
  "source": "vector_graph",
  "evidence_status": "directly_drawn",
  "supporting_edges": ["e0", "e1", "e2", "e3"],
  "supporting_nodes": ["n0", "n1", "n2", "n3"]
}
```

`kind` is always exactly one of `FACT`, `DERIVED_FACT`, `UNRESOLVED` — never
a guess. A bridge (reconstructed) edge is stored but can never silently
become evidence for a directly-drawn claim.

ComponentFacts (v0.2, component recognition):

```json
{
  "component_fact_id": "CFxxxxxxxxxxxxxxxxxxxx",
  "kind": "COMPONENT_FACT",
  "symbol_type": "SYM-038",
  "symbol_name": "Wasserzähler",
  "confidence": 0.97,
  "recognition_method": "classical_cv_template_match_rotation_invariant",
  "graph_association": {"relation": "on_edge", "node_ids": [], "edge_ids": ["e42"]},
  "ambiguity_candidates": [{"symbol_id": "SYM-038", "name": "Wasserzähler", "score": 0.97}],
  "corroboration_status": "shape_evidence_sufficient",
  "supporting_evidence": {"best_score": 0.97, "second_best_score": 0.41, "margin": 0.56, "rotation_degrees": 0}
}
```

`kind` is always exactly one of `COMPONENT_FACT`, `COMPONENT_CANDIDATE`,
`UNRESOLVED`. A weak or ambiguous visual match is never promoted to a fact —
see `docs/symbol-audit.md` for which of the 58 reference symbols can and
cannot reach `COMPONENT_FACT` on shape evidence alone.

LegendIntelligence (v0.3, plan-specific legend templates + hybrid recognition):

```json
{
  "legend_candidates": [{"legend_id": "LG1_0", "bbox_display_space": [42.5, 268.5, 524.5, 502.3],
                          "detection_method": "text_density+heading_match", "confidence": 1.0,
                          "heading_text": "LEGENDE SANITÄR", "row_count": 40}],
  "legend_entries": [{"legend_entry_id": "LExxxxxxxxxxxxxxxxxxxx", "legend_id": "LG1_0",
                       "normalized_label": "wasserzähler", "classification": "USABLE_TEMPLATE",
                       "is_line_style_swatch": false, "symbol_bbox": [...]}],
  "component_facts": [{"component_fact_id": "HCxxxxxxxxxxxxxxxxxxxx", "kind": "UNRESOLVED",
                        "plan_specific_evidence": {"label": "ventil", "best_score": 0.59, ...},
                        "generic_evidence": {"kind": "UNRESOLVED", ...}, "hybrid_conflict": false}],
  "stats": {"legend_candidates_detected": 4, "legend_entries_extracted": 110,
            "plan_specific_templates_built": 29, "components_evaluated": 68,
            "by_kind": {"COMPONENT_FACT": 0, "COMPONENT_CANDIDATE": 0, "UNRESOLVED": 68}}
}
```

Every `legend_entries` row is classified `USABLE_TEMPLATE` / `TEXT_ONLY` /
`AMBIGUOUS` / `REJECTED` — never silently kept or dropped. A row is never
assumed usable: a lookalike lettered lookup table (W-003's own "Legende
Verteilbatterie") and a pure text/notes block are both detected structurally
and rejected as a whole, and a pipe-type line-style swatch is kept as
evidence but flagged `is_line_style_swatch` and excluded from becoming a
search template (it would otherwise match almost every drawn pipe of that
type). Every `component_facts` entry here is `kind` exactly one of
`COMPONENT_FACT` / `COMPONENT_CANDIDATE` / `UNRESOLVED`, same discipline as
v0.2, plus `hybrid_conflict`: `true` whenever the plan-specific and generic
recognition paths disagree on identity — a disagreement is always recorded,
never silently resolved by picking a side. See
`docs/w003-legend-poc-report.md` for the full v0.2-vs-v0.3 comparison, the
false-positive this discipline caught and fixed during development, and the
resulting verdict.
