# GasWaterAI Engine v0.2 — external deterministic PoC

A small, standalone proof of concept answering two questions:

> Can a real water-installation plan PDF be converted into a reproducible technical `PlanFacts` model **without letting an LLM invent topology**? (v0.1)
>
> Can the engine also answer *what component is located at or near this proven graph position* — deterministically, without an LLM? (v0.2)

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
(v0.1) and `docs/w003-component-poc-report.md` (v0.2).

## Scope

```
PDF -> deterministic vector extraction -> normalized graph -> PlanFacts JSON
    -> deterministic topology analysis (cycles / dead-ends / unresolved)      [v0.1]
    -> component recognition against the 58-symbol SVGW library              [v0.2]
       (classical CV template matching, rotation/scale-invariant,
        COMPONENT_FACT / COMPONENT_CANDIDATE / UNRESOLVED)
```

Explicitly **not** in this version: Base44 integration, frontend, billing/auth,
SVGW compliance verdicts, the 150-case retrieval library, Visual-First, any
LLM topology or component inference, a trained/neural symbol detector, scale
calibration / 4×ID, cross-page topology or fusion, or a full legend parser
(investigated, not built — see `docs/w003-component-poc-report.md`). See
`docs/architecture.md` for the full boundary and which parser modules were
reused vs. newly written.

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
exact fixture (includes both `plan_facts` and `component_facts`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

32 tests, all real (no mocked PDF parsing): the real W-003 PDF is a
committed fixture (`tests/fixtures/`), and both v0.1's topology and v0.2's
component recognition have dedicated 10-repeated-run reproducibility tests
— the project's primary acceptance criterion. See `docs/w003-poc-report.md`
and `docs/w003-component-poc-report.md` for the actual results, and
`docs/symbol-audit.md` for the data audit behind the 58-symbol library.

## Performance

```bash
python3 scripts/measure_performance.py
```

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
