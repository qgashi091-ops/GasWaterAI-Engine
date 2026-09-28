# GasWaterAI Engine v0.1 — external deterministic PoC

A small, standalone proof of concept answering one question:

> Can a real water-installation plan PDF be converted into a reproducible technical `PlanFacts` model **without letting an LLM invent topology**?

## Why this exists

The Base44 application's multimodal LLM review classified the *same* W-003
crop, on the *same* plan, in response to the *same* question, as `Dead-End` /
`Dead-End` / `Loop` across three otherwise-identical runs — each reported at
high confidence. That is proof that free-form LLM interpretation is not a
reliable source of truth for pipe topology. This engine answers the same
class of question (cycle vs. dead-end vs. unresolved) **deterministically**,
straight from the vector graph a PDF's drawn geometry already encodes — no
LLM call anywhere in this service.

Base44 remains the application/frontend layer. This repository is the
external plan-analysis engine that would replace its embedded parser call if
this PoC succeeds — see `docs/architecture.md` and `docs/w003-poc-report.md`.

## Scope (v0.1)

```
PDF -> deterministic vector extraction -> normalized graph -> PlanFacts JSON
    -> deterministic topology analysis (cycles / dead-ends / unresolved)
```

Explicitly **not** in this version: Base44 integration, frontend, billing/auth,
SVGW compliance verdicts, the 150-case retrieval library, Visual-First, any
LLM topology inference, symbol-classifier training, scale calibration / 4×ID,
or cross-page topology. See `docs/architecture.md` for the full boundary and
which parser modules were reused vs. newly written.

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
exact fixture.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Includes the real W-003 PDF as a committed fixture (`tests/fixtures/`) and a
10-repeated-run reproducibility test — the project's primary acceptance
criterion. See `docs/w003-poc-report.md` for the actual results.

## Performance

```bash
python3 scripts/measure_performance.py
```

## Every PlanFact carries provenance

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
