"""GasWaterAI Engine -- external deterministic plan-analysis POC.

    POST /analyze   PDF bytes -> vector graph -> PlanFacts + ComponentFacts JSON
    GET  /health    liveness check

No database, no auth, no LLM call anywhere in this service -- see
docs/architecture.md for the full scope boundary. v0.2 adds component
recognition (component_facts) strictly additively -- every v0.1 response
field is unchanged; see docs/w003-component-poc-report.md. v0.3 adds
plan-specific legend intelligence (legend_intelligence) the same way --
see docs/w003-legend-poc-report.md.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .fingerprint import document_fingerprint
from .plan_analysis.component_facts import build_component_facts
from .plan_analysis.legend_intelligence import build_legend_intelligence
from .plan_analysis.pipeline import analyze_pdf_bytes
from .plan_analysis.plan_facts import build_document_facts

ENGINE_VERSION = "0.3.0"

app = FastAPI(title="GasWaterAI Engine", version=ENGINE_VERSION)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine_version": ENGINE_VERSION}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    pdf_bytes = await file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")

    t0 = time.perf_counter()
    try:
        doc = analyze_pdf_bytes(pdf_bytes, filename=file.filename or "upload.pdf")
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller, not swallowed
        raise HTTPException(status_code=422, detail=f"PDF analysis failed: {exc}") from exc
    parse_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    plan_facts = build_document_facts(doc)
    facts_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    try:
        component_facts = build_component_facts(pdf_bytes, doc)
    except Exception as exc:  # noqa: BLE001 -- component recognition is additive; a failure here must never take down topology results
        component_facts = {"facts": [], "stats": {"error": str(exc)}}
    component_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    try:
        legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    except Exception as exc:  # noqa: BLE001 -- legend intelligence is additive; a failure here must never take down v0.1/v0.2 results
        legend_intelligence = {"legend_candidates": [], "legend_entries": [], "component_facts": [], "stats": {"error": str(exc)}}
    legend_ms = (time.perf_counter() - t3) * 1000

    return JSONResponse({
        "engine_version": ENGINE_VERSION,
        "document_fingerprint": document_fingerprint(pdf_bytes),
        "pages": [p.model_dump() for p in doc.pages],
        "plan_facts": plan_facts,
        "component_facts": component_facts,
        "legend_intelligence": legend_intelligence,
        "diagnostics": {
            "filename": file.filename,
            "page_count": doc.page_count,
            "warnings": doc.warnings,
            "timing_ms": {
                "parse": parse_ms, "plan_facts": facts_ms, "component_facts": component_ms,
                "legend_intelligence": legend_ms,
                "total": parse_ms + facts_ms + component_ms + legend_ms,
            },
        },
    })
