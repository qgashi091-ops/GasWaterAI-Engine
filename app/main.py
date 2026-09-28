"""GasWaterAI Engine v0.1 -- external deterministic plan-analysis POC.

    POST /analyze   PDF bytes -> vector graph -> PlanFacts JSON
    GET  /health    liveness check

No database, no auth, no LLM call anywhere in this service -- see
docs/architecture.md for the full scope boundary.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .fingerprint import document_fingerprint
from .plan_analysis.pipeline import analyze_pdf_bytes
from .plan_analysis.plan_facts import build_document_facts

ENGINE_VERSION = "0.1.0"

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

    return JSONResponse({
        "engine_version": ENGINE_VERSION,
        "document_fingerprint": document_fingerprint(pdf_bytes),
        "pages": [p.model_dump() for p in doc.pages],
        "plan_facts": plan_facts,
        "diagnostics": {
            "filename": file.filename,
            "page_count": doc.page_count,
            "warnings": doc.warnings,
            "timing_ms": {"parse": parse_ms, "plan_facts": facts_ms, "total": parse_ms + facts_ms},
        },
    })
