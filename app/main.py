"""GasWaterAI Engine -- external deterministic plan-analysis POC.

    POST /analyze   PDF bytes -> vector graph -> PlanFacts + ComponentFacts JSON
    POST /check     PDF bytes -> full pipeline -> canonical inventory -> deterministic rule checks
    GET  /health    liveness check

No database, no auth, no LLM call anywhere in /analyze -- see
docs/architecture.md for the full scope boundary. v0.2 adds component
recognition (component_facts) strictly additively -- every v0.1 response
field is unchanged; see docs/w003-component-poc-report.md. v0.3 adds
plan-specific legend intelligence (legend_intelligence) the same way --
see docs/w003-legend-poc-report.md. v1.0 adds /check (ENGINE v1 epic) as a
SEPARATE endpoint, additive only: /analyze's behavior and response shape
are completely unchanged. /check's own component-identity layer may call an
optional, explicitly-configured vision fallback (see app/vision_fallback/);
/analyze never does.
"""
from __future__ import annotations

import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .fingerprint import document_fingerprint
from .multi_agent_v1.pipeline import ENGINE_VERSION as MULTI_AGENT_V1_ENGINE_VERSION
from .multi_agent_v1.pipeline import run_multi_agent_v1
from .multi_agent_v1.provider import AnthropicAgentModelProvider
from .plan_analysis.canonical_inventory import build_canonical_inventory
from .plan_analysis.component_evidence import build_component_evidence
from .plan_analysis.component_facts import build_component_facts
from .plan_analysis.legend_intelligence import build_legend_intelligence
from .plan_analysis.pipeline import analyze_pdf_bytes
from .plan_analysis.plan_facts import build_document_facts
from .rules.engine import run_checks
from .rules.water_rules import ALL_RULES

ENGINE_VERSION = "1.0.0"

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


@app.post("/check")
async def check(file: UploadFile = File(...)) -> JSONResponse:
    """PDF -> PlanFacts + component evidence fusion (Detector v1, no vision
    fallback by default -- see app/plan_analysis/component_evidence.py) ->
    canonical inventory -> deterministic rule checks. Never trains, never
    calls Base44, never produces a free-form LLM verdict (see
    app/rules/engine.py's own docstring)."""
    pdf_bytes = await file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")

    t0 = time.perf_counter()
    try:
        doc = analyze_pdf_bytes(pdf_bytes, filename=file.filename or "upload.pdf")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"PDF analysis failed: {exc}") from exc
    parse_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    plan_facts = build_document_facts(doc)
    facts_ms = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    try:
        legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    except Exception as exc:  # noqa: BLE001
        legend_intelligence = {"legend_candidates": [], "legend_entries": [], "component_facts": [], "stats": {"error": str(exc)}}
    legend_ms = (time.perf_counter() - t2) * 1000

    t3 = time.perf_counter()
    try:
        component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
    except Exception as exc:  # noqa: BLE001 -- fusion failure must never take down topology results
        component_evidence = {"evidence": [], "stats": {"error": str(exc)}}
    evidence_ms = (time.perf_counter() - t3) * 1000

    t4 = time.perf_counter()
    inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
    inventory_ms = (time.perf_counter() - t4) * 1000

    t5 = time.perf_counter()
    check_results = run_checks(inventory["inventory"], ALL_RULES)
    checks_ms = (time.perf_counter() - t5) * 1000

    total_ms = parse_ms + facts_ms + legend_ms + evidence_ms + inventory_ms + checks_ms

    return JSONResponse({
        "engine_version": ENGINE_VERSION,
        "document_fingerprint": document_fingerprint(pdf_bytes),
        "inventory": inventory["inventory"],
        "checks": check_results["checks"],
        "diagnostics": {
            "filename": file.filename,
            "page_count": doc.page_count,
            "warnings": doc.warnings,
            "inventory_stats": inventory["stats"],
            "component_evidence_stats": component_evidence.get("stats"),
            "check_stats": check_results["stats"],
            "timing_ms": {
                "parse": parse_ms, "plan_facts": facts_ms, "legend_intelligence": legend_ms,
                "component_evidence": evidence_ms, "canonical_inventory": inventory_ms,
                "rule_checks": checks_ms, "total": total_ms,
            },
        },
    })


@app.post("/multi_agent_v1/analyze")
async def multi_agent_v1_analyze(file: UploadFile = File(...)) -> JSONResponse:
    """SEPARATE, additive test path for the Multi-Agent Architecture v1
    (see app/multi_agent_v1/) -- does not touch /analyze or /check's
    behavior or response shape at all. Reuses every existing deterministic
    step exactly as /check does (same analyze_pdf_bytes, plan_facts,
    legend_intelligence, component_evidence, canonical_inventory, rule
    checks) and additionally runs the 12-agent pipeline on top, via
    AnthropicAgentModelProvider -- which, absent a configured
    GASWATERAI_VISION_API_KEY (none exists in this build/deployment
    environment), reports each model call as unavailable rather than
    failing the request; every deterministic-first agent path (plan-area
    keyword/legend detection, medium/category/circulation text patterns,
    the two model-free reconciliation agents) still runs and still produces
    real observations with zero calls."""
    pdf_bytes = await file.read()
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")

    t0 = time.perf_counter()
    try:
        doc = analyze_pdf_bytes(pdf_bytes, filename=file.filename or "upload.pdf")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"PDF analysis failed: {exc}") from exc
    parse_ms = (time.perf_counter() - t0) * 1000

    plan_facts = build_document_facts(doc)

    try:
        legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    except Exception as exc:  # noqa: BLE001
        legend_intelligence = {"legend_candidates": [], "legend_entries": [], "component_facts": [], "stats": {"error": str(exc)}}

    try:
        component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
    except Exception as exc:  # noqa: BLE001
        component_evidence = {"evidence": [], "stats": {"error": str(exc)}}

    inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
    check_results = run_checks(inventory["inventory"], ALL_RULES)

    t1 = time.perf_counter()
    result = run_multi_agent_v1(
        doc=doc, provider=AnthropicAgentModelProvider(), pdf_bytes=pdf_bytes, plan_facts=plan_facts,
        component_evidence=component_evidence.get("evidence", []), inventory=inventory["inventory"],
        rule_checks=check_results["checks"],
    )
    agents_ms = (time.perf_counter() - t1) * 1000

    return JSONResponse({
        "engine_version": MULTI_AGENT_V1_ENGINE_VERSION,
        "document_fingerprint": document_fingerprint(pdf_bytes),
        **result.to_dict(),
        "diagnostics": {
            "filename": file.filename, "page_count": doc.page_count, "warnings": doc.warnings,
            "timing_ms": {"parse": parse_ms, "agents": agents_ms, "total": parse_ms + agents_ms},
        },
    })
