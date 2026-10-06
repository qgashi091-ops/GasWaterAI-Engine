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

import hmac
import os
import time

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .fingerprint import document_fingerprint
from .multi_agent_v1.base44_gateway_diagnostics import run_diagnostics as run_base44_gateway_diagnostics
from .multi_agent_v1.base44_provider import Base44AgentModelProvider
from .multi_agent_v1.base44_provider import GATEWAY_URL_ENV_VAR as BASE44_AI_GATEWAY_URL_ENV_VAR
from .multi_agent_v1.job_manager import QueueFullError, get_job_manager
from .multi_agent_v1.job_runner import InvalidPdfError, PdfAnalysisError, run_multi_agent_v1_analysis
from .multi_agent_v1.job_store import COMPLETED, FAILED, QUEUED
from .multi_agent_v1.provider import AgentModelProvider, AnthropicAgentModelProvider
from .multi_agent_v1 import timing_diagnostics
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

# ---------------------------------------------------------------------------
# /multi_agent_v1/analyze access control.
#
# Shared-secret check, prepared but not forced on: if GASWATERAI_MULTI_AGENT_
# API_KEY is unset (today's default, and every existing test's environment),
# the endpoint stays exactly as open as it is right now -- deploying this
# unchanged would not lock anyone out. Setting that variable on the hosting
# platform is what activates enforcement; nothing in code needs to change
# again to turn it on. /analyze and /check are completely untouched -- no
# dependency is attached to either.
# ---------------------------------------------------------------------------
MULTI_AGENT_API_KEY_ENV_VAR = "GASWATERAI_MULTI_AGENT_API_KEY"


def _verify_multi_agent_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.environ.get(MULTI_AGENT_API_KEY_ENV_VAR)
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


def _require_multi_agent_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Stricter sibling of `_verify_multi_agent_api_key`, for
    /diagnostics/base44-gateway only: same env var, same X-API-Key header,
    same constant-time comparison -- but never fails open. That lenient
    endpoint intentionally stays as open as before until someone configures
    GASWATERAI_MULTI_AGENT_API_KEY; this diagnostic probe must refuse to run
    at all without a valid key, configured or not."""
    expected = os.environ.get(MULTI_AGENT_API_KEY_ENV_VAR)
    if not expected:
        raise HTTPException(status_code=503, detail=f"{MULTI_AGENT_API_KEY_ENV_VAR} is not configured.")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key.")


def _select_agent_model_provider() -> AgentModelProvider:
    """Base44AgentModelProvider is preferred whenever Base44's AI Gateway is
    configured (BASE44_AI_GATEWAY_URL set) -- this is what lets model-
    dependent agents run WITHOUT a separately-paid Anthropic API key for
    product operation. AnthropicAgentModelProvider remains the fallback
    (e.g. local development with a personal key, or before Base44's gateway
    exists) and is exactly what ran before this change -- with neither
    configured, behavior is unchanged from before: every model call reports
    `available: false` honestly, nothing fails."""
    if os.environ.get(BASE44_AI_GATEWAY_URL_ENV_VAR):
        return Base44AgentModelProvider()
    return AnthropicAgentModelProvider()


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


@app.post("/multi_agent_v1/analyze", dependencies=[Depends(_verify_multi_agent_api_key)])
async def multi_agent_v1_analyze(file: UploadFile = File(...)) -> JSONResponse:
    """SEPARATE, additive test path for the Multi-Agent Architecture v1
    (see app/multi_agent_v1/) -- does not touch /analyze or /check's
    behavior or response shape at all. Reuses every existing deterministic
    step exactly as /check does (same analyze_pdf_bytes, plan_facts,
    legend_intelligence, component_evidence, canonical_inventory, rule
    checks) and additionally runs the 12-agent pipeline on top, via
    whichever AgentModelProvider `_select_agent_model_provider()` picks
    (Base44's AI Gateway when configured, else the direct Anthropic path).
    Absent BOTH, every model call reports unavailable rather than failing
    the request; every deterministic-first agent path (plan-area keyword/
    legend detection, medium/category/circulation text patterns, the agents
    that never call a model at all) still runs and still produces real
    observations with zero calls. Protected by `_verify_multi_agent_api_key`
    (X-API-Key header) once GASWATERAI_MULTI_AGENT_API_KEY is configured;
    /analyze and /check carry no such dependency and are unaffected.
    SEPARATE from the asynchronous POST/GET /multi_agent_v1/jobs pair below
    -- both call the identical `run_multi_agent_v1_analysis()` helper
    (see job_runner.py), so this route's behavior/response shape is
    completely unchanged by the job-mode addition; it remains the
    diagnosis/development path (section 3 of the job-mode spec)."""
    timing_token = timing_diagnostics.start_request()
    try:
        pdf_bytes = await file.read()
        try:
            result_dict = run_multi_agent_v1_analysis(pdf_bytes, _select_agent_model_provider(), filename=file.filename)
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PdfAnalysisError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        timing_diagnostics.finish_and_log()
        return JSONResponse(result_dict)
    finally:
        timing_diagnostics.end_request(timing_token)


JOB_MAX_UPLOAD_BYTES_ENV_VAR = "GASWATERAI_JOB_MAX_UPLOAD_BYTES"
DEFAULT_JOB_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def _job_max_upload_bytes() -> int:
    try:
        value = int(os.environ.get(JOB_MAX_UPLOAD_BYTES_ENV_VAR, DEFAULT_JOB_MAX_UPLOAD_BYTES))
    except (TypeError, ValueError):
        return DEFAULT_JOB_MAX_UPLOAD_BYTES
    return value if value >= 1 else DEFAULT_JOB_MAX_UPLOAD_BYTES


def _job_status_dict(record) -> dict:
    def _iso(dt):
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z") if dt is not None else None

    return {
        "job_id": record.job_id,
        "status": record.status,
        "created_at": _iso(record.created_at),
        "started_at": _iso(record.started_at),
        "completed_at": _iso(record.completed_at),
        "result": record.result if record.status == COMPLETED else None,
        "error": record.error if record.status == FAILED else None,
    }


@app.post("/multi_agent_v1/jobs", status_code=202, dependencies=[Depends(_verify_multi_agent_api_key)])
async def multi_agent_v1_submit_job(file: UploadFile = File(...)) -> JSONResponse:
    """Asynchronous entry point for the Multi-Agent v1 analysis (job-mode
    epic): accepts the identical PDF upload as POST /multi_agent_v1/analyze
    but does NOT wait for the analysis -- it enqueues the EXACT SAME
    pipeline (via `run_multi_agent_v1_analysis()`, see job_runner.py) onto
    a bounded background worker pool (see job_manager.py) and returns
    immediately with 202. This exists because a full plan analysis can
    take several minutes, far longer than Base44's own ~120s request
    timeout -- see docs/multi-agent-v1-job-mode-contract.md for the full
    polling contract a caller must follow.

    Protected by the SAME `_verify_multi_agent_api_key` dependency as
    POST /multi_agent_v1/analyze (fails open until
    GASWATERAI_MULTI_AGENT_API_KEY is configured). Idempotent on input:
    submitting the identical PDF bytes while a job for them is already
    queued/processing returns that job's id instead of starting a second,
    costly analysis (document_fingerprint-based, see job_manager.py)."""
    pdf_bytes = await file.read()
    max_bytes = _job_max_upload_bytes()
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the job endpoint's {max_bytes} byte limit.",
        )
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")

    manager = get_job_manager()
    fingerprint = document_fingerprint(pdf_bytes)
    try:
        job_id, _created = manager.submit(
            pdf_bytes, _select_agent_model_provider(), fingerprint, filename=file.filename,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # On an idempotent hit (section 7: an identical input already has a job
    # in flight) the returned job may already be PROCESSING, not QUEUED --
    # report its REAL current status rather than always claiming "queued".
    record = manager.get(job_id)
    status_value = record.status if record is not None else QUEUED
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": status_value})


@app.get("/multi_agent_v1/jobs/{job_id}", dependencies=[Depends(_verify_multi_agent_api_key)])
async def multi_agent_v1_get_job(job_id: str) -> JSONResponse:
    """Status/result poll for a job created by POST /multi_agent_v1/jobs.
    Same X-API-Key protection. Status is exclusively one of queued,
    processing, completed, failed -- see job_store.py. `result` is
    populated only once `completed` (the identical body
    POST /multi_agent_v1/analyze would have returned for the same input);
    `error` is populated only once `failed`, and is always a technical
    message, never plan content or a secret."""
    manager = get_job_manager()
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")
    return JSONResponse(_job_status_dict(record))


@app.post("/diagnostics/base44-gateway", dependencies=[Depends(_require_multi_agent_api_key)])
async def diagnostics_base44_gateway() -> JSONResponse:
    """TEMPORARY -- diagnoses the live Render -> Base44 AI Gateway 403 from
    inside the actual running engine (see
    app/multi_agent_v1/base44_gateway_diagnostics.py for the full probe).
    Runs no plan check, no agent, no further model call beyond the single
    direct HTTP probe(s) it sends to Base44 (at most 2, never a retry loop).
    Requires a valid X-API-Key even if GASWATERAI_MULTI_AGENT_API_KEY is
    unset (`_require_multi_agent_api_key`, stricter than
    `_verify_multi_agent_api_key`) -- no diagnosis runs without one. DELETE
    this route (and base44_gateway_diagnostics.py) once the incident is
    resolved."""
    return JSONResponse(run_base44_gateway_diagnostics())
