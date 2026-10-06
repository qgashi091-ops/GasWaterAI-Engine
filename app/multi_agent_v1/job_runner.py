"""Shared pipeline-invocation helper for the Multi-Agent v1 engine.

Both POST /multi_agent_v1/analyze (synchronous, unchanged) and the
asynchronous job worker (see job_manager.py, POST/GET /multi_agent_v1/jobs
in app/main.py) call `run_multi_agent_v1_analysis()` below -- this is the
exact body the synchronous route handler used before the job-mode epic
(PDF-magic check -> analyze_pdf_bytes -> deterministic preprocessing ->
run_multi_agent_v1 -> response-dict assembly), extracted unchanged into
one function so both callers are guaranteed byte-for-byte identical
results for identical input BY CONSTRUCTION -- never by maintaining two
separate call paths.

Request-scoping (`timing_diagnostics.start_request()`/`end_request()`/
`finish_and_log()`) stays each CALLER's own responsibility, not this
function's: the synchronous route wraps one HTTP request exactly as
before; the job worker wraps one background job AFTER its own submitting
HTTP request has already ended and returned 202 -- those are two
different timing-recorder scopes, and this function has no opinion on
which one it is running inside (it only calls the no-op-when-inactive
`timing_diagnostics.phase(...)` helper, same as before).
"""
from __future__ import annotations

import time
from typing import Optional

from ..fingerprint import document_fingerprint
from ..plan_analysis.canonical_inventory import build_canonical_inventory
from ..plan_analysis.component_evidence import build_component_evidence
from ..plan_analysis.legend_intelligence import build_legend_intelligence
from ..plan_analysis.pipeline import analyze_pdf_bytes
from ..plan_analysis.plan_facts import build_document_facts
from ..rules.engine import run_checks
from ..rules.water_rules import ALL_RULES
from . import timing_diagnostics
from .pipeline import ENGINE_VERSION, run_multi_agent_v1
from .provider import AgentModelProvider


class InvalidPdfError(ValueError):
    """Uploaded bytes do not start with the PDF magic number -- mirrors
    the synchronous endpoint's existing HTTP 400."""


class PdfAnalysisError(RuntimeError):
    """analyze_pdf_bytes() itself raised -- mirrors the synchronous
    endpoint's existing HTTP 422."""


def run_multi_agent_v1_analysis(
    pdf_bytes: bytes, provider: AgentModelProvider, filename: Optional[str] = None,
) -> dict:
    """Runs the EXACT existing multi_agent_v1 analysis and returns the
    same response body POST /multi_agent_v1/analyze has always returned
    (minus the JSONResponse wrapper). Raises InvalidPdfError/
    PdfAnalysisError instead of HTTPException so this function has no
    FastAPI dependency and can be called from a plain background thread;
    each caller maps these to its own error handling."""
    if not pdf_bytes.startswith(b"%PDF"):
        raise InvalidPdfError("Uploaded file is not a PDF.")

    t0 = time.perf_counter()
    with timing_diagnostics.phase("pdf_parsing"):
        try:
            doc = analyze_pdf_bytes(pdf_bytes, filename=filename or "upload.pdf")
        except Exception as exc:  # noqa: BLE001
            raise PdfAnalysisError(f"PDF analysis failed: {exc}") from exc
    parse_ms = (time.perf_counter() - t0) * 1000

    with timing_diagnostics.phase("deterministic_preprocessing"):
        plan_facts = build_document_facts(doc)

        try:
            legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
        except Exception as exc:  # noqa: BLE001
            legend_intelligence = {
                "legend_candidates": [], "legend_entries": [], "component_facts": [], "stats": {"error": str(exc)},
            }

        try:
            component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
        except Exception as exc:  # noqa: BLE001
            component_evidence = {"evidence": [], "stats": {"error": str(exc)}}

        inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
        check_results = run_checks(inventory["inventory"], ALL_RULES)

    t1 = time.perf_counter()
    result = run_multi_agent_v1(
        doc=doc, provider=provider, pdf_bytes=pdf_bytes, plan_facts=plan_facts,
        component_evidence=component_evidence.get("evidence", []), inventory=inventory["inventory"],
        rule_checks=check_results["checks"],
    )
    agents_ms = (time.perf_counter() - t1) * 1000

    return {
        "engine_version": ENGINE_VERSION,
        "document_fingerprint": document_fingerprint(pdf_bytes),
        **result.to_dict(),
        "diagnostics": {
            "filename": filename, "page_count": doc.page_count, "warnings": doc.warnings,
            "timing_ms": {"parse": parse_ms, "agents": agents_ms, "total": parse_ms + agents_ms},
        },
    }
