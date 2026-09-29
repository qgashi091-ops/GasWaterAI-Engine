"""v0.4 Phase 1 -- dataset audit for a batch of new, real development plans.

Runs the EXISTING, unmodified v0.1/v0.2/v0.3 engine (pipeline.py,
plan_facts.py, component_facts.py, legend_intelligence.py) against each
plan and reports success/failure honestly -- a plan the engine cannot
parse is reported as a failure with its error, never silently skipped or
forced through. No thresholds are touched; this module only observes.

Plans are identified throughout by a PSEUDONYMOUS id (e.g. "DEV-07"), never
by their original filename -- several of the 20 real filenames embed a
customer address or surname directly (see privacy.py), so even an audit
report listing filenames would leak PII. The pseudonym<->filename mapping
lives only in a local, gitignored file the pipeline writes next to the raw
PDFs; nothing under app/ or docs/ ever reads or repeats it.
"""
from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from app.plan_analysis.component_facts import build_component_facts
from app.plan_analysis.legend_intelligence import build_legend_intelligence
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts

# A page is classified "scanned/raster" rather than "vector" when it has
# native text/vector density far below what every real CAD-exported plan in
# this engine's holdout (W-001..W-010) and this batch's vector plans show.
# This is a coarse, transparent heuristic (not a format sniff) precisely so
# it is auditable: two direct signals, either of which alone can indicate a
# scan (a raster-embedded PDF can still expose a handful of vector
# rectangles for its border).
RASTER_TEXT_CHAR_FLOOR = 50
RASTER_DRAWING_FLOOR = 200


@dataclass
class PlanAudit:
    plan_id: str
    page_count: int
    parser_outcome: str  # "ok" | "failed_to_open" | "failed_to_parse"
    error: Optional[str] = None
    rotation: Optional[int] = None
    width_pt: Optional[float] = None
    height_pt: Optional[float] = None
    native_text_chars: Optional[int] = None
    raw_drawing_count: Optional[int] = None
    is_likely_scanned: Optional[bool] = None
    text_source: Optional[str] = None
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None
    symbol_candidates: Optional[int] = None
    plan_facts_count: Optional[int] = None
    component_facts_stats: Optional[dict] = None
    legend_detected: bool = False
    legend_top_confidence: Optional[float] = None
    legend_heading: Optional[str] = None
    usable_legend_entries: Optional[int] = None
    legend_intelligence_stats: Optional[dict] = None
    runtime_seconds: Optional[float] = None
    peak_rss_delta_kb: Optional[int] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _current_rss_kb() -> int:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def audit_plan(plan_id: str, pdf_bytes: bytes) -> PlanAudit:
    t0 = time.perf_counter()
    rss0 = _current_rss_kb()
    audit = PlanAudit(plan_id=plan_id, page_count=0, parser_outcome="ok")

    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 -- a real batch WILL include unopenable files
        audit.parser_outcome = "failed_to_open"
        audit.error = f"{type(exc).__name__}: {exc}"
        audit.runtime_seconds = time.perf_counter() - t0
        return audit

    audit.page_count = len(pdf)
    page0 = pdf[0]
    audit.rotation = page0.rotation
    audit.width_pt = round(page0.rect.width, 1)
    audit.height_pt = round(page0.rect.height, 1)
    raw_text = page0.get_text()
    audit.native_text_chars = len(raw_text)
    try:
        audit.raw_drawing_count = len(page0.get_drawings())
    except Exception:  # noqa: BLE001
        audit.raw_drawing_count = None
    pdf.close()

    audit.is_likely_scanned = (
        audit.native_text_chars < RASTER_TEXT_CHAR_FLOOR
        and (audit.raw_drawing_count or 0) < RASTER_DRAWING_FLOOR
    )

    try:
        doc = analyze_pdf_bytes(pdf_bytes, filename=f"{plan_id}.pdf")
    except Exception as exc:  # noqa: BLE001
        audit.parser_outcome = "failed_to_parse"
        audit.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        audit.runtime_seconds = time.perf_counter() - t0
        audit.peak_rss_delta_kb = _current_rss_kb() - rss0
        return audit

    page_model = doc.pages[0]
    audit.text_source = page_model.text_source
    audit.graph_nodes = len(page_model.graph.nodes)
    audit.graph_edges = len(page_model.graph.edges)
    audit.symbol_candidates = len(page_model.symbols)
    audit.warnings = list(doc.warnings) + list(page_model.warnings)

    try:
        plan_facts = build_document_facts(doc)
        audit.plan_facts_count = plan_facts["stats"]["fact_count"]
    except Exception as exc:  # noqa: BLE001
        audit.warnings.append(f"plan_facts failed: {exc}")

    try:
        component_facts = build_component_facts(pdf_bytes, doc)
        audit.component_facts_stats = component_facts["stats"]
    except Exception as exc:  # noqa: BLE001
        audit.warnings.append(f"component_facts failed: {exc}")

    try:
        legend_result = build_legend_intelligence(pdf_bytes, doc)
        audit.legend_intelligence_stats = legend_result["stats"]
        candidates = legend_result["legend_candidates"]
        if candidates:
            top = max(candidates, key=lambda c: c["confidence"])
            audit.legend_detected = True
            audit.legend_top_confidence = top["confidence"]
            audit.legend_heading = top["heading_text"]
        usable = [e for e in legend_result["legend_entries"] if e["classification"] == "USABLE_TEMPLATE"]
        audit.usable_legend_entries = len(usable)
    except Exception as exc:  # noqa: BLE001
        audit.warnings.append(f"legend_intelligence failed: {exc}")

    audit.runtime_seconds = round(time.perf_counter() - t0, 3)
    audit.peak_rss_delta_kb = _current_rss_kb() - rss0
    return audit


def audit_batch(plans: dict[str, bytes]) -> list[PlanAudit]:
    """plans: {plan_id: pdf_bytes}. Order-preserving over dict insertion order."""
    return [audit_plan(pid, data) for pid, data in plans.items()]


def summarize_batch(audits: list[PlanAudit]) -> dict:
    ok = [a for a in audits if a.parser_outcome == "ok"]
    failed = [a for a in audits if a.parser_outcome != "ok"]
    runtimes = [a.runtime_seconds for a in audits if a.runtime_seconds is not None]
    return {
        "total_plans": len(audits),
        "parser_success": len(ok),
        "parser_failed": len(failed),
        "failed_plan_ids": [a.plan_id for a in failed],
        "likely_scanned_plan_ids": [a.plan_id for a in audits if a.is_likely_scanned],
        "legend_detected_count": sum(1 for a in audits if a.legend_detected),
        "total_symbol_candidates": sum(a.symbol_candidates or 0 for a in audits),
        "total_usable_legend_entries": sum(a.usable_legend_entries or 0 for a in audits),
        "runtime_seconds": {
            "min": round(min(runtimes), 3) if runtimes else None,
            "max": round(max(runtimes), 3) if runtimes else None,
            "mean": round(statistics.mean(runtimes), 3) if runtimes else None,
            "total": round(sum(runtimes), 3) if runtimes else None,
        },
    }
