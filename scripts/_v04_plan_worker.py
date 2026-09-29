"""Single-plan v0.4 worker: audit + privacy scan + candidate/crop
generation, all in ONE subprocess invocation so the (expensive) v0.1/v0.2/v0.3
engine run happens exactly once per plan, not once for the audit and again
for candidates. Run as its own subprocess (by run_v04_pipeline.py) with a
hard timeout, so one pathologically large/slow real plan can be killed
without taking the whole 20-plan batch down with it.

Usage: python3 scripts/_v04_plan_worker.py <plan_id> <pdf_path> <crops_dir> <out_json_path>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf  # noqa: E402

from app.dataset_pipeline.audit import PlanAudit, _current_rss_kb  # noqa: E402
from app.dataset_pipeline.candidates import build_candidates_for_plan  # noqa: E402
from app.dataset_pipeline.privacy import scan_text  # noqa: E402
from app.plan_analysis.component_facts import build_component_facts  # noqa: E402
from app.plan_analysis.legend_intelligence import build_legend_intelligence  # noqa: E402
from app.plan_analysis.pipeline import analyze_pdf_bytes  # noqa: E402
from app.plan_analysis.plan_facts import build_document_facts  # noqa: E402

RASTER_TEXT_CHAR_FLOOR = 50
RASTER_DRAWING_FLOOR = 200


def main() -> None:
    plan_id, pdf_path, crops_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    pdf_bytes = Path(pdf_path).read_bytes()
    t0 = time.perf_counter()
    rss0 = _current_rss_kb()

    audit = PlanAudit(plan_id=plan_id, page_count=0, parser_outcome="ok")
    candidates_out: list = []
    privacy_out: dict = {"plan_id": plan_id, "has_pii_risk": False, "findings": [], "candidates_excluded_for_pii": 0}

    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        audit.parser_outcome = "failed_to_open"
        audit.error = f"{type(exc).__name__}: {exc}"
        _write(out_path, audit, candidates_out, privacy_out)
        return

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
        audit.native_text_chars < RASTER_TEXT_CHAR_FLOOR and (audit.raw_drawing_count or 0) < RASTER_DRAWING_FLOOR
    )

    priv = scan_text(plan_id, raw_text)
    privacy_out = priv.to_dict()

    try:
        doc = analyze_pdf_bytes(pdf_bytes, filename=f"{plan_id}.pdf")
    except Exception as exc:  # noqa: BLE001
        import traceback
        audit.parser_outcome = "failed_to_parse"
        audit.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
        audit.runtime_seconds = round(time.perf_counter() - t0, 3)
        audit.peak_rss_delta_kb = _current_rss_kb() - rss0
        _write(out_path, audit, candidates_out, privacy_out)
        return

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

    legend_result = None
    try:
        legend_result = build_legend_intelligence(pdf_bytes, doc)
        audit.legend_intelligence_stats = legend_result["stats"]
        cands = legend_result["legend_candidates"]
        if cands:
            top = max(cands, key=lambda c: c["confidence"])
            audit.legend_detected = True
            audit.legend_top_confidence = top["confidence"]
            audit.legend_heading = top["heading_text"]
        usable = [e for e in legend_result["legend_entries"] if e["classification"] == "USABLE_TEMPLATE"]
        audit.usable_legend_entries = len(usable)
    except Exception as exc:  # noqa: BLE001
        audit.warnings.append(f"legend_intelligence failed: {exc}")

    pii_excluded_count = 0
    if legend_result is not None:
        try:
            cands, pii_excluded_count = build_candidates_for_plan(plan_id, pdf_bytes, doc, legend_result, crops_dir)
            candidates_out = [c.to_dict() for c in cands]
        except Exception as exc:  # noqa: BLE001
            audit.warnings.append(f"candidate generation failed: {exc}")
    privacy_out["candidates_excluded_for_pii"] = pii_excluded_count

    audit.runtime_seconds = round(time.perf_counter() - t0, 3)
    audit.peak_rss_delta_kb = _current_rss_kb() - rss0

    # Also persist the doc + legend_result so the orchestrator can build the
    # taxonomy (needs text_spans' label_hint and legend entries) without a
    # third re-parse -- kept minimal (just what taxonomy.py needs).
    taxonomy_evidence = {
        "legend_entries": legend_result["legend_entries"] if legend_result else [],
        "component_facts": legend_result["component_facts"] if legend_result else [],
        "safety_device_codes": [
            span.text.strip().upper()
            for p in doc.pages for span in p.text_spans if span.label_hint == "safety_device_code"
        ],
    }

    _write(out_path, audit, candidates_out, privacy_out, taxonomy_evidence)


def _write(out_path: str, audit: PlanAudit, candidates: list, privacy: dict, taxonomy_evidence: dict = None) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit": audit.to_dict(),
            "candidates": candidates,
            "privacy": privacy,
            "taxonomy_evidence": taxonomy_evidence or {},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
