"""v0.4 Phase 6 -- local annotation tool. A small FastAPI app + one static
HTML/JS page, run entirely on the product owner's own machine against the
local manifest + crop images this pipeline already generated. No Base44, no
network dependency, no database server.

Run:
    uvicorn app.annotation_tool.main:app --reload --port 8010
then open http://localhost:8010/

Configuration is via environment variables (never hardcoded paths, since
the manifest/crops/raw-PDF locations are local to whoever runs this):
    V04_MANIFEST_PATH   default: data/dev_plans_v04/annotation_dataset/manifest.json
    V04_CLASSES_PATH    default: data/dev_plans_v04/annotation_dataset/classes.json
    V04_RAW_PLANS_DIR   default: data/dev_plans_v04/raw  (for the on-demand wide-context view only)
"""
from __future__ import annotations

import os
from typing import Optional

import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from app.annotation_tool.store import AnnotationStore
from app.dataset_pipeline.candidates import _ocr_scan_rect_for_pii
from app.dataset_pipeline.privacy import scan_text
from app.plan_analysis.component_facts import _render_symbol_crop

MANIFEST_PATH = os.environ.get("V04_MANIFEST_PATH", "data/dev_plans_v04/annotation_dataset/manifest.json")
CLASSES_PATH = os.environ.get("V04_CLASSES_PATH", "data/dev_plans_v04/annotation_dataset/classes.json")
RAW_PLANS_DIR = os.environ.get("V04_RAW_PLANS_DIR", "data/dev_plans_v04/raw")
WIDE_CONTEXT_MULTIPLIER = 4.0  # relative to the candidate's own crop_bbox, for the "larger context" view

app = FastAPI(title="GasWaterAI v0.4 Annotation Tool")
_store: Optional[AnnotationStore] = None


def get_store() -> AnnotationStore:
    global _store
    if _store is None:
        if not os.path.exists(MANIFEST_PATH):
            raise HTTPException(status_code=500, detail=f"No manifest at {MANIFEST_PATH} -- run scripts/run_v04_pipeline.py first.")
        _store = AnnotationStore(MANIFEST_PATH)
    return _store


class AnnotateRequest(BaseModel):
    assigned_class: Optional[str] = None
    status: str = "LABELED"  # "LABELED" | "SKIPPED"
    corrected_bbox_in_crop_px: Optional[list] = None
    annotator_note: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/classes")
def api_classes() -> dict:
    if not os.path.exists(CLASSES_PATH):
        raise HTTPException(status_code=500, detail=f"No class list at {CLASSES_PATH}.")
    import json
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/progress")
def api_progress() -> dict:
    return get_store().progress()


@app.get("/api/candidates")
def api_candidates() -> list:
    """The full ordered id list -- the frontend keeps its own cursor over
    this, so previous/next/skip are pure client-side navigation and every
    other endpoint stays a simple by-id lookup."""
    return get_store().list_ids_in_order()


@app.get("/api/candidate/{candidate_id}")
def api_candidate(candidate_id: str) -> dict:
    rec = get_store().get(candidate_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown candidate_id")
    return rec


@app.get("/api/candidate/{candidate_id}/crop")
def api_candidate_crop(candidate_id: str):
    rec = get_store().get(candidate_id)
    if rec is None or not rec.get("crop_path") or not os.path.exists(rec["crop_path"]):
        raise HTTPException(status_code=404, detail="no crop image for this candidate")
    return FileResponse(rec["crop_path"], media_type="image/png")


@app.get("/api/candidate/{candidate_id}/wide_crop")
def api_candidate_wide_crop(candidate_id: str):
    """Renders a MUCH larger-context view live from the raw plan PDF (never
    precomputed/stored) -- the "optional larger-context view" the task
    calls for. Requires the raw PDF to still be present locally at
    V04_RAW_PLANS_DIR/{plan_id}.pdf, which is the normal case when this
    tool runs on the same machine the dataset pipeline ran on."""
    rec = get_store().get(candidate_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown candidate_id")
    pdf_path = os.path.join(RAW_PLANS_DIR, f"{rec['plan_id_pseudonymous']}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"raw PDF not found locally at {pdf_path} -- wide context view needs it")

    pdf = pymupdf.open(pdf_path)
    try:
        page = pdf[rec["page"] - 1]
        x0, y0, x1, y1 = rec["bbox_page_space"]  # DISPLAY-space, written by candidates.py
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        margin = WIDE_CONTEXT_MULTIPLIER * max(w, h)
        wide_display = pymupdf.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin) & page.rect

        # Same mandatory PII safety net as candidates.py's own crop
        # generation (see its module docstring for why this can't be
        # skipped just because legend_intelligence's geometric exclusion
        # already ran once, offline, at pipeline-build time): this is a
        # BIGGER, live-rendered region the offline pipeline never checked
        # at all, so it gets its own independent check here -- TWO layers,
        # matching candidates.py's own round 1 + round 2:
        #   1. text-layer regex scan (cheap, but blind to vector-outlined
        #      title-block text -- see candidates.py's module docstring for
        #      the real plan that exposed this exact gap)
        #   2. OCR the same region directly (_ocr_scan_rect_for_pii, the
        #      same mechanism candidates.py's _ocr_pii_zones uses) so this
        #      endpoint isn't relying on legend_intelligence's precomputed
        #      zones -- it has none here -- while still catching PII that
        #      was never real PDF text to begin with.
        wide_text = page.get_text("text", clip=wide_display)
        if scan_text("_wide_crop_check", wide_text).has_pii_risk:
            raise HTTPException(status_code=403, detail="wider context view withheld: overlaps text that looks like title-block/customer information")
        if _ocr_scan_rect_for_pii(page, wide_display):
            raise HTTPException(status_code=403, detail="wider context view withheld: overlaps text that looks like title-block/customer information")

        # _render_symbol_crop's contract takes a NATIVE-frame bbox and
        # applies page.rotation_matrix itself -- invert back to native here,
        # exactly like candidates.py does for the stored crop.
        native_rect = wide_display * ~page.rotation_matrix
        gray = _render_symbol_crop(page, (native_rect.x0, native_rect.y0, native_rect.x1, native_rect.y1))
        if gray is None:
            raise HTTPException(status_code=500, detail="could not render wide context crop")
        import cv2
        ok, buf = cv2.imencode(".png", gray)
        if not ok:
            raise HTTPException(status_code=500, detail="could not encode wide context crop")
        return Response(content=buf.tobytes(), media_type="image/png")
    finally:
        pdf.close()


@app.post("/api/candidate/{candidate_id}/annotate")
def api_annotate(candidate_id: str, body: AnnotateRequest) -> dict:
    store = get_store()
    if store.get(candidate_id) is None:
        raise HTTPException(status_code=404, detail="unknown candidate_id")
    if body.status == "LABELED" and not body.assigned_class:
        raise HTTPException(status_code=400, detail="assigned_class is required when status=LABELED")
    return store.annotate(
        candidate_id, assigned_class=body.assigned_class, status=body.status,
        corrected_bbox_in_crop_px=body.corrected_bbox_in_crop_px, annotator_note=body.annotator_note,
    )
