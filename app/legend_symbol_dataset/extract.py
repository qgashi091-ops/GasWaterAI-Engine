"""Extracts every real, potable-water-relevant legend symbol from one
plan's own PDF -- the sole data source for the new active annotation
workflow (see package docstring).

Reuses, unmodified:
- `app.plan_analysis.legend_detection.detect_legend_candidates` /
  `app.plan_analysis.legend_entries.classify_candidate_rows` for legend
  region detection and per-row icon/label/ambiguity classification (the
  same non-raster-identity code the earlier structural benchmark reused).
- `app.dataset_pipeline.privacy`'s three independent PII safety nets,
  exactly as `app.dataset_pipeline.candidates` already applies them to
  crops rendered from this same plan set: a regex check of the legend
  label text itself, a regex check of every native text span overlapping
  the rendered crop region, and an OCR-based check of the same region
  (catches vector-outlined "text" invisible to the native text layer --
  this is exactly the failure mode `candidates.py`'s own module docstring
  documents finding and fixing, round 2).

COORDINATE SPACES (get this wrong and crops silently show the wrong part
of a rotated page -- found the hard way twice already in this project):
`legend_detection`/`legend_entries` work entirely in DISPLAY space (their
own design, so a rotated page's "row" means what a human sees). v0.1's own
`schema.TextSpan.bbox` is in the PDF's NATIVE (unrotated) frame. Every
legend-derived bbox in this module is DISPLAY space throughout; text spans
are converted to display space once per page before any comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pymupdf
import pytesseract

from app.dataset_pipeline.candidates import (
    OCR_TIMEOUT_SECONDS,
    OCR_ZONE_DPI,
    _crop_contains_pii,
    _is_safe_plan_suggestion_label,
)
from app.dataset_pipeline.privacy import scan_text
from app.plan_analysis import schema
from app.plan_analysis.legend_detection import detect_legend_candidates
from app.plan_analysis.legend_entries import LegendEntry, classify_candidate_rows

from .naming import propose_component_name
from .relevance import classify_potable_relevance

RENDER_DPI = 300.0
SYMBOL_CROP_MARGIN_PT = 6.0
CONTEXT_CROP_MARGIN_PT = 40.0


@dataclass
class LegendSymbolRecord:
    legend_symbol_id: str
    plan_id: str
    style_family: Optional[str]
    page: int
    legend_id: str
    bbox: tuple  # DISPLAY space, tight symbol bbox
    symbol_crop_bbox: tuple  # DISPLAY space, actually-rendered symbol crop
    context_crop_bbox: tuple  # DISPLAY space, actually-rendered wider context crop
    raw_legend_text: str
    proposed_component_name: str
    relevance_reason: str
    symbol_png: Optional[np.ndarray] = field(default=None, repr=False)
    context_png: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "legend_symbol_id": self.legend_symbol_id,
            "plan_id_pseudonymous": self.plan_id,
            "style_family": self.style_family,
            "page": self.page,
            "legend_id": self.legend_id,
            "bbox": list(self.bbox),
            "symbol_crop_bbox": list(self.symbol_crop_bbox),
            "context_crop_bbox": list(self.context_crop_bbox),
            "raw_legend_text": self.raw_legend_text,
            "proposed_component_name": self.proposed_component_name,
            "relevance_reason": self.relevance_reason,
            "expert_decision": None,  # "correct" | "corrected" | "unclear"
            "expert_confirmed_name": None,
            "verified_at": None,
        }


@dataclass
class PlanExtractionStats:
    plan_id: str
    has_usable_legend: bool
    legend_entries_total: int = 0
    excluded_not_symbol: int = 0  # TEXT_ONLY/AMBIGUOUS/REJECTED rows, never a real icon
    excluded_privacy: int = 0
    excluded_non_potable: dict = field(default_factory=dict)  # reason -> count
    included: int = 0


def _clip_in_display_space(bbox_display: tuple, page_rect: "pymupdf.Rect", margin: float) -> "pymupdf.Rect":
    """`bbox_display` is ALREADY in display space (a legend_entries.py
    symbol_bbox or row bbox) -- unlike component_facts.py's
    `_display_clip_rect`, no rotation_matrix multiplication happens here,
    exactly matching legend_entries.py's own `_render_gray` convention."""
    x0, y0, x1, y1 = bbox_display
    return pymupdf.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin) & page_rect


def _render_display_crop(page: "pymupdf.Page", clip: "pymupdf.Rect") -> Optional[np.ndarray]:
    if clip.is_empty or clip.width < 1 or clip.height < 1:
        return None
    zoom = RENDER_DPI / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n).copy()


def _ocr_zone_has_pii(page: "pymupdf.Page", clip: "pymupdf.Rect") -> bool:
    """Same mechanism as `candidates._ocr_scan_rect_for_pii`, applied
    directly to this entry's own crop region rather than a page-wide legend
    zone list -- a per-entry, more targeted application of the identical
    round-2 OCR safety net. Fails SAFE (an OCR/render exception counts as
    PII, never silently passed through)."""
    if clip.is_empty:
        return False
    zoom = OCR_ZONE_DPI / 72.0
    try:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        text = pytesseract.image_to_string(arr[:, :, 0], lang="deu+eng", config="--psm 6", timeout=OCR_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 -- fail safe, see docstring
        return True
    return scan_text("_legend_symbol_ocr_check", text).has_pii_risk


def _to_display(bbox: tuple, rotation_matrix) -> tuple:
    if rotation_matrix is None:
        return bbox
    r = pymupdf.Rect(*bbox) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _bbox_overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def extract_plan(plan_id: str, pdf_path: str, doc: schema.DocumentAnalysis, style_family: Optional[str]) -> tuple[list[LegendSymbolRecord], PlanExtractionStats]:
    """`doc` is a v0.1 `analyze_pdf_file(pdf_path)` result on the SAME pdf,
    passed in so the caller parses each plan's PDF exactly once."""
    stats = PlanExtractionStats(plan_id=plan_id, has_usable_legend=False)
    records: list[LegendSymbolRecord] = []

    pdf = pymupdf.open(pdf_path)
    try:
        for page_analysis in doc.pages:
            page = pdf[page_analysis.page_number - 1]
            rotation_matrix = page.rotation_matrix if page.rotation else None
            all_spans_display = [(s, _to_display(s.bbox, rotation_matrix)) for s in page_analysis.text_spans]

            for candidate in detect_legend_candidates(page_analysis, rotation_matrix=rotation_matrix):
                entries, is_symbol_legend = classify_candidate_rows(page, candidate)
                if not is_symbol_legend:
                    continue
                stats.has_usable_legend = True

                for entry in entries:
                    stats.legend_entries_total += 1
                    if entry.classification != "USABLE_TEMPLATE" or entry.is_line_style_swatch or entry.symbol_bbox is None:
                        stats.excluded_not_symbol += 1
                        continue

                    raw_label = entry.label_text
                    if not _is_safe_plan_suggestion_label(raw_label):
                        stats.excluded_privacy += 1
                        continue

                    symbol_clip = _clip_in_display_space(entry.symbol_bbox, page.rect, SYMBOL_CROP_MARGIN_PT)
                    context_bbox = (
                        min(entry.symbol_bbox[0], entry.bbox[0]), min(entry.symbol_bbox[1], entry.bbox[1]),
                        max(entry.symbol_bbox[2], entry.bbox[2]), max(entry.symbol_bbox[3], entry.bbox[3]),
                    )
                    context_clip = _clip_in_display_space(context_bbox, page.rect, CONTEXT_CROP_MARGIN_PT)

                    # Relevance is checked BEFORE the privacy scans (cheap regex
                    # vs. an OCR render+tesseract call per entry): most legend
                    # rows are not potable-water apparatus at all, and there is
                    # no reason to PII-scan a crop this pipeline will discard
                    # regardless of what it contains.
                    relevance = classify_potable_relevance(raw_label)
                    if not relevance["include"]:
                        stats.excluded_non_potable[relevance["reason"]] = stats.excluded_non_potable.get(relevance["reason"], 0) + 1
                        continue

                    if _crop_contains_pii((context_clip.x0, context_clip.y0, context_clip.x1, context_clip.y1), all_spans_display):
                        stats.excluded_privacy += 1
                        continue
                    if _ocr_zone_has_pii(page, context_clip):
                        stats.excluded_privacy += 1
                        continue

                    symbol_png = _render_display_crop(page, symbol_clip)
                    context_png = _render_display_crop(page, context_clip)
                    if symbol_png is None or context_png is None:
                        stats.excluded_privacy += 1  # render failure -- fail safe, don't publish an unrendered entry
                        continue

                    stats.included += 1
                    records.append(LegendSymbolRecord(
                        legend_symbol_id=entry.legend_entry_id,
                        plan_id=plan_id,
                        style_family=style_family,
                        page=page_analysis.page_number,
                        legend_id=candidate.legend_id,
                        bbox=entry.symbol_bbox,
                        symbol_crop_bbox=(symbol_clip.x0, symbol_clip.y0, symbol_clip.x1, symbol_clip.y1),
                        context_crop_bbox=(context_clip.x0, context_clip.y0, context_clip.x1, context_clip.y1),
                        raw_legend_text=raw_label,
                        proposed_component_name=propose_component_name(raw_label),
                        relevance_reason=relevance["reason"],
                        symbol_png=symbol_png,
                        context_png=context_png,
                    ))
    finally:
        pdf.close()

    return records, stats
