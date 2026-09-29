"""Per-plan legend extraction: detect the plan's own symbol legend, classify
its rows, and build a PLAN-SPECIFIC catalog of (legend entry -> structural
fingerprint) -- never a raster template.

Reuse boundary (see the investigation this benchmark is built on):

- `app.plan_analysis.legend_detection.detect_legend_candidates` is reused
  AS-IS. It is pure text-geometry (row/column clustering + optional heading
  match) -- it never touches a pixel, so reusing it is not "the raster
  approach that didn't work" (that was v0.3's `symbol_templates.py` +
  `legend_intelligence.py`'s `cv2.matchTemplate` matching stage, which this
  benchmark never imports).
- `app.plan_analysis.legend_entries.classify_candidate_rows` is also reused
  AS-IS, but ONLY for what it decides, not for what it renders: it tells us
  which rows are a real single icon+label pair (`USABLE_TEMPLATE`) versus a
  merged/ambiguous row, a text-only/coded-lookup row, or ink shaped like a
  table border rather than an icon (`REJECTED`) -- genuinely useful,
  already-tested geometric/shape judgment about the LEGEND ROW's own
  structure. What this benchmark explicitly does NOT reuse from that
  function's result is `symbol_canonical` (the canonicalized raster mask) --
  it is discarded; only `symbol_bbox` (a location) and `classification`/
  `is_line_style_swatch` (row-quality judgments) are read. The symbol's
  actual IDENTITY is decided purely from `vector_features.py`'s own
  independent geometric read of that same bbox, computed fresh here.

`component_type` assignment is a deterministic keyword mapping against the
row's own normalized label text onto this dataset's existing 10-class
taxonomy (`data/dev_plans_v04/annotation_web_export/classes.json`) -- never
a model, never tuned against any human label (it was written by reading the
CLASS NAMES only, before this benchmark's Phase A ever ran).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from app.plan_analysis import schema
from app.plan_analysis.legend_detection import LegendCandidate, detect_legend_candidates
from app.plan_analysis.legend_entries import LegendEntry, classify_candidate_rows

from .vector_features import StructuralFingerprint, build_page_index, compute_structural_fingerprint

# Ordered, first-match-wins keyword rules against the row's own
# `normalized_label` (casefolded, whitespace-collapsed). Word-boundary
# regexes throughout so e.g. "ba" never matches inside "abstellventil".
_KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdusch(e|brause)?\b"), "dusche"),
    (re.compile(r"\bk(ü|ue)che\b|\bsp(ü|ue)le\b"), "küche"),
    (re.compile(r"\bwc\b|\bklosett\b|\bcloset\b"), "wc up"),
    (re.compile(r"\bsecomat\b"), "secomat"),
    (re.compile(r"\bverteiler\b.*\bunter\b.*\bw(t|aschtisch)\b|\bunter\b.*\bw(t|aschtisch)\b.*\bverteiler\b"), "up verteiler unter wt"),
    (re.compile(r"\bpex\b.*\bverteiler\b|\bverteiler\b.*\bpex\b"), "pex-verteiler"),
    (re.compile(r"^\s*ba\s*$|\bba\b"), "BA"),
]

# Labels naming a MEDIUM/line-network system rather than a discrete
# component -- these are real, legitimate legend rows, just never
# components (matches the same "wastewater/medium legend != a component"
# distinction the v0.4 dataset cleanup already applied to candidate
# labels). A row matching this is NOT_A_COMPONENT regardless of whether it
# happens to also produce ink in its lookback zone.
_MEDIUM_LEGEND_PATTERN = re.compile(
    r"schmutzwasser|abwasser|regenwasser|entw(ä|ae)sserung|"
    r"kaltwasser|warmwasser|zirkulation|heizung|l(ü|ue)ftung|erdung|"
    r"fallstrang|fallleitung|steigleitung"
)


def classify_component_type(normalized_label: str, is_line_style_swatch: bool) -> Optional[str]:
    """Pure keyword mapping onto the existing 10-class taxonomy. Returns
    None when the label gives no confident signal either way (component
    identity is then left to the structural/graph matching evidence, never
    guessed from text alone)."""
    if is_line_style_swatch or _MEDIUM_LEGEND_PATTERN.search(normalized_label):
        return "NOT_A_COMPONENT"
    for pattern, component_type in _KEYWORD_RULES:
        if pattern.search(normalized_label):
            return component_type
    return None


@dataclass
class LegendEntryRecord:
    legend_entry_id: str
    symbol_bbox: Optional[tuple]  # DISPLAY space -- for the human-readable legend table
    raw_label: str
    normalized_label: str
    component_type: Optional[str]
    usable: bool
    reason: Optional[str]
    fingerprint: Optional[StructuralFingerprint] = field(default=None, repr=False, compare=False)
    symbol_bbox_raw: Optional[tuple] = None  # PDF-native space -- what matching.py compares against

    def to_dict(self) -> dict:
        return {
            "legend_entry_id": self.legend_entry_id,
            "symbol_bbox": list(self.symbol_bbox) if self.symbol_bbox else None,
            "raw_label": self.raw_label,
            "normalized_label": self.normalized_label,
            "component_type": self.component_type,
            "usable": self.usable,
            "reason": self.reason,
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
        }


@dataclass
class PlanLegend:
    plan_id: str
    page: Optional[int]
    legend_id: Optional[str]
    confidence: float
    entries: list[LegendEntryRecord]
    legend_bbox: Optional[tuple] = None  # DISPLAY space
    legend_bbox_raw: Optional[tuple] = None  # PDF-native space -- what matching.py excludes against

    def usable_catalog(self) -> dict[str, LegendEntryRecord]:
        return {e.legend_entry_id: e for e in self.entries if e.usable}

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "page": self.page,
            "legend_id": self.legend_id,
            "confidence": round(self.confidence, 4),
            "legend_bbox": list(self.legend_bbox) if self.legend_bbox else None,
            "entries": [e.to_dict() for e in self.entries],
        }


def _entry_reason(entry: LegendEntry, component_type: Optional[str]) -> Optional[str]:
    if entry.classification == "USABLE_TEMPLATE":
        if entry.is_line_style_swatch:
            return "line-style/medium swatch, not a discrete component icon"
        if component_type is None:
            return "usable icon, but its label text matched no known component keyword (kept as OTHER_RELEVANT_SYMBOL)"
        return None
    return entry.ambiguity_reason


def _entry_usable(entry: LegendEntry) -> bool:
    return entry.classification == "USABLE_TEMPLATE" and not entry.is_line_style_swatch


def extract_plan_legend(plan_id: str, pdf_path: str, doc_pages: list[schema.PageAnalysis]) -> PlanLegend:
    """`doc_pages` is `DocumentAnalysis.pages` from a v0.1
    `analyze_pdf_file(pdf_path)` run on the SAME pdf -- passed in rather
    than re-parsed here so the caller (Phase A) parses each plan's PDF
    exactly once."""
    pdf = pymupdf.open(pdf_path)
    try:
        best: Optional[tuple[int, LegendCandidate, list[LegendEntry]]] = None
        for page_analysis in doc_pages:
            page = pdf[page_analysis.page_number - 1]
            rotation_matrix = page.rotation_matrix if page.rotation else None
            for candidate in detect_legend_candidates(page_analysis, rotation_matrix=rotation_matrix):
                entries, is_symbol_legend = classify_candidate_rows(page, candidate)
                if not is_symbol_legend:
                    continue
                usable_count = sum(1 for e in entries if _entry_usable(e))
                if best is None or (candidate.confidence, usable_count) > (best[1].confidence, sum(1 for e in best[2] if _entry_usable(e))):
                    best = (page_analysis.page_number, candidate, entries)

        if best is None:
            return PlanLegend(plan_id=plan_id, page=None, legend_id=None, confidence=0.0, entries=[])

        page_number, candidate, entries = best
        page = pdf[page_number - 1]
        rotation_matrix = page.rotation_matrix if page.rotation else None
        inverse_rotation_matrix = ~rotation_matrix if rotation_matrix else None
        drawings = page.get_drawings()
        index = build_page_index(drawings, page.rect.width, page.rect.height)

        # `legend_detection`/`legend_entries` work entirely in DISPLAY space
        # (they must, to line up with how a human reads a rotated page --
        # see legend_detection.py's own module docstring), but this
        # benchmark's `PageVectorIndex` is built from `page.get_drawings()`,
        # which PyMuPDF always returns in the page's RAW, pre-rotation
        # coordinate frame regardless of its /Rotate value -- the exact same
        # display-vs-raw distinction `app/dataset_pipeline/candidates.py`
        # already had to handle for crop bboxes. For an unrotated page
        # (rotation_matrix is None) the two frames are identical and this is
        # a no-op; for a rotated one (true for DEV-03 in this benchmark's
        # own 5-plan selection -- rotation 90) skipping this conversion
        # silently fingerprints the WRONG region of the page for every
        # legend entry, which is exactly the bug this comment replaced.
        def _to_raw(bbox: tuple) -> tuple:
            if inverse_rotation_matrix is None:
                return bbox
            r = pymupdf.Rect(*bbox) * inverse_rotation_matrix
            return (r.x0, r.y0, r.x1, r.y1)

        records: list[LegendEntryRecord] = []
        for entry in entries:
            usable = _entry_usable(entry)
            component_type = classify_component_type(entry.normalized_label, entry.is_line_style_swatch) if entry.normalized_label else None
            if entry.classification == "AMBIGUOUS":
                component_type = "AMBIGUOUS"
            elif not usable and component_type is None and entry.normalized_label:
                component_type = "NOT_A_COMPONENT"
            fingerprint = None
            raw_symbol_bbox = _to_raw(entry.symbol_bbox) if entry.symbol_bbox is not None else None
            if usable and raw_symbol_bbox is not None:
                fingerprint = compute_structural_fingerprint(index, raw_symbol_bbox)
                if fingerprint.total_item_count == 0:
                    usable = False  # ink was detected by legend_entries' raster check, but our
                    # own independent vector read of the same zone found no line/curve/rect/quad
                    # items at all (can happen for very faint anti-aliased renders) -- honest to
                    # exclude rather than match on an empty fingerprint.
            if usable and component_type is None:
                component_type = "OTHER_RELEVANT_SYMBOL"  # a real, usable icon whose label text
                # matched no narrower keyword -- still a genuine component for matching purposes.
            records.append(
                LegendEntryRecord(
                    legend_entry_id=entry.legend_entry_id,
                    symbol_bbox=entry.symbol_bbox,
                    raw_label=entry.label_text,
                    normalized_label=entry.normalized_label,
                    component_type=component_type,
                    usable=usable,
                    reason=_entry_reason(entry, component_type) if usable else (entry.ambiguity_reason or "no usable icon"),
                    fingerprint=fingerprint,
                    symbol_bbox_raw=raw_symbol_bbox,
                )
            )

        return PlanLegend(
            plan_id=plan_id,
            page=page_number,
            legend_id=candidate.legend_id,
            confidence=candidate.confidence,
            entries=records,
            legend_bbox=candidate.bbox,
            legend_bbox_raw=_to_raw(candidate.bbox),
        )
    finally:
        pdf.close()
