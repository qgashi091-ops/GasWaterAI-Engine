"""Searches the ENTIRE drawing area of one page for occurrences of a
`SymbolTemplate`'s own vector geometry. No candidate generator and no text
matching are used to propose a region -- every candidate window comes
purely from a coarse geometric scan of the page's own vector primitives.

Two-pass per window, so the expensive geometry read only ever runs on
windows that already look plausible:
  Pass 1 (cheap): count primitives overlapping the window via the existing
    `PageVectorIndex.query` -- skip immediately if the count isn't within a
    generous multiple of the template's own item count.
  Pass 2 (full): compute the window's own `StructuralFingerprint`, require
    it to pass the SAME `graphical_symbol_gate` the template itself had to
    pass (this is what keeps text, bare pipe runs, dimension lines and
    table borders out -- they simply cannot pass that gate), then score it
    against the template with `structural_similarity`.

Overlapping high-scoring windows for the same underlying occurrence are
merged by greedy non-max suppression on score.

Corroborating evidence (nearby text, pipe-boundary crossings) is computed
ONLY for windows that already passed the geometric match -- it is never
part of the match decision itself, per the experiment's own rule that text
and the network graph may corroborate a match but never create one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from app.legend_structural_benchmark.vector_features import (
    PageVectorIndex,
    compute_structural_fingerprint,
    structural_similarity,
)
from app.legend_symbol_dataset.graphical_symbol_gate import (
    compute_text_area_ratio,
    evaluate_graphical_symbol,
)

from .template import SymbolTemplate

BBox = tuple[float, float, float, float]

SIMILARITY_THRESHOLD = 0.72
SCALES = (1.0, 1.3)
STRIDE_FRACTION = 0.6
MIN_WINDOW = 8.0
NMS_CENTER_DISTANCE_FRACTION = 0.6  # of the larger of the two windows' size


@dataclass
class Occurrence:
    bbox: BBox
    scale: float
    similarity: dict
    fingerprint: dict
    nearby_text: list = field(default_factory=list)
    boundary_crossing_count: int = 0

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "scale": self.scale,
            "similarity": self.similarity,
            "fingerprint": self.fingerprint,
            "nearby_text": self.nearby_text,
            "boundary_crossing_pipe_count": self.boundary_crossing_count,
        }


def _center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _nearby_text(text_spans_native: list, bbox: BBox, margin: float) -> list[str]:
    x0, y0, x1, y1 = bbox
    zone = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    out = []
    for span in text_spans_native:
        sb = span.bbox if hasattr(span, "bbox") else span
        if sb[0] < zone[2] and zone[0] < sb[2] and sb[1] < zone[3] and zone[1] < sb[3]:
            text = getattr(span, "text", None)
            if text:
                out.append(text)
    return out


def _boundary_crossings(index: PageVectorIndex, bbox: BBox, tol: float = 1.5) -> int:
    """Counts line-like items whose CLIPPED segment (see
    `PageVectorIndex.query`) has an endpoint within `tol` of the window's
    own boundary -- i.e. a pipe line entering or leaving the matched
    region, exactly the "port geometry" this experiment reports as
    corroborating evidence, never as part of the match itself."""
    x0, y0, x1, y1 = bbox
    count = 0
    for op, p0, p1, _is_fill in index.query(bbox, min_clipped_length=0.5, area_overlap_ratio_min=0.3):
        if op not in ("l", "qu") or p0 is None:
            continue
        for p in (p0, p1):
            on_boundary = (
                abs(p[0] - x0) <= tol or abs(p[0] - x1) <= tol or abs(p[1] - y0) <= tol or abs(p[1] - y1) <= tol
            )
            if on_boundary:
                count += 1
                break
    return count


def search_page_for_templates(
    index: PageVectorIndex,
    page_rect: "pymupdf.Rect",
    templates: list[SymbolTemplate],
    text_spans_native: list,
) -> dict[str, list[Occurrence]]:
    """Shares the expensive pass-1 grid walk across every template whose
    core size falls in the same scale bucket (in this experiment all 5
    DEV-03 templates have a near-identical ~13.7-13.8pt core, so this
    collapses what would otherwise be 5 independent full-page scans into
    one) -- a page is scanned once per distinct window size, and every
    surviving window is then scored against every template that size was
    computed for. This is an efficiency measure only; it changes nothing
    about what counts as a match (each template still gets its own
    independent gate + similarity decision in pass 2)."""
    by_window_size: dict[float, list[SymbolTemplate]] = {}
    for t in templates:
        core = min(t.width, t.height)
        for scale in SCALES:
            win = round(max(MIN_WINDOW, core * scale), 3)
            by_window_size.setdefault(win, []).append(t)

    # window -> list of (bbox,) candidates passing the cheap item-count
    # prefilter for AT LEAST ONE of the templates sharing that window size.
    candidates_by_window: dict[float, list[BBox]] = {}
    for win, group_templates in by_window_size.items():
        stride = max(win * STRIDE_FRACTION, 4.0)
        item_ranges = [
            (max(2, int(t.fingerprint.total_item_count * 0.4)), max(6, int(t.fingerprint.total_item_count * 2.5)))
            for t in group_templates
        ]
        lo, hi = min(r[0] for r in item_ranges), max(r[1] for r in item_ranges)
        found: list[BBox] = []
        x = page_rect.x0
        while x < page_rect.x1:
            y = page_rect.y0
            while y < page_rect.y1:
                bbox = (x, y, min(x + win, page_rect.x1), min(y + win, page_rect.y1))
                n = len(index.query(bbox, min_clipped_length=0.5, area_overlap_ratio_min=0.3))
                if lo <= n <= hi:
                    found.append(bbox)
                y += stride
            x += stride
        candidates_by_window[win] = found

    results: dict[str, list[Occurrence]] = {t.legend_symbol_id: [] for t in templates}
    fingerprint_cache: dict[BBox, tuple] = {}
    for t in templates:
        core = min(t.width, t.height)
        scored: list[Occurrence] = []
        for scale in SCALES:
            win = round(max(MIN_WINDOW, core * scale), 3)
            for bbox in candidates_by_window.get(win, ()):
                if bbox not in fingerprint_cache:
                    fp = compute_structural_fingerprint(index, bbox)
                    text_ratio = compute_text_area_ratio(bbox, text_spans_native)
                    gate = evaluate_graphical_symbol(fp, text_ratio)
                    fingerprint_cache[bbox] = (fp, gate.passes)
                fp, passes = fingerprint_cache[bbox]
                if not passes:
                    continue
                sim = structural_similarity(fp, t.fingerprint)
                if sim["structural_score"] < SIMILARITY_THRESHOLD:
                    continue
                scored.append(Occurrence(bbox=bbox, scale=scale, similarity=sim, fingerprint=fp.to_dict()))

        scored.sort(key=lambda o: o.similarity["structural_score"], reverse=True)
        kept: list[Occurrence] = []
        for occ in scored:
            c = _center(occ.bbox)
            w = max(occ.bbox[2] - occ.bbox[0], occ.bbox[3] - occ.bbox[1])
            if any(
                _dist(c, _center(k.bbox)) <= NMS_CENTER_DISTANCE_FRACTION * max(w, max(k.bbox[2] - k.bbox[0], k.bbox[3] - k.bbox[1]))
                for k in kept
            ):
                continue
            kept.append(occ)

        margin = core * 3.0
        for occ in kept:
            occ.nearby_text = _nearby_text(text_spans_native, occ.bbox, margin)
            occ.boundary_crossing_count = _boundary_crossings(index, occ.bbox)
        kept.sort(key=lambda o: o.similarity["structural_score"], reverse=True)
        results[t.legend_symbol_id] = kept

    return results
