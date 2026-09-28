"""Legend row extraction -- v0.3, new (Phase 2). Splits a detected legend
region (legend_detection.LegendCandidate) into individual entries, each
pairing a candidate SYMBOL graphic with its LABEL text -- or reporting
honestly that no usable symbol graphic could be isolated.

"Do not assume every row is usable" (explicit task requirement): every row
is classified, never silently kept or dropped:

- USABLE_TEMPLATE  -- a symbol-sized ink blob was found immediately next to
  the label, shaped like an icon (not a border/gridline), and the row is a
  single icon+label pair (not several merged together).
- TEXT_ONLY        -- no ink found next to the label (a numeric-table row,
  a heading, or a purely textual note).
- AMBIGUOUS        -- more than one label fragment sits inside what
  clustered as a single row (a wide, uneven gap between text spans),
  so the symbol/label pairing itself cannot be trusted.
- REJECTED         -- ink was found, but its shape looks like a table
  border/gridline rather than an icon (see `_looks_like_border`), OR the
  row belongs to a legend candidate whose usable-icon ratio is too low to
  be a real symbol legend at all (a lookalike bordered/numeric table, e.g.
  W-003's "Legende Verteilbatterie" -- see `classify_candidate_rows`).

WHERE THE SYMBOL ZONE COMES FROM: v0.1's text extraction only captures
TEXT -- the drawn icon graphic sitting next to a legend label is vector
ink, not text, so it is never part of any TextSpan bbox. This module looks
for it directly: render the raster region immediately to the LEFT of a
row's own text (the direction every observed W-003 legend entry places its
icon), the same way v0.2's component_facts.py already renders symbol
candidates from their vector-detected bboxes -- reusing
symbol_library.canonicalize() to test for and normalize any ink found
there, rather than adding a second raster pipeline.

This module invents no label text and no symbol identity: a legend row's
own text IS its label (native or OCR, exactly as v0.1 extracted it), and a
symbol's identity is never guessed here -- Phase 3/4/5 do that by matching
the extracted graphic, never this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

import cv2
import numpy as np
import pymupdf

from .legend_detection import LegendCandidate, LegendRow
from .symbol_library import canonicalize

# How far to the left of a row's own label text to look for its icon
# graphic (DISPLAY-space pt). Measured on W-003's real "LEGENDE SANITAER"
# block: the legend's own left margin sits ~52pt left of column 1's label
# text start (42.5 vs 94.4) -- the icon lives in exactly that gap. Used as
# a fixed lookback rather than per-column-derived because the true left
# edge of an icon zone is graphical (not text), so it cannot be measured
# from text bboxes alone without first assuming the answer.
SYMBOL_LOOKBACK_PT = 58.0
ROW_VERTICAL_PAD_PT = 1.5
# A gap this large between two text spans inside one already-clustered row
# means the row-clustering merged two separate icon+label pairs that only
# happened to land at the same display-y (e.g. two short labels in
# adjacent sub-columns) -- their symbol/label pairing can't be trusted.
MULTI_LABEL_GAP_PT = 45.0
# Ink below this many "on" pixels in the canonicalized 128x128 mask is
# treated as scan/render noise, not a real icon.
INK_MIN_PIXELS = 12
# A candidate legend where fewer than this fraction of its non-trivial rows
# produced a usable icon is almost certainly not a real symbol legend at
# all (a bordered numeric table, a purely textual note block, ...).
CANDIDATE_USABLE_RATIO_MIN = 0.15
# A row shaped "<short code> = <description>" (e.g. "a = Schraegsitzventil
# 5/4\"") is a coded LOOKUP TABLE entry, not a symbol-legend entry -- this is
# exactly W-003's own "Legende Verteilbatterie" ("a = ...", "b = ...", ...),
# a real, plan-labelled "Legende" that has no symbol column at all. Ink
# picked up in such a row's lookback zone is unreliable evidence here
# specifically because these tables are often drawn right next to (or
# overlapping) the real installation drawing, unlike an isolated legend
# block on blank paper -- so text SHAPE, not ink presence, is what decides
# this case. A candidate dominated by this pattern is rejected outright,
# regardless of any ink its lookback zones happened to catch.
CODE_ASSIGNMENT_PATTERN = re.compile(r"^\s*[A-Za-z0-9]{1,3}\s*=\s*\S")
CODE_ASSIGNMENT_RATIO_MAX = 0.3

RENDER_DPI = 300


def _stable_id(*parts: str) -> str:
    return "LE" + sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _normalize_label(text: str) -> str:
    # W-003's own extracted text can carry stray inter-glyph spaces from
    # font/kerning quirks (observed on section headings, e.g.
    # "Dä mmu ng sl eg e nd e") -- collapsing whitespace can't undo that,
    # so normalization here only does what's safe: casefold + collapse runs
    # of whitespace + strip. It does not attempt to "fix" glyph spacing.
    return re.sub(r"\s+", " ", text).strip().casefold()


@dataclass
class LegendEntry:
    legend_entry_id: str
    legend_id: str
    page: int
    bbox: tuple  # DISPLAY-space, label + symbol zone
    symbol_bbox: Optional[tuple]
    label_text: str
    normalized_label: str
    label_source: str  # "native" | "ocr" | "mixed" | "none"
    classification: str  # USABLE_TEMPLATE | TEXT_ONLY | AMBIGUOUS | REJECTED
    ambiguity_reason: Optional[str]
    symbol_ink_present: bool
    is_line_style_swatch: bool = False
    symbol_canonical: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "legend_entry_id": self.legend_entry_id,
            "legend_id": self.legend_id,
            "page": self.page,
            "bbox": list(self.bbox),
            "symbol_bbox": list(self.symbol_bbox) if self.symbol_bbox else None,
            "label_text": self.label_text,
            "normalized_label": self.normalized_label,
            "label_source": self.label_source,
            "classification": self.classification,
            "ambiguity_reason": self.ambiguity_reason,
            "symbol_ink_present": self.symbol_ink_present,
            "is_line_style_swatch": self.is_line_style_swatch,
        }


def _label_source(row: LegendRow) -> str:
    sources = {span.source for span, _ in row.spans}
    if not sources:
        return "none"
    if len(sources) == 1:
        return next(iter(sources))
    return "mixed"


def _internal_gap_reason(row: LegendRow) -> Optional[str]:
    boxes = sorted((dbbox for _, dbbox in row.spans), key=lambda b: b[0])
    for a, b in zip(boxes, boxes[1:]):
        if b[0] - a[2] > MULTI_LABEL_GAP_PT:
            return f"internal text gap {b[0] - a[2]:.1f}pt exceeds {MULTI_LABEL_GAP_PT}pt -- likely merged entries"
    return None


def _render_gray(page: "pymupdf.Page", display_bbox: tuple) -> Optional[np.ndarray]:
    """display_bbox is already in the SAME frame as page.rect (DISPLAY
    space) -- unlike component_facts.py's v0.1-bbox renderer, no
    rotation_matrix multiplication is needed here, since legend_detection.py
    already transformed every row bbox into display space itself."""
    clip = pymupdf.Rect(*display_bbox) & page.rect
    if clip.is_empty or clip.width < 1 or clip.height < 1:
        return None
    zoom = RENDER_DPI / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n >= 3:
        return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    return arr[:, :, 0]


def _looks_like_border(gray: np.ndarray) -> bool:
    """A table border/gridline renders as a thin band of ink running the
    FULL width or FULL height of the zone -- unlike a real icon, which
    (by legend-drawing convention, and confirmed on W-003) sits with
    whitespace margin on every side of its own cell. Distinguishing the two
    from shape alone avoids ever training/matching on a table's own
    ruling as if it were a component."""
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    h, w = gray.shape
    ink_w = xs.max() - xs.min() + 1
    ink_h = ys.max() - ys.min() + 1
    spans_full_width = ink_w >= 0.92 * w
    spans_full_height = ink_h >= 0.92 * h
    thin_relative_to_span = (ink_h <= 0.12 * h) or (ink_w <= 0.12 * w)
    return bool((spans_full_width or spans_full_height) and thin_relative_to_span)


def _looks_like_line_style_swatch(gray: np.ndarray) -> bool:
    """A pipe-type line-style swatch (solid/dashed/dotted/colored line
    sample, e.g. W-003's "Warmwasser"/"Kaltwasser" entries) is dangerous as
    a Phase 4 search template: it would match countless unrelated straight
    pipe runs across the drawing on shape alone, the opposite of
    "precision before recall" -- so it must be flagged and excluded even
    though it IS genuine, usable, document-local evidence otherwise. Two
    complementary checks, both measured against W-003's real 37 usable
    entries:
    1. Ink only a handful of pixels TALL overall (a plain solid/colored
       line) -- no real icon body was observed under ~13px tall.
    2. A DASHED/DOTTED line (e.g. "Schmutzabwasser", "Regenabwasser") is
       taller per (1) alone, since dash marks poke slightly above/below the
       baseline, but still reads as many small, disconnected ink islands
       strung out along a thin horizontal band -- unlike a real icon, which
       is a small number of solid, often-touching strokes/blobs even when
       multi-part (confirmed on W-003: multi-stroke icons like "Filter"
       stay under this height/ink ratio; only true dash patterns clear
       both bars at once)."""
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return False
    ink_h = ys.max() - ys.min() + 1
    ink_w = xs.max() - xs.min() + 1
    if ink_h <= 4:
        return True
    n_components, labels = cv2.connectedComponents(mask)
    island_count = sum(1 for i in range(1, n_components) if np.sum(labels == i) >= 3)
    return bool(island_count >= 6 and (ink_h / ink_w) <= 0.15)


def _extract_symbol(page: "pymupdf.Page", row: LegendRow, legend_bbox: tuple) -> tuple:
    """Returns (symbol_bbox_or_None, ink_present, canonical_or_None, is_border, is_line_swatch)."""
    x0, y0, x1, y1 = row.bbox
    zone_x0 = max(legend_bbox[0], x0 - SYMBOL_LOOKBACK_PT)
    zone = (zone_x0, y0 - ROW_VERTICAL_PAD_PT, x0, y1 + ROW_VERTICAL_PAD_PT)
    if zone[2] - zone[0] < 3:
        return None, False, None, False, False
    gray = _render_gray(page, zone)
    if gray is None:
        return None, False, None, False, False
    canon = canonicalize(gray)
    if canon is None or int((canon > 0).sum()) < INK_MIN_PIXELS:
        return zone, False, None, False, False
    if _looks_like_border(gray):
        return zone, True, canon, True, False
    return zone, True, canon, False, _looks_like_line_style_swatch(gray)


def extract_entries(page: "pymupdf.Page", candidate: LegendCandidate) -> list[LegendEntry]:
    """Pure-per-candidate extraction; does not decide whether the candidate
    AS A WHOLE is a real symbol legend -- see `classify_candidate_rows`,
    which wraps this and applies the whole-candidate REJECTED rule."""
    entries: list[LegendEntry] = []
    for idx, row in enumerate(candidate.rows or []):
        label_text = row.text
        normalized = _normalize_label(label_text)
        source = _label_source(row)
        ambiguity_reason = _internal_gap_reason(row)

        if ambiguity_reason is not None:
            entries.append(LegendEntry(
                legend_entry_id=_stable_id(candidate.legend_id, str(idx), label_text),
                legend_id=candidate.legend_id, page=candidate.page,
                bbox=row.bbox, symbol_bbox=None, label_text=label_text,
                normalized_label=normalized, label_source=source,
                classification="AMBIGUOUS", ambiguity_reason=ambiguity_reason,
                symbol_ink_present=False,
            ))
            continue

        symbol_bbox, ink_present, canon, is_border, is_line_swatch = _extract_symbol(page, row, candidate.bbox)
        full_bbox = row.bbox if symbol_bbox is None else (
            min(symbol_bbox[0], row.bbox[0]), min(symbol_bbox[1], row.bbox[1]),
            max(symbol_bbox[2], row.bbox[2]), max(symbol_bbox[3], row.bbox[3]),
        )

        if not normalized:
            classification, reason = "TEXT_ONLY", "empty label text"
        elif is_border:
            classification, reason = "REJECTED", "symbol zone ink shaped like a table border/gridline, not an icon"
        elif ink_present:
            classification = "USABLE_TEMPLATE"
            reason = "line-style/pipe-type swatch -- kept as evidence, excluded from Phase 4 search templates" if is_line_swatch else None
        else:
            classification, reason = "TEXT_ONLY", "no ink found in symbol zone"

        entries.append(LegendEntry(
            legend_entry_id=_stable_id(candidate.legend_id, str(idx), label_text),
            legend_id=candidate.legend_id, page=candidate.page,
            bbox=full_bbox, symbol_bbox=symbol_bbox, label_text=label_text,
            normalized_label=normalized, label_source=source,
            classification=classification, ambiguity_reason=reason,
            symbol_ink_present=ink_present, is_line_style_swatch=is_line_swatch,
            symbol_canonical=canon if classification == "USABLE_TEMPLATE" else None,
        ))
    return entries


def classify_candidate_rows(page: "pymupdf.Page", candidate: LegendCandidate) -> tuple[list[LegendEntry], bool]:
    """Extracts entries, then applies the whole-candidate rule: a candidate
    where too few of its real (non-empty-label) rows produced a usable icon
    is not a real symbol legend -- e.g. W-003's "Legende Verteilbatterie",
    a bordered NUMERIC table with no symbol column at all. In that case
    every entry is downgraded to REJECTED, regardless of its own per-row
    result, since the whole candidate's premise (this is a symbol legend)
    failed. Returns (entries, is_symbol_legend)."""
    entries = extract_entries(page, candidate)
    real_rows = [e for e in entries if e.normalized_label]
    usable = [e for e in real_rows if e.classification == "USABLE_TEMPLATE"]
    ratio = (len(usable) / len(real_rows)) if real_rows else 0.0

    code_rows = [e for e in real_rows if CODE_ASSIGNMENT_PATTERN.match(e.label_text)]
    code_ratio = (len(code_rows) / len(real_rows)) if real_rows else 0.0
    is_coded_lookup_table = code_ratio > CODE_ASSIGNMENT_RATIO_MAX

    is_symbol_legend = (ratio >= CANDIDATE_USABLE_RATIO_MIN) and not is_coded_lookup_table

    if not is_symbol_legend:
        if is_coded_lookup_table:
            reason = (
                f"candidate-level: {len(code_rows)}/{len(real_rows)} rows "
                f"({code_ratio:.0%}) match a '<code> = <description>' coded "
                f"lookup-table pattern -- a real legend structurally, but with "
                f"no symbol column to extract templates from"
            )
        else:
            reason = (
                f"candidate-level: only {len(usable)}/{len(real_rows)} rows "
                f"({ratio:.0%}) produced a usable icon -- below "
                f"{CANDIDATE_USABLE_RATIO_MIN:.0%}, treated as a non-symbol "
                f"table/text block rather than a real legend"
            )
        for e in entries:
            e.classification = "REJECTED"
            e.ambiguity_reason = reason
            e.symbol_canonical = None
    return entries, is_symbol_legend
