"""Text extraction: native PDF text spans, with an OCR fallback for pages
whose "text" is actually vector glyph outlines (a common CAD export quirk,
see vectors.py) or a scanned/rasterized drawing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pymupdf
import pytesseract
from PIL import Image

from .geometry import BBox, bbox_center

# Below this many native text spans, a page is assumed to have no usable
# embedded text and gets OCR'd instead.
NATIVE_TEXT_MIN_SPANS = 10
OCR_DPI = 200
OCR_LANG = "deu+eng"

# Tesseract's default page-segmentation mode (full automatic layout analysis)
# throws away most text on a small-format page that's dense with line art --
# it reads the drawing as "not a text region" and skips it. PSM 6 ("assume a
# single uniform block of text") recovers far more of it. Reading order
# doesn't matter here since every span is placed by its own bounding box
# and matched to geometry spatially, not read as a paragraph.
OCR_PSM = 6

# A small-format page (e.g. a Letter/A4-ish export) rendered at OCR_DPI can
# leave individual characters just a few pixels tall when the drawing is
# dense, which is below what Tesseract needs to recognize them at all. Scale
# DPI up for such pages so the long edge reaches this many pixels; large
# native-CAD-canvas pages already clear this at OCR_DPI, so they're
# untouched (and their runtime doesn't change). MAX_OCR_DPI bounds how far a
# very small page gets scaled, keeping OCR time bounded.
MIN_OCR_LONG_EDGE_PX = 4400
MAX_OCR_DPI = 450

# Rendering the whole page as one raster image is what actually blows the
# memory budget on a large-format plan sheet -- Tesseract's own working
# memory scales with image pixel count, not just the raw buffer size. Tiling
# the page into a grid and OCR'ing (and freeing) one tile at a time caps the
# peak size of any single in-memory image+Tesseract pass to roughly
# total_pixels / (cols*rows), at the *same* DPI (same pixel density, so no
# quality loss). Each tile's render region is expanded by OCR_TILE_OVERLAP_PT
# on its interior edges so a word straddling a grid line is still fully
# contained in at least one tile's image rather than cut in half; the
# resulting duplicate recognition is removed deterministically afterward
# (see _dedupe_ocr_spans). 80pt (~1.1cm) comfortably covers a single
# plan-label word/run at the font sizes these plans use.
OCR_TILE_GRID_COLS = 2
OCR_TILE_GRID_ROWS = 2
OCR_TILE_OVERLAP_PT = 80.0
OCR_TILE_DEDUP_IOU_THRESHOLD = 0.3

# Some source PDFs embed a font with no usable ToUnicode map, so PyMuPDF
# returns the font's raw glyph-index-derived codepoints instead of the
# actual rendered characters -- a systematic substitution cipher, not
# noise, and span *count* alone can't catch it (the page can have plenty
# of "text", all of it wrong). A span counts as garbled when more than
# this fraction of its letters fall outside the alphabet we expect for
# German/technical plan text; a page's native text is discarded once more
# than GARBLED_PAGE_RATIO_THRESHOLD of its spans are garbled.
_ALLOWED_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜßøØ")
GARBLED_SPAN_LETTER_MIN = 3
GARBLED_SPAN_CHAR_THRESHOLD = 0.3
GARBLED_PAGE_RATIO_THRESHOLD = 0.05


@dataclass
class RawTextSpan:
    text: str
    bbox: BBox
    font_size: float | None = None
    confidence: float | None = None


def extract_native_text_spans(page: "pymupdf.Page") -> list[RawTextSpan]:
    page_dict = page.get_text("dict")
    spans: list[RawTextSpan] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                spans.append(
                    RawTextSpan(text=text, bbox=tuple(span["bbox"]), font_size=span.get("size"))
                )
    return spans


def _is_span_garbled(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < GARBLED_SPAN_LETTER_MIN:
        return False
    weird = sum(1 for c in letters if c not in _ALLOWED_LETTERS)
    return (weird / len(letters)) > GARBLED_SPAN_CHAR_THRESHOLD


def native_text_is_garbled(spans: list[RawTextSpan]) -> bool:
    if not spans:
        return False
    garbled = sum(1 for s in spans if _is_span_garbled(s.text))
    return (garbled / len(spans)) > GARBLED_PAGE_RATIO_THRESHOLD


def compute_ocr_dpi(page_width: float, page_height: float, base_dpi: int = OCR_DPI) -> float:
    long_edge_pt = max(page_width, page_height)
    if long_edge_pt <= 0:
        return base_dpi
    base_long_edge_px = long_edge_pt * base_dpi / 72.0
    if base_long_edge_px >= MIN_OCR_LONG_EDGE_PX:
        return base_dpi
    target_dpi = MIN_OCR_LONG_EDGE_PX / (long_edge_pt / 72.0)
    return min(MAX_OCR_DPI, target_dpi)


def _tile_rects(
    page_width: float, page_height: float, cols: int, rows: int, overlap: float
) -> list[tuple[BBox, BBox, int, int]]:
    """Yields (core, clip, row, col) for a cols x rows grid over the page.

    ``core`` rects exactly partition the page (no gaps, no overlap) -- every
    page-space point belongs to exactly one core rect. ``clip`` is ``core``
    expanded by ``overlap`` on interior edges only (clamped to the page
    bounds), used as the actual render region so a word near a grid line is
    fully captured rather than cut off.
    """
    cell_w = page_width / cols
    cell_h = page_height / rows
    tiles = []
    for row in range(rows):
        for col in range(cols):
            cx0, cy0 = col * cell_w, row * cell_h
            cx1 = page_width if col == cols - 1 else cx0 + cell_w
            cy1 = page_height if row == rows - 1 else cy0 + cell_h
            core = (cx0, cy0, cx1, cy1)
            clip = (
                max(0.0, cx0 - overlap) if col > 0 else cx0,
                max(0.0, cy0 - overlap) if row > 0 else cy0,
                min(page_width, cx1 + overlap) if col < cols - 1 else cx1,
                min(page_height, cy1 + overlap) if row < rows - 1 else cy1,
            )
            tiles.append((core, clip, row, col))
    return tiles


def _point_in_core(point: tuple[float, float], core: BBox, row: int, col: int, rows: int, cols: int) -> bool:
    x, y = point
    x0, y0, x1, y1 = core
    x_ok = (x0 <= x < x1) or (col == cols - 1 and x0 <= x <= x1)
    y_ok = (y0 <= y < y1) or (row == rows - 1 and y0 <= y <= y1)
    return x_ok and y_ok


def _bbox_iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_duplicate(candidate: RawTextSpan, accepted: list[RawTextSpan]) -> bool:
    for other in accepted:
        if _bbox_iou(candidate.bbox, other.bbox) < OCR_TILE_DEDUP_IOU_THRESHOLD:
            continue
        if candidate.text == other.text or candidate.text in other.text or other.text in candidate.text:
            return True
    return False


def _dedupe_ocr_spans(owned: list[RawTextSpan], fringe: list[RawTextSpan]) -> list[RawTextSpan]:
    """A word whose true position sits almost exactly on a grid line can have
    its *own* recognized bbox land on either side of that line, independently,
    in each neighboring tile's own crop (sub-pixel rendering/rounding
    differences between two independent renders) -- so two "owned" spans
    from different tiles are not actually guaranteed to be geometrically
    disjoint in practice, only in the idealized exact-math case. Every
    candidate (owned first, then fringe, each in tile scan order) is
    therefore checked against everything already accepted, deterministically:
    the first occurrence of a given word wins, and a later matching
    (bbox-overlapping, text-matching) recognition of the same word from
    another tile is dropped. A fringe span with no matching owned span is
    kept as new information (so a word missed by its own tile but caught via
    a neighbor's overlap margin is never silently lost).
    """
    accepted: list[RawTextSpan] = []
    for span in owned + fringe:
        if not _is_duplicate(span, accepted):
            accepted.append(span)
    return accepted


def _ocr_tile(page: "pymupdf.Page", clip: BBox, zoom: float, lang: str, psm: int) -> list[RawTextSpan]:
    matrix = pymupdf.Matrix(zoom, zoom)
    clip_rect = pymupdf.Rect(*clip)
    pixmap = page.get_pixmap(matrix=matrix, clip=clip_rect, alpha=False)
    # Image.frombytes() copies pixmap.samples into its own buffer, so the
    # source pixmap is redundant from this point on -- release it immediately
    # rather than holding a same-sized second copy through the Tesseract call.
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    del pixmap

    data = pytesseract.image_to_data(
        image, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    del image
    spans: list[RawTextSpan] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        raw_conf = data["conf"][i]
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and confidence < 0:
            confidence = None
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # Offset by the tile's own clip origin to map back into page space.
        bbox = (
            clip[0] + x / zoom,
            clip[1] + y / zoom,
            clip[0] + (x + w) / zoom,
            clip[1] + (y + h) / zoom,
        )
        spans.append(RawTextSpan(text=text, bbox=bbox, confidence=confidence))
    del data
    return spans


def ocr_page(
    page: "pymupdf.Page", dpi: float = OCR_DPI, lang: str = OCR_LANG, psm: int = OCR_PSM
) -> list[RawTextSpan]:
    """Renders and OCRs the page in a grid of overlapping tiles, one at a
    time, at the exact same pixel density (DPI) as a single full-page
    render -- only the memory footprint of any one render+recognize pass is
    reduced, not the resolution Tesseract sees. See module-level
    OCR_TILE_* constants and _dedupe_ocr_spans for how boundary words are
    kept intact exactly once.
    """
    zoom = dpi / 72.0
    page_width, page_height = page.rect.width, page.rect.height
    cols, rows = OCR_TILE_GRID_COLS, OCR_TILE_GRID_ROWS

    owned: list[RawTextSpan] = []
    fringe: list[RawTextSpan] = []
    for core, clip, row, col in _tile_rects(page_width, page_height, cols, rows, OCR_TILE_OVERLAP_PT):
        for span in _ocr_tile(page, clip, zoom, lang, psm):
            if _point_in_core(bbox_center(span.bbox), core, row, col, rows, cols):
                owned.append(span)
            else:
                fringe.append(span)

    return _dedupe_ocr_spans(owned, fringe)
