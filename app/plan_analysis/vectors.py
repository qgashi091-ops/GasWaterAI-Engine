"""Classifies raw PyMuPDF vector drawings into three buckets:

- ``line_segments``: straight strokes that make up the pipe network.
- ``glyph_primitives``: tiny filled outlines that are actually text which
  the source CAD export "burned" into curves instead of real font glyphs
  (very common for AutoCAD/Revit PDF exports). These feed the OCR fallback
  rather than the graph/symbol detectors.
- ``symbol_primitives``: everything else (circles, arcs, hatching, filled
  icons) — the raw material for symbol-candidate clustering.

Classification is heuristic and relative to page size, because plan pages
in this pipeline range from small A4 exports to huge native CAD canvases.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .geometry import BBox, Point, bbox_max_dim, dist


@dataclass
class LineSegment:
    p0: Point
    p1: Point
    color_rgb: tuple[float, float, float] | None
    stroke_width: float | None

    @property
    def bbox(self) -> BBox:
        return (
            min(self.p0[0], self.p1[0]),
            min(self.p0[1], self.p1[1]),
            max(self.p0[0], self.p1[0]),
            max(self.p0[1], self.p1[1]),
        )


@dataclass
class VectorPrimitive:
    bbox: BBox
    has_curves: bool
    item_count: int
    is_fill: bool


@dataclass
class PageVectors:
    line_segments: list[LineSegment] = field(default_factory=list)
    background_line_segments: list[LineSegment] = field(default_factory=list)
    glyph_primitives: list[VectorPrimitive] = field(default_factory=list)
    symbol_primitives: list[VectorPrimitive] = field(default_factory=list)


# Fractions of the page diagonal used to tell a text glyph apart from a
# small pipe fitting symbol, and a symbol apart from a long pipe run.
GLYPH_MAX_DIAGONAL_FRACTION = 0.012
SYMBOL_MAX_DIAGONAL_FRACTION = 0.08

# --- Background-line detection -------------------------------------------
#
# Plan PDFs mix real pipe strokes with architectural reference geometry
# (walls, gridlines, dimension lines, CAD "construction lines" kept in the
# export at near-invisible colors) drawn as the same kind of straight
# stroke, so classify_page_vectors alone can't tell them apart. We never
# assume any specific color (white/black/gray/etc.) is or isn't a pipe --
# planners and CAD exports vary. Instead we look at each page's own color
# distribution: real multi-media plumbing (per a Farblegende-style
# convention, or just several distinctly-colored systems) leaves
# meaningful total length in more than one color. A single color that
# monopolizes the page's drawn line length while every other color
# combined is nearly nothing is far more likely to be one ubiquitous
# background layer than a pipe network -- regardless of what that color
# actually is.
#
# Segments flagged this way are never deleted: they move to
# `background_line_segments` and stay in the output for inspection: only
# `line_segments` feeds the graph/symbol builders.
COLOR_GROUP_ROUND_STEP = 0.05
DOMINANT_COLOR_MIN_SHARE = 0.85
DOMINANT_COLOR_MAX_OTHER_SHARE = 0.15
MIN_OTHER_COLOR_GROUPS = 3
MIN_OTHER_GROUP_SEGMENTS = 5


def _color_group_key(color: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if color is None:
        return None
    step = COLOR_GROUP_ROUND_STEP
    return tuple(round(c / step) * step for c in color)


def partition_background_lines(
    segments: list[LineSegment],
) -> tuple[list[LineSegment], list[LineSegment]]:
    """Splits line segments into (pipe_candidates, background_candidates).

    Returns the input unchanged (everything kept as a pipe candidate) unless
    one color group's share of total drawn length is an extreme, otherwise
    unexplainable outlier for *this specific page* -- see module docstring.
    """
    if not segments:
        return segments, []

    groups: dict[tuple[float, float, float] | None, list[LineSegment]] = defaultdict(list)
    lengths: dict[tuple[float, float, float] | None, float] = defaultdict(float)
    total_length = 0.0
    for seg in segments:
        key = _color_group_key(seg.color_rgb)
        length = dist(seg.p0, seg.p1)
        groups[key].append(seg)
        lengths[key] += length
        total_length += length

    if total_length <= 0 or len(groups) < 2:
        return segments, []

    ranked = sorted(groups.keys(), key=lambda k: -lengths[k])
    top_key = ranked[0]
    top_share = lengths[top_key] / total_length
    other_share = 1.0 - top_share
    other_groups_with_signal = [
        k for k in ranked[1:] if len(groups[k]) >= MIN_OTHER_GROUP_SEGMENTS
    ]

    is_background = (
        top_share >= DOMINANT_COLOR_MIN_SHARE
        and other_share <= DOMINANT_COLOR_MAX_OTHER_SHARE
        and len(other_groups_with_signal) >= MIN_OTHER_COLOR_GROUPS
    )
    if not is_background:
        return segments, []

    background = groups[top_key]
    kept = [seg for key, segs in groups.items() if key != top_key for seg in segs]
    return kept, background


def classify_page_vectors(drawings: list[dict], page_width: float, page_height: float) -> PageVectors:
    diagonal = (page_width**2 + page_height**2) ** 0.5
    glyph_max = diagonal * GLYPH_MAX_DIAGONAL_FRACTION
    symbol_max = diagonal * SYMBOL_MAX_DIAGONAL_FRACTION

    result = PageVectors()

    for d in drawings:
        items = d.get("items") or []
        if not items:
            continue
        rect = d.get("rect")
        bbox: BBox = (rect.x0, rect.y0, rect.x1, rect.y1) if rect is not None else None
        is_fill = d.get("type") in ("f", "fs")
        has_curve_item = any(it[0] == "c" for it in items)
        line_items = [it for it in items if it[0] == "l"]

        max_dim = bbox_max_dim(bbox) if bbox else 0.0

        if is_fill and max_dim <= glyph_max:
            result.glyph_primitives.append(
                VectorPrimitive(bbox=bbox, has_curves=has_curve_item, item_count=len(items), is_fill=True)
            )
            continue

        if not is_fill and not has_curve_item and line_items and max_dim > glyph_max:
            # Pure straight-line stroke: treat every segment individually so
            # the graph builder can snap shared endpoints across drawings.
            color = d.get("color")
            width = d.get("width")
            for it in line_items:
                p0 = (it[1].x, it[1].y)
                p1 = (it[2].x, it[2].y)
                if p0 == p1:
                    continue
                result.line_segments.append(
                    LineSegment(p0=p0, p1=p1, color_rgb=tuple(color) if color else None, stroke_width=width)
                )
            continue

        # Everything else: curved strokes/fills of plausible symbol size,
        # or small stroke clusters (arrowheads, hatch marks). Oversized
        # blobs (> symbol_max) are almost certainly merged pipe runs or
        # title-block borders, not a real symbol, so they're dropped.
        if max_dim <= symbol_max:
            result.symbol_primitives.append(
                VectorPrimitive(bbox=bbox, has_curves=has_curve_item, item_count=len(items), is_fill=is_fill)
            )

    result.line_segments, result.background_line_segments = partition_background_lines(
        result.line_segments
    )
    return result
