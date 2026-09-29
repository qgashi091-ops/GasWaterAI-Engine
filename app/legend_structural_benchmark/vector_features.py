"""Deterministic structural (vector-geometry) fingerprint for a small PDF
region -- either a legend icon zone or a schema-area symbol candidate's
bbox.

This is new code, not a v0.1 module: v0.1's own `symbols.py` deliberately
stops at bbox/curve-presence/primitive-count ("This module does not attempt
to name the symbol ... we only hand over geometry" -- see its docstring),
which is not enough to compare two symbol INSTANCES against each other.
This module goes one level deeper, re-reading the SAME raw
`page.get_drawings()` items (never v0.1's bucketed `line_segments`/
`symbol_primitives`, so a legend icon built from bare straight lines --
common in this dataset, see module docstring in matching.py -- is handled
identically to a curved one) and computing several independent, explainable
geometric properties per region:

- line/curve/rect/quad item counts (a "qu" item is PyMuPDF's rendering of a
  THICK straight stroke as a quadrilateral; its longest edge is folded into
  the line-angle evidence below rather than discarded, or every icon drawn
  with heavy strokes would silently lose its structure)
- a length-weighted angle histogram over undirected line orientation
  (0-180 degrees, 12 x 15-degree bins), then CIRCULARLY SHIFTED so the
  bin holding the most weight sits at index 0. This is a real, but
  DELIBERATELY LIMITED, rotation-invariance: it only cancels rotations that
  are (approximately) a multiple of one 15-degree bin, not every continuous
  rotation angle. That is stated plainly here and in the benchmark report
  rather than implied to be exact.
- bounding-box aspect ratio (max(w,h)/min(w,h), so 90-degree rotation and
  mirroring don't change it)
- proper line-segment crossing count (true interior intersections only --
  segments that merely share an endpoint, as any two touching pipe/icon
  strokes normally do, are NOT counted as a crossing)
- "curviness" (fraction of items that are Bezier curves) as a cheap,
  honestly-named proxy for circular/arc content -- this is NOT a circle fit;
  a fingerprint with only straight strokes always reads curviness=0, which
  is correct and expected for line-built icons.

None of this is raster/pixel-based and none of it is trained: every feature
is read directly from the PDF's own vector path data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

BBox = tuple[float, float, float, float]
Point = tuple[float, float]

ANGLE_BINS = 12
BIN_WIDTH_DEG = 180.0 / ANGLE_BINS

# Chosen once, before this module was ever run against any of the 5
# benchmark plans' schema areas or any human label -- see
# docs/v04-legend-structural-benchmark-report.md section 3. Never
# re-tuned afterwards.
STRUCTURAL_WEIGHTS = {
    "angle": 0.30,
    "composition": 0.20,
    "size": 0.15,
    "count": 0.15,
    "crossing": 0.10,
    "curviness": 0.10,
}
assert abs(sum(STRUCTURAL_WEIGHTS.values()) - 1.0) < 1e-9


def _seg_angle_deg(p0: Point, p1: Point) -> float:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _seg_len(p0: Point, p1: Point) -> float:
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


@dataclass
class _LineSample:
    p0: Point
    p1: Point
    length: float
    angle_deg: float


@dataclass
class StructuralFingerprint:
    bbox: BBox
    line_count: int
    curve_count: int
    rect_count: int
    quad_count: int
    total_item_count: int
    angle_histogram: list = field(default_factory=list)
    angle_histogram_rotation_normalized: list = field(default_factory=list)
    aspect_ratio: float = 1.0
    crossing_count: int = 0
    curviness: float = 0.0
    fill_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "line_count": self.line_count,
            "curve_count": self.curve_count,
            "rect_count": self.rect_count,
            "quad_count": self.quad_count,
            "total_item_count": self.total_item_count,
            "angle_histogram": [round(v, 4) for v in self.angle_histogram],
            "angle_histogram_rotation_normalized": [round(v, 4) for v in self.angle_histogram_rotation_normalized],
            "aspect_ratio": round(self.aspect_ratio, 4),
            "crossing_count": self.crossing_count,
            "curviness": round(self.curviness, 4),
            "fill_fraction": round(self.fill_fraction, 4),
        }


def _rotation_normalize(hist: list[float]) -> list[float]:
    if not any(hist):
        return list(hist)
    max_idx = max(range(len(hist)), key=lambda i: hist[i])
    return hist[max_idx:] + hist[:max_idx]


def _segments_cross(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    """Proper (interior) intersection test -- excludes shared/near-shared
    endpoints, which is what any two touching strokes of one icon normally
    have and must never be double-counted as a structural 'crossing'."""

    def _turn(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = _turn(b0, b1, a0), _turn(b0, b1, a1)
    d3, d4 = _turn(a0, a1, b0), _turn(a0, a1, b1)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0


def _clip_segment_to_bbox(p0: Point, p1: Point, bbox: BBox) -> Optional[tuple[Point, Point]]:
    """Liang-Barsky segment-vs-rectangle clip. Returns the portion of the
    segment that lies inside `bbox`, or None if it doesn't cross it at all.

    This is the fix for a real bug an axis-aligned-line bbox exposes: a
    perfectly horizontal or vertical line segment has a ZERO-AREA bounding
    box (its own height or width is exactly 0), so an area-RATIO overlap
    test (intersection-area / item-area) always evaluates to 0/0-or-tiny
    for it, no matter how much of the segment actually runs through the
    target region -- silently discarding the overwhelming majority of line
    items in this dataset's CAD exports, which are almost all perfectly
    horizontal/vertical/diagonal straight strokes with no width of their
    own. Clipping the segment itself sidesteps that: it asks "what part of
    this line is inside the box", which is well-defined even for a
    zero-thickness line, and naturally handles a long unrelated line merely
    grazing the box's corner by returning only that short clipped portion,
    not the line's full original length."""
    x0, y0, x1, y1 = bbox
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    return (p0[0] + t0 * dx, p0[1] + t0 * dy), (p0[0] + t1 * dx, p0[1] + t1 * dy)


def _bbox_overlap_ratio(item_bbox: BBox, target: BBox) -> float:
    x0, y0, x1, y1 = target
    rx0, ry0, rx1, ry1 = item_bbox
    ix0, iy0 = max(x0, rx0), max(y0, ry0)
    ix1, iy1 = min(x1, rx1), min(y1, ry1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area = max(1e-9, (rx1 - rx0) * (ry1 - ry0))
    return inter / area


@dataclass
class _ItemRecord:
    op: str
    bbox: BBox
    is_fill: bool
    p0: Point = None
    p1: Point = None


class PageVectorIndex:
    """A one-time, per-page spatial index over every INDIVIDUAL drawing
    ITEM (never the enclosing drawing dict's own aggregate `rect`, which
    frequently spans a much larger area than any one of its items -- see
    module docstring). Built once per page and reused for every
    legend-icon-zone and every schema-candidate bbox on that page, so a
    page with tens of thousands of line items (common in this dataset's
    CAD exports) is walked once, not once per query."""

    def __init__(self, drawings: list[dict], cell_size: float):
        self._cell_size = max(cell_size, 1e-3)
        self._grid: dict[tuple[int, int], list[_ItemRecord]] = {}
        for d in drawings:
            items = d.get("items") or []
            if not items:
                continue
            is_fill = d.get("type") in ("f", "fs")
            for it in items:
                rec = self._item_record(it, is_fill)
                if rec is None:
                    continue
                self._insert(rec)

    @staticmethod
    def _item_record(it: tuple, is_fill: bool) -> "_ItemRecord | None":
        op = it[0]
        if op == "l":
            p0, p1 = (it[1].x, it[1].y), (it[2].x, it[2].y)
            bbox = (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
            return _ItemRecord(op="l", bbox=bbox, is_fill=is_fill, p0=p0, p1=p1)
        if op == "c":
            pts = [(p.x, p.y) for p in it[1:]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return _ItemRecord(op="c", bbox=(min(xs), min(ys), max(xs), max(ys)), is_fill=is_fill)
        if op == "re":
            r = it[1]
            return _ItemRecord(op="re", bbox=(r.x0, r.y0, r.x1, r.y1), is_fill=is_fill)
        if op == "qu":
            q = it[1]
            pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.ll.x, q.ll.y), (q.lr.x, q.lr.y)]
            edges = [(pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[3]), (pts[2], pts[3])]
            best = max(edges, key=lambda e: _seg_len(*e))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return _ItemRecord(op="qu", bbox=(min(xs), min(ys), max(xs), max(ys)), is_fill=is_fill, p0=best[0], p1=best[1])
        return None

    def _cells_for(self, bbox: BBox):
        x0, y0, x1, y1 = bbox
        c = self._cell_size
        return (
            (cx, cy)
            for cx in range(int(x0 // c), int(x1 // c) + 1)
            for cy in range(int(y0 // c), int(y1 // c) + 1)
        )

    def _insert(self, rec: _ItemRecord) -> None:
        for cell in self._cells_for(rec.bbox):
            self._grid.setdefault(cell, []).append(rec)

    def query(self, bbox: BBox, min_clipped_length: float, area_overlap_ratio_min: float) -> list[tuple]:
        """Returns `(op, p0, p1, is_fill)` tuples -- `p0`/`p1` are the
        CLIPPED sub-segment for "l"/"qu" items (see `_clip_segment_to_bbox`)
        and `(None, None)` for "c"/"re" items, which are still filtered by
        their own bbox's area-overlap ratio (they normally have real,
        non-degenerate area in this dataset -- unlike lines -- so that test
        remains appropriate for them)."""
        seen: set[int] = set()
        out: list[tuple] = []
        for cell in self._cells_for(bbox):
            for rec in self._grid.get(cell, ()):
                key = id(rec)
                if key in seen:
                    continue
                seen.add(key)
                if rec.op in ("l", "qu"):
                    clipped = _clip_segment_to_bbox(rec.p0, rec.p1, bbox)
                    if clipped is None:
                        continue
                    if _seg_len(*clipped) < min_clipped_length:
                        continue
                    out.append((rec.op, clipped[0], clipped[1], rec.is_fill))
                else:
                    if _bbox_overlap_ratio(rec.bbox, bbox) >= area_overlap_ratio_min:
                        out.append((rec.op, None, None, rec.is_fill))
        return out


def build_page_index(drawings: list[dict], page_width: float, page_height: float) -> PageVectorIndex:
    diagonal = (page_width**2 + page_height**2) ** 0.5
    return PageVectorIndex(drawings, cell_size=max(diagonal * 0.005, 2.0))


def compute_structural_fingerprint(
    index: PageVectorIndex, bbox: BBox, min_clipped_length: float = 0.5, area_overlap_ratio_min: float = 0.3
) -> StructuralFingerprint:
    """`index` is a `PageVectorIndex` built once per page (see
    `build_page_index`) and reused across every legend-icon-zone and
    schema-candidate bbox on that page. Line-like items ("l"/"qu") are
    included via exact segment-vs-`bbox` clipping (see
    `_clip_segment_to_bbox`) with only the CLIPPED, in-region portion
    contributing length/angle evidence, so a long unrelated line merely
    grazing this bbox contributes only that short clipped stub, not its
    full original extent. Curve/rect items are included via their own
    bbox's area-overlap ratio, which remains appropriate for them since
    they are not degenerate zero-area shapes in this dataset."""
    lines: list[_LineSample] = []
    curve_count = 0
    rect_count = 0
    quad_count = 0
    fill_flags: list[bool] = []

    for op, p0, p1, is_fill in index.query(bbox, min_clipped_length, area_overlap_ratio_min):
        fill_flags.append(is_fill)
        if op == "l" or op == "qu":
            length = _seg_len(p0, p1)
            lines.append(_LineSample(p0, p1, length, _seg_angle_deg(p0, p1)))
            if op == "qu":
                quad_count += 1
        elif op == "c":
            curve_count += 1
        elif op == "re":
            rect_count += 1

    total_item_count = len(lines) + curve_count + rect_count + quad_count
    x0, y0, x1, y1 = bbox

    hist = [0.0] * ANGLE_BINS
    for ln in lines:
        b = min(ANGLE_BINS - 1, int(ln.angle_deg // BIN_WIDTH_DEG))
        hist[b] += ln.length
    hist_sum = sum(hist)
    if hist_sum > 0:
        hist = [v / hist_sum for v in hist]
    hist_norm = _rotation_normalize(hist)

    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    aspect_ratio = max(w, h) / min(w, h)

    crossing_count = 0
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            if _segments_cross(lines[i].p0, lines[i].p1, lines[j].p0, lines[j].p1):
                crossing_count += 1

    curviness = (curve_count / total_item_count) if total_item_count else 0.0
    fill_fraction = (sum(fill_flags) / len(fill_flags)) if fill_flags else 0.0

    return StructuralFingerprint(
        bbox=bbox,
        line_count=len(lines),
        curve_count=curve_count,
        rect_count=rect_count,
        quad_count=quad_count,
        total_item_count=total_item_count,
        angle_histogram=hist,
        angle_histogram_rotation_normalized=hist_norm,
        aspect_ratio=aspect_ratio,
        crossing_count=crossing_count,
        curviness=curviness,
        fill_fraction=fill_fraction,
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if (na == 0.0 and nb == 0.0) else 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _ratio_similarity(a: float, b: float, max_log_ratio: float = math.log(3.0)) -> float:
    if a <= 0 or b <= 0:
        return 1.0 if a == b else 0.0
    r = abs(math.log(a / b))
    return max(0.0, 1.0 - min(1.0, r / max_log_ratio))


def _count_similarity(a: int, b: int) -> float:
    denom = max(a, b, 1)
    return max(0.0, 1.0 - abs(a - b) / denom)


def _composition_similarity(fp_a: StructuralFingerprint, fp_b: StructuralFingerprint) -> float:
    def frac(fp: StructuralFingerprint) -> tuple[float, float, float]:
        total = max(1, fp.total_item_count)
        return (fp.line_count / total, fp.curve_count / total, (fp.rect_count + fp.quad_count) / total)

    a, b = frac(fp_a), frac(fp_b)
    l1 = sum(abs(x - y) for x, y in zip(a, b))  # in [0, 2]
    return max(0.0, 1.0 - l1 / 2.0)


def structural_similarity(fp_a: StructuralFingerprint, fp_b: StructuralFingerprint) -> dict:
    """Returns the combined `structural_score` (0..1) plus every named
    sub-score, so a benchmark report or a spot-check can see exactly which
    evidence drove (or failed to drive) a given match -- never just a single
    opaque number."""
    angle_sim = _cosine(fp_a.angle_histogram_rotation_normalized, fp_b.angle_histogram_rotation_normalized)
    composition_sim = _composition_similarity(fp_a, fp_b)
    size_sim = _ratio_similarity(fp_a.aspect_ratio, fp_b.aspect_ratio)
    count_sim = _count_similarity(fp_a.total_item_count, fp_b.total_item_count)
    crossing_sim = _count_similarity(fp_a.crossing_count, fp_b.crossing_count)
    curviness_sim = 1.0 - abs(fp_a.curviness - fp_b.curviness)

    score = (
        STRUCTURAL_WEIGHTS["angle"] * angle_sim
        + STRUCTURAL_WEIGHTS["composition"] * composition_sim
        + STRUCTURAL_WEIGHTS["size"] * size_sim
        + STRUCTURAL_WEIGHTS["count"] * count_sim
        + STRUCTURAL_WEIGHTS["crossing"] * crossing_sim
        + STRUCTURAL_WEIGHTS["curviness"] * curviness_sim
    )
    return {
        "structural_score": round(score, 4),
        "angle_sim": round(angle_sim, 4),
        "composition_sim": round(composition_sim, 4),
        "size_sim": round(size_sim, 4),
        "count_sim": round(count_sim, 4),
        "crossing_sim": round(crossing_sim, 4),
        "curviness_sim": round(curviness_sim, 4),
    }
