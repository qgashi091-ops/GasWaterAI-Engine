"""Deterministic gate: does an actual, spatially isolated graphical
component SYMBOL exist inside a legend entry's `symbol_bbox`, or is the
region really text, a table/dimension line, or a pipe/media-line fragment
that a text-driven relevance match wrongly let through?

Root cause this exists to fix: `relevance.py` classifies a legend entry from
its ADJACENT TEXT alone ("52.1 Wohnen / Essen Pex-Verteiler 54" contains
"pex-verteiler", a positive apparatus term) and `legend_entries.py`'s row
classifier only checks that a row LOOKS like an icon+label pair, never that
the icon zone contains a real icon. Neither step can catch a legend row
whose "icon column" is empty, is itself more text, or is a stray pipe/table
line -- exactly the domain-expert-reported noise (repeated PEX/distribution
rows, text-as-symbol, line styles, near-duplicate rows). This module is the
missing check, run BEFORE text is ever used to propose a name:
GRAPHIC SYMBOL must be established first; text may only NAME a symbol that
already passed this gate, never conjure one into existence.

Reuses `app.legend_structural_benchmark.vector_features` unmodified for the
actual geometry read (`page.get_drawings()`, same PageVectorIndex/
StructuralFingerprint machinery already proven in the legend structural
benchmark) -- this module only adds the pass/reject DECISION on top of that
fingerprint, plus a text-dominance check the benchmark never needed (it
matched two already-known-to-be-icons regions against each other; it never
had to decide whether a region was an icon in the first place).

Deliberately conservative throughout, per explicit instruction not to lower
the gate to grow the dataset: every threshold below is checked against a
50-item visual contact-sheet before publishing (see
scripts/apply_graphical_symbol_gate.py), and any threshold this module ships
with has been through at least one such round.

FIRST-ROUND FINDING, kept here so the "why" survives future edits: an
initial version of this gate used the proposed `symbol_bbox`'s own raw
aspect ratio (width/height) as a blanket "too elongated to be an icon"
veto. Running it against all 20 plans rejected 310 of 335 failures for
that reason ALONE -- including every one of the 150 already-human-reviewed
entries confirmed "correct" (Filter, Rueckschlagventil, Absperrklappe,
Sicherheitsventil, Wasserzaehler M 3, ...). Inspecting their own bboxes
showed why: this dataset's legend template draws a valve/fitting icon
flanked by short connecting pipe stubs INSIDE the same bounding box the
row classifier hands over, so a real icon's bbox is inherently wide and
short by drafting convention -- exactly as wide and short as a stretch of
bare pipe or a VPE dimension tick-series has. Raw bbox aspect ratio cannot
tell the two apart in this dataset; it was dropped from the gate entirely
in favor of two signals that the same data showed DO separate them
cleanly: how many drawn primitives the region holds (a real icon: a
handful to a few dozen; a tick-series/dimension notation: 70-270+), and
how concentrated their orientations are (a tick series is >=78% one
direction; every confirmed real icon sampled was <=57%, except one 3-item
icon at 87% -- explicitly exempted below, see MIN_ITEMS_FOR_DOMINANCE_CHECK
-- since a dominance ratio computed from only 2-3 segments is not a
meaningful statistic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.legend_structural_benchmark.vector_features import (
    ANGLE_BINS,
    StructuralFingerprint,
    compute_structural_fingerprint,
)

BBox = tuple[float, float, float, float]

# Every confirmed real icon sampled across all 20 plans had at most 29
# drawn primitives; every confirmed dimension/tick-series noise item had at
# least 76. A region with more primitives than this, and no curve/rect
# evidence of a real compound body, is a dense field of tick marks or
# hatching, not a discrete icon.
MAX_ITEM_COUNT_WITHOUT_CURVE_OR_RECT = 45

# Fraction of a rotation-normalized angle histogram's total weight sitting
# in a single 15-degree bin above which the region is "essentially all one
# direction" -- a pipe run, a parallel media-line pair, a dashed-line
# fragment run, or a VPE-style dimension tick series, never a compound icon
# shape (see module docstring for the measured gap between the two
# populations). Only applied once there are enough segments for the
# statistic to mean anything.
COLLINEAR_DOMINANCE_MAX = 0.75
MIN_ITEMS_FOR_DOMINANCE_CHECK = 6

# A genuine legend icon (valve, meter, filter body, fitting) always shows
# ONE of: a curve (circle/arc -- valve wheel, meter dial), a rect (a body,
# a box symbol), a true interior line crossing (a bowtie/junction the icon
# is built from), enough angular spread across >=2 line directions (a
# triangle/chevron/elbow shape, whose edges typically MEET AT A VERTEX
# rather than cross through each other's interior -- so crossing_count
# alone misses this whole class of icon), or a small thick-stroke ("qu")
# element paired with very few other primitives (a compact box/tick icon
# too small for the dominance statistic to apply). Absent all of these,
# there is no positive evidence of compound icon structure, only line
# fragments -- and per the hard rule, absence of evidence must exclude.
MIN_ITEM_COUNT = 2
SMALL_QUAD_ICON_MAX_ITEMS = 15

# A bare rect is common in both real icons AND boxed pipe-type/media-line
# swatches ("WWV", "WW-R", ... a colored dash-dot line drawn through a
# labeled box) -- measured across 5 confirmed swatch instances, all had
# dominance >=0.61; this threshold sits below every one of them while still
# well above genuinely multi-directional content.
RECT_DIVERSITY_MAX = 0.60

# A crop whose area is mostly covered by text glyphs, with only trivial
# vector content, is a text label that happened to sit inside the proposed
# symbol_bbox -- not an icon. Real icons carry little to no text of their
# own (a lone reference letter/number at most).
TEXT_AREA_RATIO_MAX = 0.30
TEXT_DOMINANT_MAX_ITEMS = 2

# A table/dimension-line grid is built almost entirely from many straight
# segments at exactly two orthogonal angles (0 deg and 90 deg bins) with no
# curves and no filled/rect body -- unlike a real icon, which rarely has
# more than a handful of primitives at legend scale.
GRID_MIN_LINE_COUNT = 8
GRID_ORTHOGONAL_BIN_SPAN = 1  # bins within +-1 of 0 deg or 90 deg count as "orthogonal"


@dataclass
class GateResult:
    passes: bool
    reason: str  # "ok" when passes, else one of the REJECT_* codes below
    fingerprint: StructuralFingerprint
    text_area_ratio: float

    def to_dict(self) -> dict:
        d = self.fingerprint.to_dict()
        d["passes"] = self.passes
        d["reason"] = self.reason
        d["text_area_ratio"] = round(self.text_area_ratio, 4)
        return d


REJECT_NO_STRUCTURE = "no_structure"  # 0-1 primitives: a bare line/curve, nothing else
REJECT_TOO_MANY_PRIMITIVES = "dense_tick_or_hatch_field"  # tick-series/dimension notation, not an icon
REJECT_COLLINEAR = "collinear_lines"  # parallel media lines / dashed-line fragments / tick series
REJECT_TEXT_DOMINATED = "text_dominated"  # mostly text glyphs, trivial geometry
REJECT_GRID = "table_or_dimension_grid"  # orthogonal line grid, no curves/body
REJECT_NO_COMPOUND_EVIDENCE = "no_compound_icon_evidence"  # enough primitives, but no real 2D shape evidence


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_area(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def compute_text_area_ratio(symbol_bbox_native: BBox, text_spans_native: list) -> float:
    """`text_spans_native` are objects/tuples exposing a `.bbox` in the SAME
    native PDF frame as `symbol_bbox_native` (v0.1's `schema.TextSpan.bbox`
    already is -- see extract.py's own coordinate-space docstring). Sums
    each overlapping span's own clipped area; a span only partially inside
    the bbox contributes only its overlapping portion, never its full
    extent, so one long label merely grazing the region cannot alone flag
    it as text-dominated."""
    area = _bbox_area(symbol_bbox_native)
    if area <= 0:
        return 0.0
    covered = 0.0
    for span in text_spans_native:
        span_bbox = span.bbox if hasattr(span, "bbox") else span
        covered += _overlap_area(tuple(span_bbox), symbol_bbox_native)
    return min(1.0, covered / area)


def _orthogonal_dominance(raw_histogram: list[float]) -> float:
    """Fraction of RAW (not rotation-normalized) angle-histogram weight
    sitting in the two orthogonal bin groups (~0 deg and ~90 deg) -- the
    signature of a table border or a dimension line's grid, which a
    rotation-normalized histogram would otherwise obscure by always
    shifting the largest bin to index 0 regardless of which axis it is."""
    n = len(raw_histogram)
    if n == 0 or sum(raw_histogram) <= 0:
        return 0.0
    quarter = n // 4  # 90 deg in bin units (n=12 -> 3 bins = 45deg... adjust below)
    # ANGLE_BINS covers 0-180 deg; 90 deg sits at bin index n/2.
    zero_idx, ninety_idx = 0, n // 2
    span = GRID_ORTHOGONAL_BIN_SPAN
    weight = 0.0
    for idx in range(n):
        d0 = min(abs(idx - zero_idx), n - abs(idx - zero_idx))
        d90 = min(abs(idx - ninety_idx), n - abs(idx - ninety_idx))
        if d0 <= span or d90 <= span:
            weight += raw_histogram[idx]
    return weight


def evaluate_graphical_symbol(
    fingerprint: StructuralFingerprint,
    text_area_ratio: float,
) -> GateResult:
    fp = fingerprint
    # NOTE: curves are treated as strong, unconditional evidence (rare in
    # this dataset, reliably a valve wheel/meter dial); a bare RECT is not
    # -- a text label's own border, or a boxed pipe-type/media-line swatch
    # ("WWV", "WW-R", ... boxes with a colored dash-dot line running
    # through them, exactly the "dashed/solid media line" noise this gate
    # must reject) both draw a rect just as readily as a real component
    # body does. A rect only counts as compound evidence when paired with
    # real angular diversity inside the box (see has_compound_evidence
    # below); it never exempts a region from the item-count/collinear/grid
    # vetoes the way a curve does.
    has_curve = fp.curve_count >= 1
    dominant_bin_weight = max(fp.angle_histogram_rotation_normalized, default=0.0)

    if fp.total_item_count <= 1:
        return GateResult(False, REJECT_NO_STRUCTURE, fp, text_area_ratio)

    if text_area_ratio > TEXT_AREA_RATIO_MAX and fp.total_item_count <= TEXT_DOMINANT_MAX_ITEMS:
        return GateResult(False, REJECT_TEXT_DOMINATED, fp, text_area_ratio)

    if fp.total_item_count > MAX_ITEM_COUNT_WITHOUT_CURVE_OR_RECT and not has_curve:
        return GateResult(False, REJECT_TOO_MANY_PRIMITIVES, fp, text_area_ratio)

    if (
        fp.total_item_count >= MIN_ITEMS_FOR_DOMINANCE_CHECK
        and dominant_bin_weight > COLLINEAR_DOMINANCE_MAX
        and not has_curve
    ):
        return GateResult(False, REJECT_COLLINEAR, fp, text_area_ratio)

    if (
        fp.line_count >= GRID_MIN_LINE_COUNT
        and not has_curve
        and fp.fill_fraction == 0.0
        and fp.crossing_count <= 1  # a real valve/fitting icon in this range (16-27 items)
        # always showed >=5 true interior crossings when it had no curve/rect
        # of its own (Motor. Absperrklappe, Rueckschlagklappe); an
        # orthogonal-heavy region with 0-1 crossings at this item count is a
        # vector-drawn numeral/label glyph cluster or a table/dimension
        # grid, not an icon.
        and _orthogonal_dominance(fp.angle_histogram) > 0.90
    ):
        return GateResult(False, REJECT_GRID, fp, text_area_ratio)

    # Positive evidence of a real, non-collinear 2D shape: a curve (circle/
    # arc -- always sufficient alone), a true interior crossing (bowtie/
    # junction), angular spread across >=2 directions (a triangle/chevron/
    # elbow, whose edges typically meet AT a vertex rather than cross
    # THROUGH one another -- crossing_count alone is blind to this shape
    # family), a small thick-stroke element in an otherwise near-empty
    # region (too few items for the dominance statistic to mean anything),
    # or a rect COMBINED with real diversity inside it -- a bare rect is
    # never enough by itself (see has_curve's docstring above: a boxed
    # pipe-type/media-line swatch draws a rect just as readily as a real
    # component body).
    has_compound_evidence = (
        has_curve
        or fp.crossing_count >= 1
        # Same MIN_ITEMS_FOR_DOMINANCE_CHECK floor as the collinear-reject
        # check above, for the same reason: a directional-spread statistic
        # from only 2-5 stray segments (e.g. a header's underline plus a
        # small bullet accent -- LEGENDE SANITAER, 4 items, no crossing, no
        # curve/rect) is not meaningful evidence of a real icon shape either
        # way.
        # A box drawn from 4 plain line segments (never a native "re" rect
        # op, so rect_count stays 0) plus a dash-line inside it -- exactly
        # the boxed pipe-type swatches this gate exists to reject -- splits
        # its weight between the SAME two orthogonal (0/90 deg) bins a real
        # rectangle border always produces, which can land its single-bin
        # dominant_bin_weight below COLLINEAR_DOMINANCE_MAX (measured:
        # 0.65-0.68 for 4 confirmed swatch instances) even though the
        # region has no genuine OFF-AXIS diversity at all. Requiring
        # orthogonal_dominance below the same GRID threshold here closes
        # that gap: a real triangle/chevron/elbow icon's non-90-degree edge
        # keeps its orthogonal_dominance low, while a box+axis-aligned-line
        # region's does not.
        or (
            fp.total_item_count >= MIN_ITEMS_FOR_DOMINANCE_CHECK
            and dominant_bin_weight <= COLLINEAR_DOMINANCE_MAX
            and _orthogonal_dominance(fp.angle_histogram) <= 0.90
        )
        or (fp.quad_count >= 1 and fp.total_item_count <= SMALL_QUAD_ICON_MAX_ITEMS)
        or (
            fp.rect_count >= 1
            and fp.total_item_count >= MIN_ITEMS_FOR_DOMINANCE_CHECK
            and dominant_bin_weight <= RECT_DIVERSITY_MAX
        )
    )
    if fp.total_item_count < MIN_ITEM_COUNT or not has_compound_evidence:
        return GateResult(False, REJECT_NO_COMPOUND_EVIDENCE, fp, text_area_ratio)

    return GateResult(True, "ok", fp, text_area_ratio)
