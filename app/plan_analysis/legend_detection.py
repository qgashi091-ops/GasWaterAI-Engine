"""Legend region detection -- v0.3, new. Locates a plan's OWN legend
(Legende/Symbole/Zeichenerklaerung block) using only deterministic evidence
already available from v0.1's text extraction -- no hardcoded coordinates,
no hardcoded filenames, and no dependency on any specific plan.

EVIDENCE USED (combined, none alone is required):
- Text density / geometric separation: a real legend or table is a compact,
  densely-packed 2D cluster of text rows -- something a scattered
  installation drawing's own labels never produce, since they sit wherever
  their pipe/component happens to be. Measured with the SAME spatial
  clustering utility v0.1 already uses for grouping nearby vector
  primitives into a symbol (geometry.SpatialBBoxClusterer) -- applied here
  to text-row bounding boxes instead. This alone needs no reference to what
  the labels say, so it also works on a plan with no explicit heading.
- Heading match: a text span containing "Legende"/"Symbole"/
  "Zeichenerklaerung" near the top of a dense block is strong, corroborating
  (not required) evidence.

Multiple qualifying regions are ALL returned (see AMBIGUITY): W-003 itself
has more than one dense text block that passes this test -- "LEGENDE
SANITAER" (a real symbol legend) and "Legende Verteilbatterie" (a bordered
numeric table with no symbol column at all) chief among them. Telling a
usable symbol legend apart from a lookalike table is Phase 2's job
(legend_entries.py), not this module's -- this module's only claim is "here
is a dense text block," which is true of both.

ROTATION HANDLING (found while building v0.2's component_facts.py and
directly relevant here too): schema.PageAnalysis text bboxes are in the
PDF's own UNROTATED coordinate frame. For a rotated page (true for W-003 --
rotated 90 degrees) that frame does NOT match how a human reads the page:
a "row" of displayed text is actually a run of THIN, TALL bboxes at nearly
the same unrotated-X (not the same unrotated-Y). Every bbox is rotated
through the page's own rotation_matrix (identity when rotation==0) before
any clustering, so "row"/"dense block" mean the same thing regardless of a
given plan's page rotation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import schema
from .geometry import SpatialBBoxClusterer, UnionFind, bbox_union

HEADING_PATTERN = re.compile(r"legende|symbole|zeichenerkl(ae|ä)rung", re.IGNORECASE)

MIN_ROWS = 8  # a title block / address letterhead never has this many densely-packed rows
ROW_Y_TOLERANCE = 2.0  # pt, in DISPLAY space -- spans within this y-center distance are "the same row"
# Clustering gap for INDIVIDUAL text-span bboxes (DISPLAY space, pt), not
# for whole-row unions. A whole-row union bbox is unsafe to cluster on a
# busy technical drawing: many unrelated text groups happen to sit at the
# same display-y (dimension chains, other tables, scattered pipe labels),
# so grouping "everything at this y, anywhere on the page" into one row
# bbox -- and then clustering those -- silently bridges completely
# unrelated content the moment two such groups share a height, producing a
# nonsense near-full-page-width "legend." Clustering individual spans in
# both X and Y avoids this: two spans only ever merge when they are
# actually near each other in 2D, which is what "one column of a
# legend/table" means. Measured on W-003's real "LEGENDE SANITAER" block:
# within one column, consecutive entries sit 3-20pt apart; the gap to the
# NEXT column over is 75-90pt -- far more than any within-column gap, so a
# threshold in between (here, comfortably below 75) separates columns from
# each other while never breaking a single column apart.
SEED_CLUSTER_GAP = 20.0
# Two independently-dense seed clusters (see above) are treated as sibling
# columns of the SAME legend/table -- and their bboxes merged -- only when
# both hold: (a) they sit close enough in X to plausibly be adjacent
# columns (real measured inter-column gap on W-003: ~75-90pt; unrelated
# content elsewhere on a technical drawing that also happens to be dense is
# essentially always much farther away in X, since it belongs to a
# different drawn feature), and (b) their Y-extents substantially overlap,
# since parallel columns of one table/legend run over roughly the same
# vertical span -- an unrelated dense block positioned to one side but at a
# different height would fail this even if X-close.
COLUMN_MERGE_MAX_X_GAP = 100.0
COLUMN_MERGE_MIN_Y_OVERLAP_RATIO = 0.25


@dataclass
class LegendRow:
    """One dense-text row inside a detected legend/table region, in DISPLAY
    space. `spans` keeps the original (schema.TextSpan, display_bbox) pairs
    so Phase 2 (legend_entries.py) can inspect provenance (native/ocr) and
    reuse the exact geometry without re-deriving it."""
    bbox: tuple
    text: str
    spans: list  # list[tuple[schema.TextSpan, tuple]]


@dataclass
class LegendCandidate:
    legend_id: str
    page: int
    bbox: tuple  # DISPLAY-space bbox (x0, y0, x1, y1)
    detection_method: str
    confidence: float
    evidence: dict
    row_count: int
    rows: list = None  # list[LegendRow], internal -- not part of the public dict
    heading_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "legend_id": self.legend_id,
            "page": self.page,
            "bbox_display_space": list(self.bbox),
            "detection_method": self.detection_method,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "row_count": self.row_count,
            "heading_text": self.heading_text,
            "rows_preview": [
                {"bbox": list(r.bbox), "text": r.text} for r in (self.rows or [])
            ],
        }


def _identity_transform(bbox: tuple) -> tuple:
    return bbox


def _make_rotator(rotation_matrix):
    if rotation_matrix is None:
        return _identity_transform

    def _rotate(bbox: tuple) -> tuple:
        import pymupdf
        r = pymupdf.Rect(bbox) * rotation_matrix
        return (r.x0, r.y0, r.x1, r.y1)

    return _rotate


def _row_key(y_center: float, existing_keys: list) -> float:
    for k in existing_keys:
        if abs(k - y_center) <= ROW_Y_TOLERANCE:
            return k
    return y_center


def _rows_within_cluster(members: list[tuple]) -> list[dict]:
    """Groups (span, display_bbox) pairs -- already known to be one spatial
    cluster -- into rows by DISPLAY-space y-center proximity. This is safe to
    do globally *within* a cluster (unlike across the whole page): every
    member here already passed the 2D spatial-proximity test, so two members
    sharing a y-band are actually part of the same local structure."""
    rows: dict[float, list] = {}
    row_keys: list[float] = []
    for span, dbbox in members:
        yc = (dbbox[1] + dbbox[3]) / 2.0
        key = _row_key(yc, row_keys)
        if key not in rows:
            rows[key] = []
            row_keys.append(key)
        rows[key].append((span, dbbox))

    out = []
    for k, spans in sorted(rows.items(), key=lambda kv: kv[0]):
        bbox = bbox_union([dbbox for _, dbbox in spans])
        out.append({"y": k, "bbox": bbox, "spans": spans})
    return out


def _y_overlap_ratio(a: tuple, b: tuple) -> float:
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    if overlap <= 0:
        return 0.0
    shorter = min(a[3] - a[1], b[3] - b[1])
    return overlap / shorter if shorter > 0 else 0.0


def _x_gap(a: tuple, b: tuple) -> float:
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0  # overlapping in x


def _is_sibling_column(seed_bbox: tuple, other_bbox: tuple) -> bool:
    return (
        _x_gap(seed_bbox, other_bbox) <= COLUMN_MERGE_MAX_X_GAP
        and _y_overlap_ratio(seed_bbox, other_bbox) >= COLUMN_MERGE_MIN_Y_OVERLAP_RATIO
    )


def detect_legend_candidates(page: schema.PageAnalysis, rotation_matrix=None) -> list[LegendCandidate]:
    """rotation_matrix: a pymupdf.Matrix (page.rotation_matrix) -- pass the
    real page's matrix for correct results on a rotated page; None is
    correct (identity) for an unrotated page or synthetic test data.

    Two passes:
    1. Cluster individual text spans with a small, intra-column gap so each
       resulting cluster is (at most) one dense column of rows -- never a
       page-wide blob, since real unrelated content is always much farther
       away than one column's own internal spacing.
    2. Union clusters that look like sibling columns of the same
       legend/table (close in X, overlapping in Y), transitively -- so
       column1<->column2<->column3 links up even when column1 and column3
       alone are not close enough in X -- reconstructing the full
       multi-column legend. A resulting group is only ever promoted to a
       candidate when it contains at least one cluster that independently
       proves row-density (a "seed") on its own, so two merely-adjacent
       sparse clusters elsewhere on the page can never combine into a false
       candidate by themselves.
    """
    to_display = _make_rotator(rotation_matrix)
    spans = [(span, to_display(span.bbox)) for span in page.text_spans if span.text.strip()]
    if len(spans) < MIN_ROWS:
        return []

    clusterer = SpatialBBoxClusterer([dbbox for _, dbbox in spans], gap=SEED_CLUSTER_GAP)
    raw_clusters = []
    for member_indices in clusterer.clusters():
        members = [spans[i] for i in member_indices]
        member_rows = _rows_within_cluster(members)
        bbox = bbox_union([r["bbox"] for r in member_rows])
        raw_clusters.append({"rows": member_rows, "bbox": bbox})

    seed_indices = [i for i, c in enumerate(raw_clusters) if len(c["rows"]) >= MIN_ROWS]

    # Union raw clusters that look like sibling columns of the same
    # legend/table (pairwise, over ALL clusters -- not just seeds -- so a
    # column can be reached transitively: column1<->column2<->column3, even
    # when column1 and column3 alone are not close enough in X). A group is
    # only ever promoted to a candidate below if it contains at least one
    # seed, so two merely-adjacent-but-sparse clusters can never combine
    # into a false candidate on their own.
    uf = UnionFind(len(raw_clusters))
    for i in range(len(raw_clusters)):
        for j in range(i + 1, len(raw_clusters)):
            if _is_sibling_column(raw_clusters[i]["bbox"], raw_clusters[j]["bbox"]):
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(raw_clusters)):
        groups.setdefault(uf.find(i), []).append(i)

    candidates: list[LegendCandidate] = []
    cand_idx = 0
    for member_idxs in groups.values():
        if not any(i in seed_indices for i in member_idxs):
            continue
        all_rows = [r for i in member_idxs for r in raw_clusters[i]["rows"]]
        column_count = len(member_idxs)
        cand_idx += 1
        all_rows.sort(key=lambda r: r["y"])
        bbox = bbox_union([r["bbox"] for r in all_rows])
        x0, y0, x1, y1 = bbox
        height = max(y1 - y0, 1e-6)
        density = len(all_rows) / height

        gaps = [all_rows[i + 1]["y"] - all_rows[i]["y"] for i in range(len(all_rows) - 1) if all_rows[i + 1]["y"] - all_rows[i]["y"] > 1e-6]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0

        heading = None
        search_margin = max(mean_gap * 4, 20.0)
        for span in page.text_spans:
            if not HEADING_PATTERN.search(span.text):
                continue
            hb = to_display(span.bbox)
            if (y0 - search_margin) <= hb[1] <= (y0 + mean_gap) and hb[0] <= x1 and hb[2] >= x0 - search_margin:
                heading = span.text
                break

        # Two-tier by construction: a heading match is strong, corroborating
        # evidence a plain dense-text block (e.g. a lookalike numeric table)
        # never has, so it must never be out-ranked by a headingless block's
        # row count or density alone -- the two bands (0.0-0.6, 0.6-1.0)
        # cannot overlap.
        density_term = min(0.2, density * 2.0)
        if heading:
            confidence = 0.6 + min(0.4, 0.02 * len(all_rows) + density_term)
        else:
            confidence = min(0.6, 0.2 + 0.01 * len(all_rows) + density_term)
        method = "text_density+heading_match" if heading else "text_density"

        legend_rows = [
            LegendRow(
                bbox=r["bbox"],
                text=" ".join(s.text.strip() for s, _ in r["spans"] if s.text.strip()),
                spans=r["spans"],
            )
            for r in all_rows
        ]

        candidates.append(LegendCandidate(
            legend_id=f"LG{page.page_number}_{cand_idx}",
            page=page.page_number,
            bbox=bbox,
            detection_method=method,
            confidence=confidence,
            evidence={
                "row_count": len(all_rows),
                "column_count": column_count,
                "mean_row_gap_pt": round(mean_gap, 2),
                "rows_per_pt_height": round(density, 4),
                "heading_found": heading is not None,
            },
            row_count=len(all_rows),
            rows=legend_rows,
            heading_text=heading,
        ))

    candidates.sort(key=lambda c: -c.confidence)
    return candidates
