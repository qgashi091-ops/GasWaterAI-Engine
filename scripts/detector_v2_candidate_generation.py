"""DETECTOR v2 -- targeted object-detection dataset, candidate generation.

Converts the ~113 text-localized occurrences of the 5 selected classes
(wasserzaehler, absperrarmatur, systemtrenner_ba, filter,
sicherheitsgruppe_ventil -- see the prior text-inventory turn) into
candidate BOUNDING BOXES for human review. Text is used ONLY to find WHERE
TO LOOK; it never becomes ground truth here or anywhere downstream until a
human confirms it (see the review tool this feeds).

METHOD per occurrence:
  0. EXCLUDE any text hit that falls inside a detected legend/table region
     on that page (reusing `app.plan_analysis.legend_detection` unmodified,
     fed this script's own already-extracted native text spans -- no need
     to run the full v0.1 pipeline for this). Found live, not hypothetical:
     without this step, a legend row naming a component (e.g. "Absperrklappe"
     in a "Legende Sanitär" table) was being treated as a physical
     occurrence, and the local geometry search then locked onto a
     DIFFERENT component's legend icon sitting nearby in the same table --
     silently wrong in exactly the way this task's whole point is to avoid
     automating past a human. Excluded hits are counted separately
     (`legend_only_excluded`), never included as occurrences.
     Only candidates with an actual matched heading (`heading_text` set,
     confidence 1.0, `detection_method == "text_density+heading_match"`)
     count as a real legend/table zone -- `detect_legend_candidates` also
     returns many low-confidence (~0.5-0.6) `text_density`-only candidates
     with no heading, which on a dense page can number in the dozens and
     blanket nearly the whole plan (confirmed live on DEV-07: 41 such
     candidates); a first version of this filter used all candidates
     unfiltered and wrongly excluded most real occurrences plan-wide.
  1. Cluster same-page text hits (NATIVE space, matching
     `page.get_drawings()`'s own frame) into physical occurrences -- same
     clustering approach already used in the prior text-inventory turn.
  2. Search a LOCAL neighborhood around the text cluster's centroid (a
     bounded grid of offsets x window sizes, NOT a full-page search) for
     the closest vector-geometry region that passes the already-validated,
     unmodified `graphical_symbol_gate` (excludes text/pipe-run/dimension-
     grid noise, requires real compound icon-like evidence). This is
     deliberately NOT the full-page structural-similarity search that
     failed in the symbol-reidentification experiment
     (data/dev_plans_v04/symbol_reid_experiment/report.md, verdict "STOP
     HANDCRAFTED SYMBOL MATCHING") -- there is no template-similarity
     scoring here at all, only "is there a real icon-like geometry cluster
     near this label," searched over a small bounded local neighborhood
     (roughly 200x200pt), not the whole page (millions of positions). A
     human reviews and can reject or redraw every single proposal, so a
     locally-wrong proposal costs one click, not a false statistic -- the
     failure mode the earlier experiment could not tolerate.
  3. Falls back to a fixed padded box around the text label itself when no
     nearby geometry passes the gate -- still reviewable, just flagged
     `bbox_source=text_fallback_no_geometry_match` so the human (and any
     later analysis) knows it is a weaker starting point.
  4. Renders one context crop per occurrence (large enough to see the
     component in its plumbing context) with the proposed box's pixel
     position recorded for a client-side highlight overlay -- no box is
     baked into the pixels, so the review tool's bbox editor can redraw
     precisely.

Only the 20 DEV-xx plans are read; W-001..W-010 and tests/golden_holdout
are never opened.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from app.legend_structural_benchmark.vector_features import (
    build_page_index,
    compute_structural_fingerprint,
)
from app.legend_symbol_dataset.graphical_symbol_gate import (
    compute_text_area_ratio,
    evaluate_graphical_symbol,
)
from app.plan_analysis.legend_detection import detect_legend_candidates
from app.plan_analysis.text import extract_native_text_spans

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
LEGEND_SYMBOLS_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "legend_symbol_dataset" / "legend_symbols.json"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v2_candidates"
OUT_CROPS_DIR = OUT_DIR / "crops"

PLAN_IDS = [f"DEV-{i:02d}" for i in range(1, 21)]

CLUSTER_DISTANCE_PT = 60.0

# Local geometry search around each text cluster's centroid -- bounded,
# not full-page (see module docstring).
SEARCH_OFFSET_RANGE_PT = 90.0
SEARCH_OFFSET_STEP_PT = 15.0
SEARCH_WINDOW_HALF_SIZES_PT = (12.0, 18.0, 25.0)
TEXT_OVERLAP_REJECT_FRACTION = 0.5  # a candidate window mostly covering the text's own bbox is the text, not a symbol

RENDER_DPI = 300.0
CONTEXT_HALF_SIDE_PT = 260.0  # -> ~520pt square context crop, generous per "sufficiently large context crop"
_RENDER_DPI_SCALE = RENDER_DPI / 72.0

CLASSES: dict[str, list[re.Pattern]] = {
    "wasserzaehler": [re.compile(p, re.IGNORECASE) for p in [
        r"wasserz(ä|ae)hler", r"\bwz\b(?!\s*\d*\s*x)",
    ]],
    "absperrarmatur": [re.compile(p, re.IGNORECASE) for p in [
        r"absperrarmatur", r"absperrventil", r"absperrhahn", r"absperrklappe", r"absperrschieber",
    ]],
    "systemtrenner_ba": [re.compile(p, re.IGNORECASE) for p in [
        r"systemtrenner", r"rohrtrenner", r"sicherungsarmatur\s*ba", r"\bba\b",
    ]],
    "filter": [re.compile(p, re.IGNORECASE) for p in [
        r"feinfilter", r"schmutzf(ä|ae)nger", r"\bfilter\b",
    ]],
    "sicherheitsgruppe_ventil": [re.compile(p, re.IGNORECASE) for p in [
        r"sicherheitsventil", r"sicherheitsgruppe", r"\bsv\b\s*\d",
    ]],
}


def _center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _bbox_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _to_display(bbox: tuple, rotation_matrix) -> tuple:
    if rotation_matrix is None:
        return bbox
    r = pymupdf.Rect(*bbox) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _overlap_fraction(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return inter / area_a


def _stable_id(*parts: str) -> str:
    return "DV" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


@dataclass
class TextCluster:
    plan_id: str
    page: int
    native_bbox: tuple
    texts: list = field(default_factory=list)


def _find_text_clusters(plan_id: str, page_number: int, spans_native: list, patterns: list[re.Pattern]) -> list[TextCluster]:
    hits = [(s.bbox, s.text) for s in spans_native if s.text.strip() and any(p.search(s.text.strip()) for p in patterns)]
    clusters: list[TextCluster] = []
    for bbox, text in hits:
        c = _center(bbox)
        merged = False
        for cl in clusters:
            if _dist(c, _center(cl.native_bbox)) <= CLUSTER_DISTANCE_PT:
                cl.native_bbox = _bbox_union(cl.native_bbox, bbox)
                cl.texts.append(text)
                merged = True
                break
        if not merged:
            clusters.append(TextCluster(plan_id=plan_id, page=page_number, native_bbox=bbox, texts=[text]))
    return clusters


def _legend_zones_native(page_number: int, spans_native: list, rotation_matrix) -> list[tuple]:
    """Detected legend/table regions on this page, in NATIVE space (to match
    the text clusters this filters). Reuses `legend_detection` unmodified,
    fed a minimal duck-typed page carrying only what it actually reads
    (`text_spans`, `page_number`) -- no need to run the full v0.1 pipeline
    just for this filter.

    `detect_legend_candidates` also returns many low-confidence
    (~0.5-0.6) `text_density`-only candidates with no matched heading --
    on a dense riser-diagram page these can number in the dozens and cover
    nearly the whole page (confirmed live on DEV-07: 41 such candidates).
    Those are NOT real legend tables; only a candidate with an actual
    matched heading (`heading_text` set, confidence 1.0,
    `detection_method == "text_density+heading_match"` -- confirmed on the
    real "Legende Sanitär" / "Legende Leitungen" / "FARBLEGENDE SANITÄR"
    cases) is treated as a genuine legend/table region here."""
    fake_page = type("P", (), {"text_spans": spans_native, "page_number": page_number})()
    candidates = detect_legend_candidates(fake_page, rotation_matrix=rotation_matrix)
    inverse = ~rotation_matrix if rotation_matrix else None
    zones = []
    for c in candidates:
        if not c.heading_text:
            continue
        if inverse is None:
            zones.append(c.bbox)
        else:
            r = pymupdf.Rect(*c.bbox) * inverse
            zones.append((r.x0, r.y0, r.x1, r.y1))
    return zones


def _center_inside_any(bbox: tuple, zones: list[tuple], margin: float = 20.0) -> bool:
    cx, cy = _center(bbox)
    for z in zones:
        if (z[0] - margin) <= cx <= (z[2] + margin) and (z[1] - margin) <= cy <= (z[3] + margin):
            return True
    return False


def _propose_bbox(index, text_native_bbox: tuple, text_spans_native: list) -> tuple[tuple, str]:
    """Returns (proposed_native_bbox, bbox_source)."""
    cx, cy = _center(text_native_bbox)
    best = None  # (distance, bbox)
    offsets = [i * SEARCH_OFFSET_STEP_PT for i in range(int(-SEARCH_OFFSET_RANGE_PT // SEARCH_OFFSET_STEP_PT), int(SEARCH_OFFSET_RANGE_PT // SEARCH_OFFSET_STEP_PT) + 1)]
    for dx in offsets:
        for dy in offsets:
            wx, wy = cx + dx, cy + dy
            for half in SEARCH_WINDOW_HALF_SIZES_PT:
                bbox = (wx - half, wy - half, wx + half, wy + half)
                if _overlap_fraction(bbox, text_native_bbox) >= TEXT_OVERLAP_REJECT_FRACTION:
                    continue
                fp = compute_structural_fingerprint(index, bbox)
                text_ratio = compute_text_area_ratio(bbox, text_spans_native)
                gate = evaluate_graphical_symbol(fp, text_ratio)
                if not gate.passes:
                    continue
                d = _dist((wx, wy), (cx, cy))
                if best is None or d < best[0]:
                    best = (d, bbox)
    if best is not None:
        return best[1], "geometry_gate_match"

    # Fallback: a fixed padded box immediately around the text label itself
    # -- still a reviewable starting point, explicitly flagged as weaker.
    x0, y0, x1, y1 = text_native_bbox
    pad = 20.0
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad), "text_fallback_no_geometry_match"


def _plan_style_families() -> dict[str, str]:
    fam: dict[str, str] = {}
    for e in json.loads(LEGEND_SYMBOLS_PATH.read_text()):
        fam[e["plan_id_pseudonymous"]] = e["style_family"]
    for c in json.loads(MANIFEST_PATH.read_text()):
        fam.setdefault(c["plan_id_pseudonymous"], c["style_family"])
    return fam


def main() -> None:
    style_family_by_plan = _plan_style_families()
    OUT_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    stats_by_class: dict[str, dict] = defaultdict(lambda: {"occurrences": 0, "geometry_matched": 0, "legend_only_excluded": 0})

    for plan_id in PLAN_IDS:
        pdf_path = RAW_DIR / f"{plan_id}.pdf"
        if not pdf_path.exists():
            continue
        pdf = pymupdf.open(str(pdf_path))
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            page_number = page_index + 1
            rotation_matrix = page.rotation_matrix if page.rotation else None
            spans_native = [s for s in extract_native_text_spans(page) if s.text.strip()]

            drawings = page.get_drawings()
            index = build_page_index(drawings, page.rect.width, page.rect.height)
            legend_zones_native = _legend_zones_native(page_number, spans_native, rotation_matrix)

            for class_name, patterns in CLASSES.items():
                all_clusters = _find_text_clusters(plan_id, page_number, spans_native, patterns)
                for cluster in all_clusters:
                    if _center_inside_any(cluster.native_bbox, legend_zones_native):
                        stats_by_class[class_name]["legend_only_excluded"] += 1
                        continue
                    proposed_native, bbox_source = _propose_bbox(index, cluster.native_bbox, spans_native)

                    text_display = _to_display(cluster.native_bbox, rotation_matrix)
                    proposed_display = _to_display(proposed_native, rotation_matrix)

                    cid = _stable_id(plan_id, str(page_number), class_name, ",".join(f"{v:.1f}" for v in cluster.native_bbox))

                    cx, cy = _center(proposed_display)
                    clip = pymupdf.Rect(
                        cx - CONTEXT_HALF_SIDE_PT, cy - CONTEXT_HALF_SIDE_PT,
                        cx + CONTEXT_HALF_SIDE_PT, cy + CONTEXT_HALF_SIDE_PT,
                    ) & page.rect
                    zoom = RENDER_DPI / 72.0
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
                    crop_path = OUT_CROPS_DIR / f"{cid}.png"
                    pixmap.save(str(crop_path))

                    px0, py0, px1, py1 = proposed_display
                    bbox_in_crop_px = (
                        round((px0 - clip.x0) * _RENDER_DPI_SCALE, 1), round((py0 - clip.y0) * _RENDER_DPI_SCALE, 1),
                        round((px1 - clip.x0) * _RENDER_DPI_SCALE, 1), round((py1 - clip.y0) * _RENDER_DPI_SCALE, 1),
                    )

                    # Page-space coordinate mapping for this crop (pt-per-pixel
                    # is constant/uniform since crops are rendered with a plain
                    # zoom matrix, no rotation within the crop itself) -- lets
                    # the review tool convert any pixel position a human draws
                    # on the rendered crop back into ORIGINAL PAGE COORDINATES:
                    # page_x = crop_page_space_origin[0] + px * crop_pt_per_px.
                    pt_per_px = 1.0 / _RENDER_DPI_SCALE
                    crop_page_space_origin = (
                        round(px0 - bbox_in_crop_px[0] * pt_per_px, 3),
                        round(py0 - bbox_in_crop_px[1] * pt_per_px, 3),
                    )

                    candidates.append({
                        "occurrence_id": cid,
                        "target_class": class_name,
                        "plan_id": plan_id,
                        "style_family": style_family_by_plan.get(plan_id),
                        "page": page_number,
                        "source_text": sorted(set(cluster.texts)),
                        "text_bbox_page_space": list(text_display),
                        "proposed_bbox_page_space": list(proposed_display),
                        "bbox_in_crop_px": list(bbox_in_crop_px),
                        "bbox_source": bbox_source,
                        "crop_path": str(crop_path.relative_to(REPO_ROOT)),
                        "crop_page_space_origin": list(crop_page_space_origin),
                        "crop_pt_per_px": pt_per_px,
                    })
                    stats_by_class[class_name]["occurrences"] += 1
                    if bbox_source == "geometry_gate_match":
                        stats_by_class[class_name]["geometry_matched"] += 1
        pdf.close()

    manifest_out = {
        "target_classes": sorted(CLASSES),
        "total_candidates": len(candidates),
        "by_class": {
            cls: {
                "occurrences": stats_by_class[cls]["occurrences"],
                "geometry_matched": stats_by_class[cls]["geometry_matched"],
                "legend_only_excluded": stats_by_class[cls]["legend_only_excluded"],
                "plans": sorted({c["plan_id"] for c in candidates if c["target_class"] == cls}),
                "style_families": sorted({c["style_family"] for c in candidates if c["target_class"] == cls and c["style_family"]}),
            }
            for cls in sorted(CLASSES)
        },
        "candidates": candidates,
    }
    out_path = OUT_DIR / "candidates_manifest.json"
    out_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False, sort_keys=True))

    print(f"Total candidates: {len(candidates)}")
    for cls, d in manifest_out["by_class"].items():
        print(f"  {cls:30s} occurrences={d['occurrences']:3d} geometry_matched={d['geometry_matched']:3d} "
              f"legend_only_excluded={d['legend_only_excluded']:3d} "
              f"plans={len(d['plans']):2d} styles={len(d['style_families']):2d}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
