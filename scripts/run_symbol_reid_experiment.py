"""ONE-PLAN SYMBOL REIDENTIFICATION EXPERIMENT -- orchestration script.

Picks DEV-03 (its entire legend is 7 entries, all 7 already confirmed by
independent human review as real potable-water component symbols -- see
data/dev_plans_v04/legend_symbol_dataset/gate_report.json /
reviews_snapshot.json), selects 5 of them, builds a structural template for
each from its OWN legend geometry, and searches the entire page for
occurrences using nothing but that geometry (see app/symbol_reid_experiment/
search.py for exactly how text/pipe/table/dimension regions are excluded).

Writes (before any manual inspection of the plan happens):
  data/dev_plans_v04/symbol_reid_experiment/predictions.json
  data/dev_plans_v04/symbol_reid_experiment/crops/<symbol>_legend.png
  data/dev_plans_v04/symbol_reid_experiment/crops/<symbol>_occ<N>.png
  data/dev_plans_v04/symbol_reid_experiment/contact_sheet.png

Run with --verify-determinism to run the whole search twice and diff the
frozen predictions (byte-for-byte on every geometric field) before writing
anything -- confirms this is pure, repeatable computation with no hidden
randomness or ordering dependency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.legend_structural_benchmark.vector_features import build_page_index
from app.plan_analysis.pipeline import analyze_pdf_file
from app.symbol_reid_experiment.search import search_page_for_templates
from app.symbol_reid_experiment.template import build_template

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"
LEGEND_DATASET_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "legend_symbol_dataset"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "symbol_reid_experiment"
CROPS_DIR = OUT_DIR / "crops"

PLAN_ID = "DEV-03"
# 5 of DEV-03's 7 legend entries, chosen for diversity across the
# requested examples (meter / filter-family / check-valve / shutoff-valve /
# safety-valve); all 5 already independently confirmed real by human review
# (see reviews_snapshot.json) -- selection made BEFORE any search ran.
SELECTED_IDS = [
    "LEd227fe0fe8abf393c705",  # Wasserzaehler M 3
    "LEe5e804f3120845578f2a",  # Rueckschlagventil
    "LE726b190a783302b66cbd",  # Absperrklappe
    "LE3f176c2478036418ce65",  # Sicherheitsventil
    "LEfbcca8dacc47835fa457",  # Rueckschlagklappe
]

RENDER_DPI = 300.0
CROP_MARGIN_PT = 6.0
CONTACT_ROW_HEIGHT = 180
CONTACT_LEGEND_COL_WIDTH = 200
CONTACT_OCC_CELL = 150
CONTACT_MAX_OCC_PER_ROW = 6


def _to_native(bbox_display, inverse_rotation_matrix):
    if inverse_rotation_matrix is None:
        return tuple(bbox_display)
    r = pymupdf.Rect(*bbox_display) * inverse_rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _to_display(bbox_native, rotation_matrix):
    if rotation_matrix is None:
        return tuple(bbox_native)
    r = pymupdf.Rect(*bbox_native) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _render_native_bbox(page, bbox_native, rotation_matrix, margin=CROP_MARGIN_PT):
    disp = _to_display(bbox_native, rotation_matrix)
    clip = pymupdf.Rect(disp[0] - margin, disp[1] - margin, disp[2] + margin, disp[3] + margin) & page.rect
    if clip.is_empty or clip.width < 1 or clip.height < 1:
        return None
    zoom = RENDER_DPI / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n).copy()


def _bbox_overlap_fraction(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a


def run_experiment() -> dict:
    with open(LEGEND_DATASET_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = {e["legend_symbol_id"]: e for e in json.load(f)}

    pdf_path = str(RAW_DIR / f"{PLAN_ID}.pdf")
    doc = analyze_pdf_file(pdf_path)
    assert len(doc.pages) == 1, f"expected DEV-03 to be single-page, found {len(doc.pages)}"
    page_analysis = doc.pages[0]
    text_spans_native = page_analysis.text_spans

    pdf = pymupdf.open(pdf_path)
    try:
        page = pdf[0]
        rotation_matrix = page.rotation_matrix if page.rotation else None
        inverse_rotation_matrix = ~rotation_matrix if rotation_matrix else None
        drawings = page.get_drawings()
        index = build_page_index(drawings, page.rect.width, page.rect.height)

        templates = []
        entries_by_id = {}
        for legend_symbol_id in SELECTED_IDS:
            entry = manifest[legend_symbol_id]
            assert entry["plan_id_pseudonymous"] == PLAN_ID
            native_bbox = _to_native(entry["bbox"], inverse_rotation_matrix)
            template = build_template(
                legend_symbol_id, entry["raw_legend_text"], native_bbox, index, text_spans_native
            )
            templates.append(template)
            entries_by_id[legend_symbol_id] = entry

        occurrences_by_id = search_page_for_templates(index, page.rect, templates, text_spans_native)

        results = []
        for template in templates:
            entry = entries_by_id[template.legend_symbol_id]
            occurrences = occurrences_by_id[template.legend_symbol_id]
            # Drop any candidate window that is really just the legend
            # symbol's OWN footprint being rediscovered by the scan.
            occurrences = [
                occ for occ in occurrences
                if _bbox_overlap_fraction(occ.bbox, template.native_bbox) < 0.5
                and _bbox_overlap_fraction(template.native_bbox, occ.bbox) < 0.5
            ]
            results.append({
                "legend_symbol_id": template.legend_symbol_id,
                "component_name_from_legend": entry["raw_legend_text"],
                "legend_native_bbox": list(template.native_bbox),
                "legend_fingerprint": template.fingerprint.to_dict(),
                "occurrences": [occ.to_dict() for occ in occurrences],
            })

        return {
            "plan_id_pseudonymous": PLAN_ID,
            "page": 1,
            "page_rect": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
            "rotation": page.rotation,
            "results": results,
        }
    finally:
        pdf.close()


def render_outputs(frozen: dict) -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open(str(RAW_DIR / f"{PLAN_ID}.pdf"))
    page = pdf[0]
    rotation_matrix = page.rotation_matrix if page.rotation else None

    row_images = []
    for r in frozen["results"]:
        safe_name = "".join(c if c.isalnum() else "_" for c in r["component_name_from_legend"])[:40]
        legend_crop = _render_native_bbox(page, tuple(r["legend_native_bbox"]), rotation_matrix, margin=10.0)
        legend_path = CROPS_DIR / f"{safe_name}_legend.png"
        if legend_crop is not None:
            cv2.imwrite(str(legend_path), cv2.cvtColor(legend_crop, cv2.COLOR_RGB2BGR))

        occ_paths = []
        for i, occ in enumerate(r["occurrences"]):
            crop = _render_native_bbox(page, tuple(occ["bbox"]), rotation_matrix)
            if crop is None:
                continue
            p = CROPS_DIR / f"{safe_name}_occ{i}.png"
            cv2.imwrite(str(p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            occ_paths.append((p, occ["similarity"]["structural_score"]))

        row_images.append((r["component_name_from_legend"], legend_path if legend_crop is not None else None, occ_paths))

    pdf.close()

    n_rows = len(row_images)
    sheet_w = CONTACT_LEGEND_COL_WIDTH + CONTACT_OCC_CELL * CONTACT_MAX_OCC_PER_ROW
    sheet = np.full((CONTACT_ROW_HEIGHT * n_rows, sheet_w, 3), 255, dtype=np.uint8)
    for row_i, (name, legend_path, occ_paths) in enumerate(row_images):
        y0 = row_i * CONTACT_ROW_HEIGHT
        cv2.rectangle(sheet, (0, y0), (sheet_w - 1, y0 + CONTACT_ROW_HEIGHT - 1), (200, 200, 200), 1)
        cv2.putText(sheet, name[:28], (4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        if legend_path is not None:
            img = cv2.imread(str(legend_path))
            if img is not None:
                h, w = img.shape[:2]
                scale = min((CONTACT_LEGEND_COL_WIDTH - 10) / w, (CONTACT_ROW_HEIGHT - 24) / h, 1.0)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                resized = cv2.resize(img, (nw, nh))
                sheet[y0 + 20:y0 + 20 + nh, 4:4 + nw] = resized
        cv2.line(sheet, (CONTACT_LEGEND_COL_WIDTH, y0), (CONTACT_LEGEND_COL_WIDTH, y0 + CONTACT_ROW_HEIGHT), (150, 150, 150), 1)
        for occ_i, (p, score) in enumerate(occ_paths[:CONTACT_MAX_OCC_PER_ROW]):
            img = cv2.imread(str(p))
            if img is None:
                continue
            cx0 = CONTACT_LEGEND_COL_WIDTH + occ_i * CONTACT_OCC_CELL
            h, w = img.shape[:2]
            scale = min((CONTACT_OCC_CELL - 10) / w, (CONTACT_ROW_HEIGHT - 24) / h, 1.0)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (nw, nh))
            sheet[y0 + 20:y0 + 20 + nh, cx0 + 4:cx0 + 4 + nw] = resized
            cv2.putText(sheet, f"{score:.2f}", (cx0 + 4, y0 + CONTACT_ROW_HEIGHT - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "contact_sheet.png"), sheet)


def main() -> None:
    verify_determinism = "--verify-determinism" in sys.argv
    frozen = run_experiment()
    if verify_determinism:
        frozen_again = run_experiment()
        s1 = json.dumps(frozen, sort_keys=True)
        s2 = json.dumps(frozen_again, sort_keys=True)
        print(f"Determinism check: {'IDENTICAL' if s1 == s2 else 'DIFFERENT'} "
              f"({len(s1)} vs {len(s2)} bytes serialized)")
        if s1 != s2:
            raise SystemExit("Non-deterministic output detected -- STOPPING before publishing.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)

    for r in frozen["results"]:
        print(f"{r['component_name_from_legend']!r}: {len(r['occurrences'])} occurrence(s) detected "
              f"(scores: {[round(o['similarity']['structural_score'], 3) for o in r['occurrences']]})")

    render_outputs(frozen)
    print(f"\nWrote {OUT_DIR / 'predictions.json'}")
    print(f"Wrote {OUT_DIR / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
