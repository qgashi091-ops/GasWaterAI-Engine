"""Phase A of the legend + vector-geometry structural benchmark: produce
predictions for the 5 selected dev plans WITHOUT ever reading a human
annotation.

HOLDOUT ENFORCEMENT: this script imports only `app.plan_analysis.*` (v0.1's
existing parser) and `app.legend_structural_benchmark.*` (this benchmark's
own new code). Neither import graph contains `ArtifactData`, any hosted
Artifact client, or any read of a `class`/`verification_status` field --
grep-verifiable (see `docs/v04-legend-structural-benchmark-report.md`
section 2, which records the literal grep command and its empty result).
`app.legend_structural_benchmark.select_plans` IS imported (for the plan
list), and that module itself also never reads label content -- see its
own docstring.

Usage: python3 scripts/legend_benchmark_phaseA_predict.py
Writes: data/dev_plans_v04/legend_structural_benchmark/predictions.json
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.legend_structural_benchmark.legend_extraction import extract_plan_legend
from app.legend_structural_benchmark.matching import match_page
from app.legend_structural_benchmark.select_plans import select_five_plans
from app.legend_structural_benchmark.vector_features import build_page_index
from app.plan_analysis.legend_detection import detect_legend_candidates
from app.plan_analysis.pipeline import analyze_pdf_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "legend_structural_benchmark"


def run_plan(plan_id: str) -> dict:
    pdf_path = str(RAW_DIR / f"{plan_id}.pdf")
    t0 = time.monotonic()
    doc = analyze_pdf_file(pdf_path)
    plan_legend = extract_plan_legend(plan_id, pdf_path, doc.pages)

    all_match_records = []
    pdf = pymupdf.open(pdf_path)
    try:
        for page_analysis in doc.pages:
            page = pdf[page_analysis.page_number - 1]
            rotation_matrix = page.rotation_matrix if page.rotation else None
            inverse_rotation_matrix = ~rotation_matrix if rotation_matrix else None

            def _to_raw(bbox: tuple) -> tuple:
                if inverse_rotation_matrix is None:
                    return bbox
                r = pymupdf.Rect(*bbox) * inverse_rotation_matrix
                return (r.x0, r.y0, r.x1, r.y1)

            all_legend_candidates = detect_legend_candidates(page_analysis, rotation_matrix=rotation_matrix)
            # detect_legend_candidates returns DISPLAY-space bboxes (see its
            # own docstring); match_page's exclusion check compares against
            # v0.1 symbol bboxes, which are in the PDF's raw/native space --
            # convert here, or every excluded-region check silently misses
            # on a rotated page (see legend_extraction.py's `_to_raw` for
            # the same fix applied to legend entries).
            other_dense_regions = [_to_raw(c.bbox) for c in all_legend_candidates if c.legend_id != plan_legend.legend_id]

            drawings = page.get_drawings()
            index = build_page_index(drawings, page.rect.width, page.rect.height)
            records = match_page(plan_id, page_analysis, index, plan_legend, other_dense_regions)
            all_match_records.extend(records)
    finally:
        pdf.close()

    elapsed = time.monotonic() - t0
    return {
        "plan_id": plan_id,
        "legend": plan_legend.to_dict(),
        "matches": [r.to_dict() for r in all_match_records],
        "elapsed_seconds": round(elapsed, 3),
        "page_count": doc.page_count,
        "total_symbols_detected": sum(len(p.symbols) for p in doc.pages),
    }


def main() -> None:
    selection = select_five_plans()
    selected_plans = selection["selected_plans"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "selected_plans.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    results = []
    t_start = time.monotonic()
    for plan_id in selected_plans:
        print(f"Running plan {plan_id}...", flush=True)
        result = run_plan(plan_id)
        results.append(result)
        status_counts: dict[str, int] = {}
        for m in result["matches"]:
            status_counts[m["status"]] = status_counts.get(m["status"], 0) + 1
        print(f"  {plan_id}: {result['elapsed_seconds']}s, {len(result['matches'])} candidates, {status_counts}")

    total_elapsed = time.monotonic() - t_start
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    predictions = {
        "benchmark": "legend_structural_v1",
        "selected_plans": selected_plans,
        "plan_results": results,
        "performance": {
            "total_elapsed_seconds": round(total_elapsed, 3),
            "peak_rss_kb": peak_rss_kb,
            "peak_rss_mib": round(peak_rss_kb / 1024, 1),
        },
    }

    out_path = OUT_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"\nWrote {out_path}")
    print(f"Total: {total_elapsed:.1f}s, peak RSS {peak_rss_kb/1024:.1f} MiB")


if __name__ == "__main__":
    main()
