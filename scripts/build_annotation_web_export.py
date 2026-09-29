"""v0.4 Phase 6b -- prepares a slim, web-ready export of the ALREADY-GENERATED,
already privacy-checked annotation dataset for the hosted browser annotation
tool (a Claude Artifact backed by a persistent server-side database).

Does NOT re-process any PDF and does NOT generate any new candidate -- it
only reshapes the existing manifest.json (837 candidates, unchanged) for a
web page that has no local filesystem to read crop_path from. Run this
AFTER scripts/run_v04_pipeline.py, never instead of it.

What changes vs. manifest.json, and why:
  - crop_path (a local filesystem path, meaningless in a browser) is
    replaced by crop_filename (its basename only) -- the actual image bytes
    are uploaded separately, once per crop, to the artifact's own asset
    store, and the resulting asset URL is stitched back in afterward
    (see docs/v04-annotation-web-deployment.md).
  - bbox_page_space / crop_bbox_page_space are dropped: they only matter
    for live-rendering a wider view from the RAW PDF, which the hosted
    version deliberately never has access to (raw PDFs are never
    published -- see that same doc for why "larger context" is limited
    there).
  - class / verification_status / annotator_note / corrected_bbox_in_crop_px
    are dropped: those become per-viewer, server-side database rows in the
    hosted tool, not static page content -- keeping them out of the
    published page is also what lets the SAME 837-candidate export be
    reused as multiple people annotate without ever re-publishing the page.

Usage: python3 scripts/build_annotation_web_export.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dev_plans_v04" / "annotation_dataset"
OUT_DIR = ROOT / "data" / "dev_plans_v04" / "annotation_web_export"


def main() -> None:
    with open(DATASET_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(DATASET_DIR / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    records = []
    for rec in manifest:
        records.append({
            "candidate_id": rec["candidate_id"],
            "plan_id_pseudonymous": rec["plan_id_pseudonymous"],
            "style_family": rec["style_family"],
            "split": rec["split"],
            "page": rec["page"],
            "crop_filename": Path(rec["crop_path"]).name if rec["crop_path"] else None,
            "bbox_in_crop_px": rec["bbox_in_crop_px"],
            "graph_association": rec["graph_association"],
            "engine_suggestions": rec["engine_suggestions"],
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "candidates.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {OUT_DIR / 'candidates.json'}")
    print(f"Wrote {len(classes)} classes to {OUT_DIR / 'classes.json'}")
    print("Next step (done once, interactively, by Claude): upload each crop_filename's "
          "PNG from annotation_dataset/crops/ to the hosted page's asset store, then stitch "
          "the returned asset URL into each record's own 'asset_url' field before publishing.")


if __name__ == "__main__":
    main()
