"""v0.4 Phase 8 -- detector-ready export format. Produces a manifest and a
class list now, so a future YOLO (or other detector) export can run without
re-annotating anything -- but writes no label files for anything not yet
verified by a human, and never invokes any training.

EXPORT PHILOSOPHY (task requirement, explicit): "do not automatically treat
unverified matches as training truth." An engine SUGGESTION
(`plan_specific_suggestion`/`generic_suggestion`) is carried in the
manifest purely as provenance/context -- the `class` field a future
training step would actually read is always `assigned_class`, which is
None until a human sets it via the annotation tool. `verification_status`
mirrors the annotation tool's own status field (`UNLABELED` /
`LABELED` / `SKIPPED`) so a training step can filter to human-verified rows
only, by construction.
"""
from __future__ import annotations

import json
import os


def write_manifest(candidates: list, family_of: dict, split_of: dict, out_path: str) -> dict:
    """candidates: list[AnnotationCandidate]. family_of/split_of:
    {plan_id: value}, from families.py/splits.py. Writes one JSON manifest
    (a list of records) and returns it."""
    records = []
    for c in candidates:
        d = c.to_dict()
        records.append({
            "candidate_id": d["candidate_id"],
            "plan_id_pseudonymous": d["plan_id"],
            "style_family": family_of.get(d["plan_id"]),
            "split": split_of.get(d["plan_id"]),
            "page": d["page"],
            "crop_path": d["crop_path"],
            "bbox_page_space": d["bbox"],
            "crop_bbox_page_space": d["crop_bbox"],
            "bbox_in_crop_px": d["bbox_in_crop_px"],
            "corrected_bbox_in_crop_px": d["corrected_bbox"],
            "class": d["assigned_class"],
            "verification_status": d["status"],
            "graph_association": d["graph_association"],
            "engine_suggestions": {
                "plan_specific": d["plan_specific_suggestion"],
                "generic": d["generic_suggestion"],
            },
            "annotator_note": d["annotator_note"],
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return {"path": out_path, "record_count": len(records)}


def write_class_list(classes: list, out_path: str) -> dict:
    """classes: ordered list of class-name strings (taxonomy.py). Writes
    {name: index} so a later YOLO export has a stable, reviewable mapping
    (index order never silently changes across re-exports of the same
    `classes` list -- callers must pass the SAME list, not a re-derived one,
    to keep pre-existing annotations' class indices valid)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mapping = {name: i for i, name in enumerate(classes)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    return mapping


def export_yolo_labels(manifest_path: str, class_list_path: str, crops_dir: str, out_dir: str) -> dict:
    """Converts only VERIFIED, non-ambiguous, real-component rows
    (`verification_status == "LABELED"` and class not in
    {"NOT_A_COMPONENT", "AMBIGUOUS"}) into YOLO-format .txt label files
    (one per crop image, normalized center-x/y/w/h). Safe to call at any
    point -- with zero labeled rows (a fresh dataset) it writes zero files
    and reports that honestly, rather than fabricating placeholder labels.
    Does not train anything; this is a format conversion only."""
    with open(manifest_path, encoding="utf-8") as f:
        records = json.load(f)
    with open(class_list_path, encoding="utf-8") as f:
        class_index = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    skipped_unverified = 0
    for rec in records:
        if rec["verification_status"] != "LABELED":
            skipped_unverified += 1
            continue
        cls = rec["class"]
        if cls in (None, "NOT_A_COMPONENT", "AMBIGUOUS") or cls not in class_index:
            continue
        bbox = rec["corrected_bbox_in_crop_px"] or rec["bbox_in_crop_px"]
        if bbox is None or not rec["crop_path"] or not os.path.exists(rec["crop_path"]):
            continue
        import cv2
        img = cv2.imread(rec["crop_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        label_path = os.path.join(out_dir, os.path.splitext(os.path.basename(rec["crop_path"]))[0] + ".txt")
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write(f"{class_index[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        written += 1

    return {"labels_written": written, "skipped_unverified": skipped_unverified, "out_dir": out_dir}
