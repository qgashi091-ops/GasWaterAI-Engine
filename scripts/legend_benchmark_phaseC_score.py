"""Phase C of the legend + vector-geometry structural benchmark: score
Phase A/B's FROZEN predictions against the 387 real human annotations.

This is the ONLY script in this benchmark that reads human annotation
content, and it may only run after Phase B has already hashed and frozen
predictions.json -- enforced here by re-verifying that hash before doing
anything else.

Usage:
  python3 scripts/legend_benchmark_phaseC_score.py <fresh_annotation_dump_dir>

<fresh_annotation_dump_dir> is a directory of one JSON file per annotation
document (id.json, containing {class, verification_status, annotator_note,
updated_at}), as produced by `ArtifactData`'s `list` action with `out_dir`
against the hosted tool's `annotations` collection -- fetched AFTER Phase B
froze predictions.json, never before.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "legend_structural_benchmark"
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"

# The original v0.4 dataset (app/dataset_pipeline/candidates.py) stores
# `bbox_page_space` in DISPLAY space (via page.rotation_matrix -- identity
# when rotation==0), so it lines up with the rendered crop pixels; v0.1's
# own `schema.SymbolCandidate.bbox` (what this benchmark's Phase A emits)
# is in the PDF's raw, UNROTATED space. For an unrotated page the two
# coincide; for a rotated one (DEV-03, rotation=90 in this selection) they
# do not, and naively comparing them drops every real match on that plan.
# This transform is applied ONLY to join Phase A's already-frozen bboxes
# back to their original candidate_id for scoring -- predictions.json
# itself is never re-touched, and the transform is a deterministic function
# of (plan, page, rotation) alone, never of any label value.


def _rotation_matrices(selected_plans: list[str]) -> dict:
    out = {}
    for plan_id in selected_plans:
        pdf = pymupdf.open(str(RAW_DIR / f"{plan_id}.pdf"))
        for i in range(len(pdf)):
            page = pdf[i]
            out[(plan_id, i + 1)] = page.rotation_matrix if page.rotation else None
        pdf.close()
    return out


def _to_display(bbox: tuple, rotation_matrix) -> tuple:
    if rotation_matrix is None:
        return tuple(bbox)
    r = pymupdf.Rect(*bbox) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)

TAXONOMY_CLASSES = [
    "pex-verteiler", "dusche", "küche", "up verteiler unter wt", "BA",
    "secomat", "wc up", "OTHER_RELEVANT_SYMBOL", "NOT_A_COMPONENT", "AMBIGUOUS",
]
NON_COMPONENT_CLASSES = {"NOT_A_COMPONENT", "AMBIGUOUS"}


def verify_freeze() -> dict:
    with open(OUT_DIR / "freeze_manifest.json", encoding="utf-8") as f:
        freeze = json.load(f)
    raw = (OUT_DIR / "predictions.json").read_bytes()
    actual = sha256(raw).hexdigest()
    if actual != freeze["predictions_sha256"]:
        raise SystemExit(
            f"INTEGRITY FAILURE: predictions.json has changed since freeze! "
            f"expected {freeze['predictions_sha256']}, got {actual}"
        )
    print(f"Freeze verified OK: predictions.json SHA-256 matches {actual}")
    return freeze


def load_predictions() -> dict:
    with open(OUT_DIR / "predictions.json", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_bbox_index(selected_plans: list[str]) -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    index = {}
    by_candidate_id = {}
    for r in manifest:
        if r["plan_id_pseudonymous"] not in selected_plans:
            continue
        key = (r["plan_id_pseudonymous"], r["page"], _round_bbox(r["bbox_page_space"]))
        index[key] = r["candidate_id"]
        by_candidate_id[r["candidate_id"]] = r
    return index, by_candidate_id


def _round_bbox(bbox, ndigits: int = 1) -> tuple:
    return tuple(round(v, ndigits) for v in bbox)


def load_fresh_annotations(dump_dir: str) -> dict:
    out = {}
    for fn in os.listdir(dump_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(dump_dir, fn), encoding="utf-8") as f:
            doc = json.load(f)
        out[fn[:-5]] = doc
    return out


def predicted_class_for(match: dict) -> str:
    status = match["status"]
    if status in ("COMPONENT_FACT", "COMPONENT_CANDIDATE"):
        return match["component_type"] or "OTHER_RELEVANT_SYMBOL"
    if status == "UNRESOLVED":
        return "UNRESOLVED"
    return "EXCLUDED"  # any EXCLUDED_* status


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <fresh_annotation_dump_dir>")
    dump_dir = sys.argv[1]

    freeze = verify_freeze()
    predictions = load_predictions()
    selected_plans = predictions["selected_plans"]
    bbox_index, manifest_by_id = load_manifest_bbox_index(selected_plans)
    rotation_matrices = _rotation_matrices(selected_plans)
    annotations = load_fresh_annotations(dump_dir)
    print(f"Loaded {len(annotations)} annotation documents (fetched after freeze).")

    # Build stable_match_id -> match record dict, and try to join every
    # matches record back to its ORIGINAL v0.4 candidate_id via bbox.
    joined = []  # list of dicts: candidate_id, plan_id, gt_class, gt_status, pred_class, match
    not_found_in_rerun = []  # manifest candidates (in the 5 plans) with no corresponding Phase A record at all
    matched_candidate_ids: set[str] = set()

    for plan_result in predictions["plan_results"]:
        plan_id = plan_result["plan_id"]
        for match in plan_result["matches"]:
            page = match["page"]
            rot = rotation_matrices.get((plan_id, page))
            display_bbox = _to_display(match["bbox"], rot)
            key = (plan_id, page, _round_bbox(display_bbox))
            candidate_id = bbox_index.get(key)
            if candidate_id is None:
                continue  # a symbol v0.1 found on this fresh rerun with no counterpart in the original manifest
            matched_candidate_ids.add(candidate_id)
            ann = annotations.get(candidate_id)
            if ann is None or ann.get("verification_status") != "LABELED":
                continue  # no human ground truth for this one yet -- not scored, per task instruction
            joined.append({
                "candidate_id": candidate_id,
                "plan_id": plan_id,
                "gt_class": ann["class"],
                "pred_class": predicted_class_for(match),
                "status": match["status"],
                "exclusion_reason": match.get("exclusion_reason"),
                "structural_score": match.get("structural_score"),
            })

    # Any LABELED candidate in the 5 plans that never showed up in Phase A's output at all
    for candidate_id, r in manifest_by_id.items():
        ann = annotations.get(candidate_id)
        if ann is None or ann.get("verification_status") != "LABELED":
            continue
        if candidate_id not in matched_candidate_ids:
            not_found_in_rerun.append({"candidate_id": candidate_id, "plan_id": r["plan_id_pseudonymous"], "gt_class": ann["class"]})

    print(f"Scorable (LABELED + matched to a Phase A record): {len(joined)}")
    print(f"LABELED but NOT reproduced by the fresh v0.1 rerun at all: {len(not_found_in_rerun)}")

    # ---- per-class precision/recall/F1 ----
    all_classes = sorted(set(TAXONOMY_CLASSES) | {"UNRESOLVED", "EXCLUDED"})
    tp = Counter()
    fp = Counter()
    fn = Counter()
    confusion = defaultdict(Counter)  # gt_class -> Counter(pred_class)

    for j in joined:
        gt, pred = j["gt_class"], j["pred_class"]
        confusion[gt][pred] += 1
        if gt == pred:
            tp[gt] += 1
        else:
            fn[gt] += 1
            fp[pred] += 1

    per_class = {}
    for c in all_classes:
        support = tp[c] + fn[c]
        if support == 0 and (tp[c] + fp[c]) == 0:
            continue
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else None
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
        per_class[c] = {
            "support": support, "tp": tp[c], "fp": fp[c], "fn": fn[c],
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    total_tp = sum(tp.values())
    total = len(joined)
    overall_accuracy = round(total_tp / total, 4) if total else None
    macro_p = [v["precision"] for v in per_class.values() if v["precision"] is not None]
    macro_r = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    macro_f1 = [v["f1"] for v in per_class.values() if v["f1"] is not None]

    # ---- the task's own named buckets ----
    component_correct = sum(1 for j in joined if j["gt_class"] not in NON_COMPONENT_CLASSES and j["gt_class"] not in ("OTHER_RELEVANT_SYMBOL",) and j["pred_class"] == j["gt_class"])
    wrong_component_class = sum(1 for j in joined if j["gt_class"] not in NON_COMPONENT_CLASSES and j["gt_class"] != "OTHER_RELEVANT_SYMBOL" and j["pred_class"] not in NON_COMPONENT_CLASSES and j["pred_class"] not in ("UNRESOLVED", "EXCLUDED", "OTHER_RELEVANT_SYMBOL") and j["pred_class"] != j["gt_class"])
    missed_relevant_symbol = sum(1 for j in joined if j["gt_class"] not in NON_COMPONENT_CLASSES and j["pred_class"] in ("UNRESOLVED", "EXCLUDED"))
    false_positive = sum(1 for j in joined if j["gt_class"] in NON_COMPONENT_CLASSES and j["pred_class"] not in NON_COMPONENT_CLASSES and j["pred_class"] not in ("UNRESOLVED", "EXCLUDED"))
    correct_not_a_component = sum(1 for j in joined if j["gt_class"] == "NOT_A_COMPONENT" and (j["pred_class"] in NON_COMPONENT_CLASSES or j["pred_class"] in ("UNRESOLVED", "EXCLUDED")))
    ambiguous_gt = [j for j in joined if j["gt_class"] == "AMBIGUOUS"]
    other_relevant_gt = [j for j in joined if j["gt_class"] == "OTHER_RELEVANT_SYMBOL"]

    report = {
        "freeze_verification": {"predictions_sha256": freeze["predictions_sha256"], "verified": True},
        "annotations_loaded": len(annotations),
        "scorable_candidates": total,
        "not_found_in_rerun_count": len(not_found_in_rerun),
        "not_found_in_rerun": not_found_in_rerun,
        "overall_accuracy": overall_accuracy,
        "macro_precision": round(sum(macro_p) / len(macro_p), 4) if macro_p else None,
        "macro_recall": round(sum(macro_r) / len(macro_r), 4) if macro_r else None,
        "macro_f1": round(sum(macro_f1) / len(macro_f1), 4) if macro_f1 else None,
        "per_class": per_class,
        "confusion_matrix": {gt: dict(preds) for gt, preds in confusion.items()},
        "named_buckets": {
            "component_classes_correct": component_correct,
            "wrong_component_class": wrong_component_class,
            "missed_relevant_symbol": missed_relevant_symbol,
            "false_positive": false_positive,
            "correct_not_a_component": correct_not_a_component,
            "ambiguous_gt_count": len(ambiguous_gt),
            "ambiguous_gt_predicted_correctly": sum(1 for j in ambiguous_gt if j["pred_class"] in ("UNRESOLVED", "EXCLUDED", "AMBIGUOUS")),
            "other_relevant_symbol_gt_count": len(other_relevant_gt),
            "other_relevant_symbol_gt_predicted_correctly": sum(1 for j in other_relevant_gt if j["pred_class"] == "OTHER_RELEVANT_SYMBOL"),
        },
        "per_plan_scorable_counts": dict(Counter(j["plan_id"] for j in joined)),
        "gt_class_distribution_scored": dict(Counter(j["gt_class"] for j in joined)),
        "detail": joined,
    }

    out_path = OUT_DIR / "phaseC_score_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in report.items() if k != "detail"}, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
