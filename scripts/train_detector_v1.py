"""Detector v1 training -- ENGINE v1 epic, section 2 ("train detector now
with available data").

Trains ONLY the classes Phase 0 of the earlier Detector POC found eligible
(docs/v04-detector-poc-phase0-report.md): kueche and wc_up. Every other
class remains unsupported by Detector v1 -- not trained, not evaluated,
reported as such.

WHAT "DETECTOR" MEANS HERE: this is a classifier over v0.1's own
already-localized candidate regions (manifest.json's bbox_page_space per
candidate, produced deterministically by the existing vector-graph/symbol
pipeline -- frozen, unmodified), not a from-scratch bounding-box detector.
Localization already comes for free from the deterministic engine; what is
missing is TYPE identification for the 2 classes with enough evidence,
which is exactly what this classifier adds. This is the "smallest
practical" choice given the data size (see app/detector/features.py).

GROUND TRUTH: only `class` from the live `annotations` collection, with
`verification_status == LABELED`, counts. AMBIGUOUS is dropped entirely
(never positive, never negative, never background). NOT_A_COMPONENT is the
background/negative class. No automatically-proposed label is ever used.

SPLIT: family-level only, via app.dataset_pipeline.splits.design_split
(unchanged, reused as-is) over ALL 20 plans' style families -- never
plan-random, never candidate-random. A held-out family's crops are never
seen at all during training, for either class.

PINNED CONFIG (for reproducibility -- see REPRODUCIBILITY block below):
  SEED = 42
  model = sklearn.ensemble.RandomForestClassifier(
      n_estimators=200, max_depth=12, random_state=SEED, n_jobs=1)
  features = app.detector.features.extract_features (HOG + intensity grid)
  split_ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from app.dataset_pipeline.splits import design_split
from app.detector.features import load_and_extract

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v1"
MODEL_PATH = OUT_DIR / "model.joblib"
CONFIG_PATH = OUT_DIR / "training_config.json"
METRICS_PATH = OUT_DIR / "metrics.json"

SEED = 42
TARGET_CLASSES = ["kueche", "wc_up"]  # normalized from "küche"/"wc up" (ASCII, unambiguous 1:1 rename, original preserved as raw_class)
LABEL_TO_TARGET = {"küche": "kueche", "wc up": "wc_up"}
BACKGROUND_LABEL = "background"
DATASET_VERSION = "annotations_dump_2026-09-30_426docs"  # the exact dump this run was trained against (see report)

# class_weight="balanced" is a fixed, principled response to a fact already
# visible in the TRAINING split alone (295 background vs 16/9 positives,
# roughly 20:1) -- chosen once, before looking at any held-out result, not
# tuned against test/validation performance.
MODEL_CONFIG = dict(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=1, class_weight="balanced")
SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def _load_labels(dump_dir: Path) -> dict[str, dict]:
    out = {}
    for f in (dump_dir / "annotations" / "annotations").glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    return out


def build_dataset(dump_dir: Path) -> list[dict]:
    manifest = {c["candidate_id"]: c for c in json.loads(MANIFEST_PATH.read_text())}
    labels = _load_labels(dump_dir)

    rows = []
    for cid, doc in labels.items():
        if doc.get("verification_status") != "LABELED":
            continue
        raw_class = doc["class"]
        if raw_class == "AMBIGUOUS":
            continue
        cand = manifest.get(cid)
        if cand is None:
            continue
        target = BACKGROUND_LABEL if raw_class == "NOT_A_COMPONENT" else LABEL_TO_TARGET.get(raw_class)
        if target is None:
            continue  # a real positive class outside Detector v1's scope (e.g. dusche) -- not used for train or background
        rows.append({
            "candidate_id": cid,
            "raw_class": raw_class,
            "target": target,
            "plan_id": cand["plan_id_pseudonymous"],
            "style_family": cand["style_family"],
            "crop_path": cand["crop_path"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = build_dataset(args.dump_dir)

    # design_split() balances PLAN COUNTS project-wide -- correct for the
    # full 20-plan corpus, but with this dataset's label concentration
    # (the 3 data-rich families are also the LARGEST families) a single
    # global call systematically routes every family that carries a real
    # kueche/wc_up instance into "train" first (greedy largest-first fills
    # train's initially-empty, furthest-below-target quota), leaving 0
    # held-out kueche instances and 1 held-out wc_up instance -- not a
    # meaningful cross-style evaluation. Calling the SAME unmodified
    # function separately for (a) the families that actually carry a
    # target-class positive and (b) the background-only families forces a
    # genuine held-out family for each class while keeping design_split's
    # own logic untouched.
    positive_family_of = {r["plan_id"]: r["style_family"] for r in rows if r["target"] != BACKGROUND_LABEL}
    positive_plans = set(positive_family_of)
    background_only_family_of = {r["plan_id"]: r["style_family"] for r in rows if r["plan_id"] not in positive_plans}

    positive_split = design_split(positive_family_of, ratios=SPLIT_RATIOS)
    plan_to_split = dict(positive_split.plan_to_split)
    family_to_split = dict(positive_split.family_to_split)
    if background_only_family_of:
        bg_split = design_split(background_only_family_of, ratios=SPLIT_RATIOS)
        plan_to_split.update(bg_split.plan_to_split)
        family_to_split.update(bg_split.family_to_split)
    split_assignment = type(positive_split)(
        plan_to_split=plan_to_split, family_to_split=family_to_split,
        counts={s: sum(1 for p in plan_to_split.values() if p == s) for s in SPLIT_RATIOS},
    )

    for r in rows:
        r["split"] = split_assignment.plan_to_split[r["plan_id"]]

    features_by_id: dict[str, np.ndarray] = {}
    dropped_unreadable = []
    for r in rows:
        feats = load_and_extract(str(REPO_ROOT / r["crop_path"]) if not Path(r["crop_path"]).is_absolute() else r["crop_path"])
        if feats is None:
            dropped_unreadable.append(r["candidate_id"])
            continue
        features_by_id[r["candidate_id"]] = feats
    rows = [r for r in rows if r["candidate_id"] in features_by_id]

    def subset(split_name: str):
        sub = [r for r in rows if r["split"] == split_name]
        X = np.stack([features_by_id[r["candidate_id"]] for r in sub]) if sub else np.empty((0,))
        y = [r["target"] for r in sub]
        return sub, X, y

    train_rows, X_train, y_train = subset("train")
    val_rows, X_val, y_val = subset("validation")
    test_rows, X_test, y_test = subset("test")

    classes_present_in_train = sorted(set(y_train))
    model = RandomForestClassifier(**MODEL_CONFIG)
    model.fit(X_train, y_train)

    def evaluate(name, rows_sub, X, y_true):
        if not rows_sub:
            return {"n": 0, "note": "empty split for this family assignment"}
        y_pred = model.predict(X)
        report = classification_report(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)), output_dict=True, zero_division=0)
        cm_labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels).tolist()
        by_plan = sorted({r["plan_id"] for r in rows_sub})
        by_family = sorted({r["style_family"] for r in rows_sub})
        return {
            "n": len(rows_sub), "plans": by_plan, "style_families": by_family,
            "classification_report": report, "confusion_matrix": {"labels": cm_labels, "matrix": cm},
        }

    metrics = {
        "seed": SEED,
        "model_config": MODEL_CONFIG,
        "feature_config": {"image_size": 96, "descriptor": "HOG(16,16,8,8,9)+4x4 intensity grid"},
        "split_ratios": SPLIT_RATIOS,
        "dataset_version": DATASET_VERSION,
        "classes_present_in_train": classes_present_in_train,
        "target_classes": TARGET_CLASSES,
        "background_label": BACKGROUND_LABEL,
        "dropped_unreadable_crops": dropped_unreadable,
        "split_assignment": split_assignment.to_dict(),
        "counts_by_split_and_target": {
            split: {t: sum(1 for r in rows if r["split"] == split and r["target"] == t) for t in TARGET_CLASSES + [BACKGROUND_LABEL]}
            for split in ("train", "validation", "test")
        },
        "train": evaluate("train", train_rows, X_train, y_train),
        "validation": evaluate("validation", val_rows, X_val, y_val),
        "test": evaluate("test", test_rows, X_test, y_test),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    CONFIG_PATH.write_text(json.dumps({
        "seed": SEED, "model_config": MODEL_CONFIG, "split_ratios": SPLIT_RATIOS,
        "target_classes": TARGET_CLASSES, "background_label": BACKGROUND_LABEL,
        "dataset_version": DATASET_VERSION, "sklearn_model_class": "RandomForestClassifier",
    }, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"Train: {len(train_rows)}  Validation: {len(val_rows)}  Test: {len(test_rows)}")
    print("Counts by split/target:", json.dumps(metrics["counts_by_split_and_target"], indent=2))
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
