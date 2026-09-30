"""DETECTOR v2 -- project/style-family separated train/validation/test split
design, for the 5 targeted classes' candidate occurrences.

PRELIMINARY, based on candidate PRESENCE (any of the ~113 proposed
occurrences, regardless of human verification outcome yet) rather than
confirmed positives, because human review has not happened yet at the time
this is first run -- see docs/v04-detector-poc-phase0-report.md and
scripts/train_detector_v1.py for why this matters: a naive single
`design_split` call over ALL 20 plans routes the largest (and here, most
class-relevant) families entirely into "train" (greedy largest-family-first
fills train's initially-empty, furthest-below-target quota), starving
validation/test of real positive coverage. The SAME fix used for Detector
v1 is reused here, unmodified in spirit: split is designed once over the
plans/families that actually carry candidates for EACH class, so every
class gets a real held-out family wherever more than one exists for it.

RE-RUN THIS after human review completes, once it is known which
occurrences are actually verified positive (RICHTIG / BOX KORRIGIEREN) --
the split assignment may change once it is based on confirmed positives
instead of raw candidate presence. Never re-derive it by hand; always
through this same deterministic function so it stays auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.dataset_pipeline.splits import design_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v2_candidates" / "candidates_manifest.json"
OUT_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v2_candidates" / "split_design.json"

SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def main() -> None:
    manifest = json.loads(CANDIDATES_PATH.read_text())
    candidates = manifest["candidates"]

    per_class_split = {}
    for class_name in manifest["target_classes"]:
        family_of = {
            c["plan_id"]: c["style_family"]
            for c in candidates if c["target_class"] == class_name and c["style_family"]
        }
        if not family_of:
            per_class_split[class_name] = {"note": "no candidates -- cannot design a split"}
            continue
        assignment = design_split(family_of, ratios=SPLIT_RATIOS)
        per_class_split[class_name] = assignment.to_dict()

    OUT_PATH.write_text(json.dumps({
        "basis": "candidate presence (pre-review) -- see module docstring; re-run after human review",
        "split_ratios": SPLIT_RATIOS,
        "per_class": per_class_split,
    }, indent=2, ensure_ascii=False, sort_keys=True))

    for class_name, assignment in per_class_split.items():
        if "note" in assignment:
            print(f"{class_name}: {assignment['note']}")
            continue
        print(f"{class_name}: {assignment['counts']}  family_to_split={assignment['family_to_split']}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
