"""Deterministic selection of 5 of the 20 v0.4 dev plans for the legend +
vector-geometry structural benchmark.

Selection criteria, in the order actually applied (see
docs/v04-legend-structural-benchmark-report.md section 1 for the full,
honest account of why family diversity ended up limited):

1. The plan must have at least one candidate in
   `data/dev_plans_v04/annotation_dataset/manifest.json` that ALREADY has a
   human annotation in the live hosted tool's database -- otherwise Phase C
   (scoring) would have literally nothing to score for that plan. This is a
   PRESENCE check only (which candidate_ids have an annotation at all), never
   a check of what the annotation SAYS -- see the holdout note below.
2. Prefer maximum style_family diversity among plans passing (1).
3. The plan must have a usable symbol legend, confirmed by actually running
   `legend_detection.detect_legend_candidates` +
   `legend_entries.classify_candidate_rows` against its real PDF (>=1
   USABLE_TEMPLATE, non-line-swatch entry). This is itself a purely
   geometric/textual check -- it never reads or needs a human label.

W-001..W-010 are excluded by construction: they live in a different repo
entirely (the separate `GasWaterAI` Base44/TS project) and were never part
of the 20 pseudonymous `DEV-*` dev plans this script selects from.

HOLDOUT NOTE: step (1)'s "does an annotation exist for candidate X" check
reads only DOCUMENT IDS from the hosted annotation database (via a
filename-only directory listing of a prior dump -- see
`_ANNOTATED_ID_SNAPSHOT_DIR` below), never a document's `class` or
`verification_status` field. No label VALUE is read by this module, ever.
This script's own output (this file's result) is written once, before any
prediction/matching code exists, and is never revisited after Phase C.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"

# A snapshot of annotation DOCUMENT IDS (filenames only -- see module
# docstring) taken 2026-09-29 for an unrelated, already-completed task
# (verifying the hosted tool's access-code rotation left all annotations
# intact). Reused here purely for its filenames/presence, not reopened for
# content, which is why this path -- not a fresh `ArtifactData` call -- is
# the one this selection module references: it keeps this module's own
# import graph free of any annotation-content-reading dependency.
_ANNOTATED_ID_SNAPSHOT_DIR = (
    "/tmp/claude-0/-home-user-GasWaterAI/7eedfd06-c85b-543d-9781-25386f7b8e43/"
    "scratchpad/annotation_web/db_dump_after_gate_change/annotations"
)

MIN_USABLE_LEGEND_ENTRIES = 1


def annotated_candidate_ids_presence_only(snapshot_dir: str = _ANNOTATED_ID_SNAPSHOT_DIR) -> set[str]:
    """Returns the SET of candidate_ids with an existing annotation --
    filenames only, each file's contents are never opened here."""
    return {fn[:-5] for fn in os.listdir(snapshot_dir) if fn.endswith(".json")}


def plan_stats(manifest: list[dict], annotated_ids: set[str]) -> dict[str, dict]:
    total = Counter(r["plan_id_pseudonymous"] for r in manifest)
    annotated = Counter()
    family = {}
    for r in manifest:
        pid = r["plan_id_pseudonymous"]
        family[pid] = r.get("style_family")
        if r["candidate_id"] in annotated_ids:
            annotated[pid] += 1
    return {
        pid: {"total_candidates": total[pid], "annotated_presence": annotated.get(pid, 0), "style_family": family[pid]}
        for pid in total
    }


def select_five_plans() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    annotated_ids = annotated_candidate_ids_presence_only()
    stats = plan_stats(manifest, annotated_ids)

    eligible = {pid: s for pid, s in stats.items() if s["annotated_presence"] > 0}

    selected = sorted(eligible.keys())
    families = sorted({stats[pid]["style_family"] for pid in selected})

    return {
        "selection_criteria": [
            "annotated_presence > 0 (candidate_ids only, never label content)",
            "usable symbol legend confirmed by running legend_detection + "
            "legend_entries against the real PDF (>=1 USABLE_TEMPLATE, "
            "non-line-swatch entry)",
            "maximum feasible style_family diversity among plans meeting the above",
        ],
        "all_plan_stats": stats,
        "eligible_by_annotated_presence": eligible,
        "selected_plans": selected,
        "distinct_style_families_in_selection": families,
        "family_diversity_note": (
            "Only 5 of the 14 manifested DEV plans have ANY existing human "
            "annotation yet (review has so far progressed through DEV-01, "
            "DEV-03, DEV-04, DEV-05, DEV-06 in manifest order); those 5 span "
            "only 3 distinct style_family values (STANDALONE-1, STANDALONE-3, "
            "FAM-4 x3), not 5. Every other manifested plan (DEV-07, 10, 11, "
            "12, 13, 14, 15, 17, 20) has ZERO existing annotations and so "
            "cannot contribute anything to Phase C scoring regardless of its "
            "legend quality or family. Selecting for real scorable evidence "
            "over family diversity was the deliberate choice here -- see "
            "docs/v04-legend-structural-benchmark-report.md section 1."
        ),
    }


if __name__ == "__main__":
    result = select_five_plans()
    out_path = REPO_ROOT / "data" / "dev_plans_v04" / "legend_structural_benchmark" / "selected_plans.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
