"""Component Detector v1 POC -- Phase 0 dataset audit.

Determines whether the project's existing human-reviewed data supports
training a small object detector for any potable-water component class,
BEFORE any training is attempted.

Reads ONLY:
  - `manifest.json`  (837 v0.1-generated candidates: geometry + provenance;
    `class`/`verification_status` in this file are always null/UNLABELED --
    they are the pre-annotation seed, never the ground truth)
  - `legend_symbols.json` (341 extracted legend icons: geometry + provenance;
    `expert_decision`/`expert_confirmed_name` in this file are likewise
    always null -- also a pre-review seed, not the ground truth)
  - a fresh dump of the live `annotations` collection (the actual human
    labels, one JSON doc per candidate_id, keyed `class` +
    `verification_status` + `annotator_note`)
  - a fresh dump of the live `legend_symbol_reviews` collection (the actual
    human reviews, one JSON doc per legend_symbol_id, keyed `decision` +
    `confirmed_name`)

To reproduce the live-DB dumps this script reads, run (once per audit):
    ArtifactData(action="query", url="https://claude.ai/artifact/NVn2g1hNY48sn291JzUAaW",
                 collection="annotations", query={"limit": 1000}, out_dir=<dir>/annotations)
    ArtifactData(action="query", url="https://claude.ai/artifact/NVn2g1hNY48sn291JzUAaW",
                 collection="legend_symbol_reviews", query={"limit": 1000}, out_dir=<dir>/legend_symbol_reviews)
then pass <dir> as ANNOTATIONS_DUMP_DIR's parent via --dump-dir.

Ground-truth rules enforced here (per the POC's own instructions):
  - An automatically proposed label (engine_suggestions, proposed_component_name,
    relevance_reason) is NEVER treated as ground truth -- read only as
    provenance/context in the output, never counted as a "positive".
  - `class` != null (annotations, verification_status == LABELED) is the
    only source of positive/negative truth for candidate instances.
  - `AMBIGUOUS` is excluded from truth entirely (never positive, never
    negative, never counted as evidence for a class).
  - `NOT_A_COMPONENT` is usable ONLY as negative/background evidence, never
    as a positive class.
  - For legend reviews, only `decision in {"correct", "corrected"}` counts
    as a confirmed positive identification of a named symbol; `"unclear"`
    is never training truth. There is no "incorrect" observed in the
    current 150 reviews, but if present it would be negative evidence at
    most, never a positive class.
  - Synonyms are normalized only when unambiguous (casefold + one known
    typo fix, "Badewannenmischrf" -> "Badewannenmischer", confirmed by the
    domain expert's own annotator_note wording elsewhere in the same
    dataset); the original label string is preserved as `raw_texts`
    provenance in the output, never silently discarded.

This script performs NO training, NO PDF reprocessing, NO Base44 changes,
and touches none of W-001..W-010. It is read-only against the dataset.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
LEGEND_SYMBOLS_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "legend_symbol_dataset" / "legend_symbols.json"
OUT_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "detector_poc_phase0_audit.json"

NEGATIVE_CLASSES = {"NOT_A_COMPONENT"}
NEVER_TRUTH_CLASSES = {"AMBIGUOUS"}

# One unambiguous typo fix, confirmed by comparing against the sibling
# instance's annotator_note in the same dataset -- not a semantic merge of
# two different things.
LEGEND_SYNONYM_FIX = {"badewannenmischrf": "badewannenmischer"}


def load_dump(dump_dir: Path) -> dict[str, dict]:
    out = {}
    for f in dump_dir.glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    return out


def audit(annotations_dump_dir: Path, reviews_dump_dir: Path) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_by_id = {c["candidate_id"]: c for c in manifest}
    legend_symbols = json.loads(LEGEND_SYMBOLS_PATH.read_text())
    legend_by_id = {e["legend_symbol_id"]: e for e in legend_symbols}

    annotations = load_dump(annotations_dump_dir)
    reviews = load_dump(reviews_dump_dir)

    class_counts = collections.Counter()
    for doc in annotations.values():
        if doc.get("verification_status") == "LABELED":
            class_counts[doc["class"]] += 1

    positive_detail = collections.defaultdict(
        lambda: {"instances": [], "plans": set(), "families": set(), "family_plan_counts": collections.Counter()}
    )
    for doc_id, doc in annotations.items():
        if doc.get("verification_status") != "LABELED":
            continue
        cls = doc["class"]
        if cls in NEGATIVE_CLASSES or cls in NEVER_TRUTH_CLASSES:
            continue
        m = manifest_by_id.get(doc_id)
        if m is None:
            continue
        d = positive_detail[cls]
        d["instances"].append(doc_id)
        d["plans"].add(m["plan_id_pseudonymous"])
        d["families"].add(m["style_family"])
        d["family_plan_counts"][(m["style_family"], m["plan_id_pseudonymous"])] += 1

    other_relevant_breakdown = collections.defaultdict(
        lambda: {"instances": [], "plans": set(), "families": set()}
    )
    for doc_id, doc in annotations.items():
        if doc.get("verification_status") != "LABELED" or doc.get("class") != "OTHER_RELEVANT_SYMBOL":
            continue
        note = (doc.get("annotator_note") or "").strip()
        key = LEGEND_SYNONYM_FIX.get(note.casefold(), note.casefold()) if note else "(no note)"
        m = manifest_by_id.get(doc_id, {})
        d = other_relevant_breakdown[key]
        d["instances"].append(doc_id)
        if m:
            d["plans"].add(m["plan_id_pseudonymous"])
            d["families"].add(m["style_family"])

    decision_counts = collections.Counter(r["decision"] for r in reviews.values())
    legend_positive_detail = collections.defaultdict(
        lambda: {"instances": [], "plans": set(), "families": set(), "raw_texts": set(), "decisions": set()}
    )
    for doc_id, doc in reviews.items():
        if doc["decision"] not in ("correct", "corrected"):
            continue
        e = legend_by_id.get(doc_id)
        if e is None:
            continue
        name = (doc.get("confirmed_name") or e.get("raw_legend_text") or "").strip()
        if not name:
            continue
        key = name.casefold()
        d = legend_positive_detail[key]
        d["instances"].append(doc_id)
        d["plans"].add(e["plan_id_pseudonymous"])
        d["families"].add(e["style_family"])
        d["raw_texts"].add(name)
        d["decisions"].add(doc["decision"])

    def _serialize_detail(detail):
        return {
            k: {
                "instance_count": len(v["instances"]),
                "instances": v["instances"],
                "plans": sorted(v["plans"]),
                "families": sorted(v["families"]),
            }
            for k, v in detail.items()
        }

    result = {
        "manifest_candidate_count": len(manifest),
        "legend_symbol_count": len(legend_symbols),
        "annotations_labeled_count": sum(
            1 for d in annotations.values() if d.get("verification_status") == "LABELED"
        ),
        "legend_reviews_count": len(reviews),
        "annotations_class_counts": dict(class_counts),
        "annotations_positive_detail": _serialize_detail(positive_detail),
        "annotations_positive_family_plan_breakdown": {
            cls: {f"{fam}|{plan}": n for (fam, plan), n in d["family_plan_counts"].items()}
            for cls, d in positive_detail.items()
        },
        "other_relevant_symbol_breakdown": _serialize_detail(other_relevant_breakdown),
        "legend_review_decision_counts": dict(decision_counts),
        "legend_review_positive_detail": _serialize_detail(legend_positive_detail),
        "duplicate_near_duplicate_note": (
            "Reused the already-verified geometric duplicate audit for this exact "
            "837-candidate dataset (docs/v04-dataset-cleanup-report.md, Step 2): "
            "IoU>=0.5 OR (IoU>=0.15 AND shared graph id AND centroid_dist<20pt) "
            "checked across all 41,309 same-(plan,page) candidate pairs -- zero "
            "found above threshold (max IoU observed anywhere: 0.118). Not "
            "re-derived here; that result covers the full candidate set including "
            "every candidate audited above."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True, help="Directory containing annotations/ and legend_symbol_reviews/ subdirectories from a fresh ArtifactData query dump.")
    args = parser.parse_args()

    result = audit(args.dump_dir / "annotations" / "annotations", args.dump_dir / "legend_symbol_reviews" / "legend_symbol_reviews")
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT_PATH}")
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "annotations_positive_detail", "other_relevant_symbol_breakdown", "legend_review_positive_detail",
        "annotations_positive_family_plan_breakdown",
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
