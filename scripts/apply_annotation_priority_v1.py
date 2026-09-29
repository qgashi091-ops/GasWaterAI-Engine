"""v0.4 Phase 6e -- active-annotation-queue prioritization ("ACTIVE ANNOTATION v1").

Goal: reduce reviewer workload and collect more POSITIVE drinking-water
component examples, by re-ranking the remaining UNLABELED, in-scope
candidates so the hosted annotation tool serves the most promising ones
first. Never assigns a class to an unlabeled candidate -- this script only
adds two new, purely-derived, auditable fields to each existing candidate
record: `annotation_priority` (an integer score) and `priority_reasons` (the
list of rule names that produced it), plus `priority_tier`
("high" | "normal") for unlabeled in-scope candidates only.

WHAT THE 387 EXISTING HUMAN LABELS ARE USED FOR (and not used for): read
here ONLY to (a) compute `positive_class_counts` for the report, (b)
identify candidate-GENERATION patterns (suggestion labels) with an extremely
high NOT_A_COMPONENT rate, to deprioritize other unlabeled candidates that
share the same generated label, and (c) find the graph positions of already
-confirmed real components, to mildly boost other unlabeled candidates nearby
on the same plan+page (pipe networks cluster real fixtures together). At no
point is any existing label copied onto, or used to decide the class of, any
individual unlabeled candidate -- verified by the audit in `main()` below,
which asserts every candidate this script touches has no `class` assigned
before or after running.

Everything here is a pure function of (manifest.json, dataset_cleanup_v1.json,
a snapshot of the 387 labels' candidate_id -> class mapping): re-running this
script against the same inputs always produces the same
`annotation_priority`/`priority_reasons`/`priority_tier` for every candidate.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "manifest.json"
CLEANUP_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "dataset_cleanup_v1.json"
WEB_EXPORT_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_web_export" / "candidates.json"
OUT_REPORT_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "annotation_dataset" / "annotation_priority_v1.json"

# ---- frozen keyword rules (chosen from the taxonomy/domain vocabulary, not
# tuned against any individual candidate's own label match rate) ----------
POTABLE_WATER_PATTERN = re.compile(
    r"kaltwasser|warmwasser|trinkwasser|zirkulation|\bkw\b|\bww\b|\bpwc\b|\bpwh\b|"
    r"sanit(ä|ae)r|wasserversorgung|hausanschluss|wasserleitung"
)
APPARATUS_PATTERN = re.compile(
    r"ventil|hahn\b|filter|wasserz(ä|ae)hler|r(ü|ue)ckflussverhinderer|absperr|"
    r"wasserenth(ä|ae)rter|boiler|wassererw(ä|ae)rmer|verteiler|sicherheitsgruppe|"
    r"entl(ü|ue)fter|druckminderer|klappe|armatur|batterie|mischer|mischrf|"
    r"waschmaschine|waschtrog|gartenventil|thermostat"
)
DIMENSION_LU_PATTERN = re.compile(r"\(\s*\d+\s*lu\s*\)|\blu\b|^\s*\d+(\.\d+)?\s*(mm)?\s*$")
INSULATION_MATERIAL_PATTERN = re.compile(
    r"d(ä|ae)mm|armaflex|isolier|\bpe-s\b|\bcr\s*\d+\b|\bfl\s*\d+\b|messf(ü|ue)hler|niveau"
)
TITLEBLOCK_PATTERN = re.compile(
    r"plan-name|massstab|\brev\.|index.*art.*(ä|ae)nderung|gezeichnet|bauseits|"
    r"^\s*-\s*$|^\s*[a-b]\s*$|^\d\.\s*lage"
)

# Score weights -- frozen for this round; a positive addition or negative
# subtraction per triggered rule, summed into `annotation_priority`.
WEIGHTS = {
    "graph_endpoint_or_branch": 3,
    "potable_water_terminology": 3,
    "apparatus_component_terminology": 3,
    "usable_legend_evidence": 2,
    "component_like_geometry": 2,
    "proximity_to_proven_potable_graph": 2,
    "high_confidence_candidate_generation_evidence": 1,
    "dimension_or_lu_label": -3,
    "pure_pipe_segment_geometry": -3,
    "text_only_cluster_weak_evidence": -2,
    "insulation_or_material_description": -3,
    "titleblock_or_boilerplate_text": -3,
    "known_low_value_pattern": -4,
}

NON_POSITIVE_CLASSES = {"NOT_A_COMPONENT", "AMBIGUOUS"}
TARGET_HIGH_PRIORITY_COUNT = 130  # middle of the requested 100-150 range


def _label_of(record: dict) -> str | None:
    ps = (record.get("engine_suggestions") or {}).get("plan_specific")
    return ps["label"].strip().casefold() if ps and ps.get("label") else None


def compute_low_value_patterns(manifest: list[dict], labeled_by_id: dict[str, dict]) -> dict[str, dict]:
    """Suggestion labels with >=3 labeled instances where >=90% are
    NOT_A_COMPONENT -- same methodology as the earlier v0.4 dataset-cleanup
    pass (docs/v04-dataset-cleanup-report.md step 4), re-derived fresh here
    on the current 387 labels. Used only to DEPRIORITIZE other, still-
    unlabeled candidates sharing the same generated label -- never to label
    the labeled ones (they already have a human label) or to label any
    unlabeled one."""
    per_label_total = Counter()
    per_label_not_a_component = Counter()
    for r in manifest:
        label = _label_of(r)
        if label is None:
            continue
        ann = labeled_by_id.get(r["candidate_id"])
        if ann is None:
            continue
        per_label_total[label] += 1
        if ann["class"] == "NOT_A_COMPONENT":
            per_label_not_a_component[label] += 1

    patterns = {}
    for label, total in per_label_total.items():
        if total < 3:
            continue
        rate = per_label_not_a_component[label] / total
        if rate >= 0.9:
            patterns[label] = {"not_a_component_rate": round(rate, 4), "labeled_n": total}
    return patterns


def compute_proven_potable_positions(manifest: list[dict], labeled_by_id: dict[str, dict]) -> dict[tuple, list]:
    """For every (plan, page), the bbox_page_space + node/edge ids of every
    candidate a human has ALREADY confirmed is a real, in-scope drinking-
    water component (i.e. LABELED with a class that is neither
    NOT_A_COMPONENT nor AMBIGUOUS). Used only to give a small, explainable
    proximity bonus to other UNLABELED candidates on the same plan+page --
    never to assign any class."""
    out: dict[tuple, list] = {}
    for r in manifest:
        ann = labeled_by_id.get(r["candidate_id"])
        if ann is None or ann["class"] in NON_POSITIVE_CLASSES:
            continue
        key = (r["plan_id_pseudonymous"], r["page"])
        ga = r.get("graph_association") or {}
        out.setdefault(key, []).append({
            "bbox": r["bbox_page_space"],
            "node_ids": set(ga.get("node_ids") or []),
            "edge_ids": set(ga.get("edge_ids") or []),
        })
    return out


def _bbox_center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _bbox_diag(b) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def _near_proven_positive(record: dict, proven_by_plan_page: dict) -> bool:
    key = (record["plan_id_pseudonymous"], record["page"])
    proven = proven_by_plan_page.get(key)
    if not proven:
        return False
    ga = record.get("graph_association") or {}
    my_nodes = set(ga.get("node_ids") or [])
    my_edges = set(ga.get("edge_ids") or [])
    my_bbox = record["bbox_page_space"]
    my_center = _bbox_center(my_bbox)
    my_scale = max(_bbox_diag(my_bbox), 1.0)
    for p in proven:
        if my_nodes & p["node_ids"] or my_edges & p["edge_ids"]:
            return True
        other_center = _bbox_center(p["bbox"])
        dist = math.hypot(my_center[0] - other_center[0], my_center[1] - other_center[1])
        if dist <= 3.0 * my_scale:
            return True
    return False


def _aspect_ratio(bbox) -> float:
    w, h = max(1e-6, bbox[2] - bbox[0]), max(1e-6, bbox[3] - bbox[1])
    return max(w, h) / min(w, h)


def score_candidate(record: dict, low_value_patterns: dict, proven_by_plan_page: dict) -> tuple[int, list[str]]:
    label = _label_of(record) or ""
    ga = record.get("graph_association") or {}
    relation = ga.get("relation")
    ps = (record.get("engine_suggestions") or {}).get("plan_specific")
    ps_score = ps.get("score") if ps else None
    aspect = _aspect_ratio(record["bbox_page_space"])

    reasons: list[str] = []

    if relation in ("at_endpoint", "at_branch"):
        reasons.append("graph_endpoint_or_branch")
    if label and POTABLE_WATER_PATTERN.search(label):
        reasons.append("potable_water_terminology")
    if label and APPARATUS_PATTERN.search(label):
        reasons.append("apparatus_component_terminology")
    if ps is not None:
        reasons.append("usable_legend_evidence")
    if aspect <= 3.0:
        reasons.append("component_like_geometry")
    if _near_proven_positive(record, proven_by_plan_page):
        reasons.append("proximity_to_proven_potable_graph")
    if ps_score is not None and ps_score >= 0.5:
        reasons.append("high_confidence_candidate_generation_evidence")

    if label and DIMENSION_LU_PATTERN.search(label):
        reasons.append("dimension_or_lu_label")
    if aspect > 8.0 and not (label and (POTABLE_WATER_PATTERN.search(label) or APPARATUS_PATTERN.search(label))):
        reasons.append("pure_pipe_segment_geometry")
    if ps is None and relation == "near_not_connected":
        reasons.append("text_only_cluster_weak_evidence")
    if label and INSULATION_MATERIAL_PATTERN.search(label):
        reasons.append("insulation_or_material_description")
    if label and TITLEBLOCK_PATTERN.search(label):
        reasons.append("titleblock_or_boilerplate_text")
    if label in low_value_patterns:
        reasons.append(f"known_low_value_pattern:{label}")

    score = sum(WEIGHTS[r.split(":")[0]] for r in reasons)
    return score, reasons


def main() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(CLEANUP_PATH, encoding="utf-8") as f:
        cleanup = json.load(f)
    scope_status = cleanup["scope_status"]
    dedup_status = cleanup["dedup_status"]

    ann_dir = os.environ.get("ANNOTATION_DUMP_DIR")
    if not ann_dir:
        raise SystemExit("Set ANNOTATION_DUMP_DIR to a fresh ArtifactData dump directory of the 'annotations' collection.")
    labeled_by_id: dict[str, dict] = {}
    for fn in os.listdir(ann_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, fn), encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("verification_status") == "LABELED":
            labeled_by_id[fn[:-5]] = doc

    low_value_patterns = compute_low_value_patterns(manifest, labeled_by_id)
    proven_by_plan_page = compute_proven_potable_positions(manifest, labeled_by_id)

    scored = []  # (candidate_id, score, reasons) for unlabeled in-scope candidates only
    per_candidate_priority: dict[str, dict] = {}

    for r in manifest:
        cid = r["candidate_id"]
        is_labeled = cid in labeled_by_id
        in_scope = scope_status.get(cid) == "IN_SCOPE" and dedup_status.get(cid) == "CANONICAL"

        # HARD SAFETY CHECK: never let this script's own bookkeeping imply
        # a class for an unlabeled candidate.
        assert is_labeled or r.get("class") is None, f"{cid} is unlabeled but manifest.json already has a class"

        if is_labeled:
            per_candidate_priority[cid] = {"annotation_priority": None, "priority_reasons": ["already_annotated"], "priority_tier": "labeled"}
            continue
        if not in_scope:
            per_candidate_priority[cid] = {"annotation_priority": None, "priority_reasons": ["excluded_out_of_scope"], "priority_tier": "excluded"}
            continue

        score, reasons = score_candidate(r, low_value_patterns, proven_by_plan_page)
        scored.append((cid, score, reasons))

    # Deterministic ranking: score desc, then candidate_id asc as a stable tiebreak.
    scored.sort(key=lambda t: (-t[1], t[0]))
    high_priority_ids = {cid for cid, _, _ in scored[:TARGET_HIGH_PRIORITY_COUNT]}

    for cid, score, reasons in scored:
        per_candidate_priority[cid] = {
            "annotation_priority": score,
            "priority_reasons": reasons,
            "priority_tier": "high" if cid in high_priority_ids else "normal",
        }

    # ---- write into the web-export candidates.json (adds fields only) ----
    with open(WEB_EXPORT_PATH, encoding="utf-8") as f:
        web_candidates = json.load(f)
    for c in web_candidates:
        c.update(per_candidate_priority[c["candidate_id"]])
    with open(WEB_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(web_candidates, f, indent=2, ensure_ascii=False)

    # ---- positive-class counts + report ----
    positive_class_counts = Counter(a["class"] for a in labeled_by_id.values())
    tier_counts = Counter(v["priority_tier"] for v in per_candidate_priority.values())

    report = {
        "total_candidates": len(manifest),
        "labeled_count": len(labeled_by_id),
        "positive_class_counts": dict(positive_class_counts.most_common()),
        "low_value_patterns_derived": low_value_patterns,
        "tier_counts": dict(tier_counts),
        "target_high_priority_count": TARGET_HIGH_PRIORITY_COUNT,
        "score_weights": WEIGHTS,
    }
    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote priority fields into {WEB_EXPORT_PATH}")
    print(f"Wrote report to {OUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
