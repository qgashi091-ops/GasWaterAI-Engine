"""v0.4 Phase 6c -- dataset-quality cleanup, run after the domain expert's
manual review of >400 of the 837 candidates surfaced two systematic
problems: (1) an impression of being asked about the same physical object
repeatedly, and (2) many candidates concerning wastewater/drainage
(Schmutzwasser/Abwasser/Regenwasser/Entwässerung) content, which is out of
scope for GasWaterAI's drinking-water/hygiene checker.

Does NOT reprocess any PDF, regenerate any candidate, or touch a single
human annotation -- it only adds two purely-derived, auditable fields to
each of the existing 837 candidate records:

  - scope_status: "IN_SCOPE" | "OUT_OF_SCOPE_WASTEWATER"
  - dedup_status: "CANONICAL" | "DUPLICATE_OF:<canonical_candidate_id>"

Both fields are a deterministic function of manifest.json alone (never of
the live human-annotation database), so this script is safe to re-run any
time manifest.json changes and always produces the same result.

DEDUPLICATION METHODOLOCY (see docs/v04-dataset-cleanup-report.md for the
full write-up): candidates were grouped by (plan_id_pseudonymous, page) and
every same-page pair's tight bbox (bbox_page_space) was compared by IoU and
centroid distance, plus graph_association node/edge-id overlap. A pair
counts as the same physical instance only when IoU>=0.5, OR
(IoU>=0.15 AND a shared graph id AND centroid distance <20pt) -- geometric
overlap corroborated by graph position, never mere visual resemblance
(explicitly never deduplicated on that basis alone). Measured result on
this batch: the highest IoU observed anywhere in the dataset was 0.118, and
no pair met either rule -- so DEDUP_GROUPS below is empty and every
candidate's dedup_status is "CANONICAL". The function stays in this script
(not hardcoded to "always CANONICAL") so a future batch with real
duplicate candidates is handled the same way, without a code change.

WASTEWATER METHODOLOGY: exact-match against a list of suggestion-label
strings confirmed, by direct inspection of this batch's manifest.json, to
describe wastewater/drainage pipe-NETWORK material or medium -- never a
fixture. A WC/Dusche/Waschtisch/Küche candidate is never matched by this
list; the exclusion is for the wastewater system itself, per the domain
expert's own instruction, not for a fixture that happens to also have a
wastewater connection.

Usage: python3 scripts/apply_dataset_cleanup.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "dev_plans_v04" / "annotation_dataset"
WEB_EXPORT_DIR = ROOT / "data" / "dev_plans_v04" / "annotation_web_export"

# Confirmed present verbatim in manifest.json's engine_suggestions
# (plan_specific.label / generic.label); see module docstring.
WASTEWATER_SUGGESTION_LABELS = {
    "13 mm armaflex schmutzwasser entlüftungen",
    "dämmschlauch schmutzwasser sammelleitungen",
    "geberit isol schmutzwasser fallstränge",
    "schmutzabwasser",
    "tropfwasser kondensabwasser",
}

DUPLICATE_IOU_MIN = 0.5
DUPLICATE_IOU_WITH_GRAPH_MIN = 0.15
DUPLICATE_CENTROID_DIST_MAX_PT = 20.0


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox: tuple) -> tuple:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _centroid_dist(a: tuple, b: tuple) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _graph_ids(rec: dict) -> set:
    ga = rec.get("graph_association") or {}
    return set(ga.get("node_ids") or []) | set(ga.get("edge_ids") or [])


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def compute_dedup_status(manifest: list[dict]) -> dict[str, str]:
    by_id = {r["candidate_id"]: r for r in manifest}
    groups = defaultdict(list)
    for r in manifest:
        groups[(r["plan_id_pseudonymous"], r["page"])].append(r)

    uf = _UnionFind(by_id.keys())
    for recs in groups.values():
        n = len(recs)
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = recs[i], recs[j]
                b1, b2 = tuple(r1["bbox_page_space"]), tuple(r2["bbox_page_space"])
                iou = _iou(b1, b2)
                is_dup = iou >= DUPLICATE_IOU_MIN
                if not is_dup and iou >= DUPLICATE_IOU_WITH_GRAPH_MIN:
                    shared = _graph_ids(r1) & _graph_ids(r2)
                    if shared and _centroid_dist(b1, b2) < DUPLICATE_CENTROID_DIST_MAX_PT:
                        is_dup = True
                if is_dup:
                    uf.union(r1["candidate_id"], r2["candidate_id"])

    members_by_root = defaultdict(list)
    for cid in by_id:
        members_by_root[uf.find(cid)].append(cid)

    dedup_status = {}
    for root, members in members_by_root.items():
        if len(members) == 1:
            dedup_status[members[0]] = "CANONICAL"
            continue

        def score_of(cid):
            es = by_id[cid].get("engine_suggestions") or {}
            ps = es.get("plan_specific")
            return ps.get("score", 0.0) if ps else 0.0

        canonical = sorted(members, key=lambda cid: (-score_of(cid), cid))[0]
        for m in members:
            dedup_status[m] = "CANONICAL" if m == canonical else f"DUPLICATE_OF:{canonical}"
    return dedup_status


def compute_scope_status(manifest: list[dict]) -> dict[str, str]:
    scope_status = {}
    for r in manifest:
        es = r.get("engine_suggestions") or {}
        labels = []
        for key in ("plan_specific", "generic"):
            ev = es.get(key)
            if ev and ev.get("label"):
                labels.append(ev["label"].strip().lower())
        scope_status[r["candidate_id"]] = (
            "OUT_OF_SCOPE_WASTEWATER"
            if any(l in WASTEWATER_SUGGESTION_LABELS for l in labels)
            else "IN_SCOPE"
        )
    return scope_status


def main() -> None:
    with open(DATASET_DIR / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    scope_status = compute_scope_status(manifest)
    dedup_status = compute_dedup_status(manifest)

    with open(WEB_EXPORT_DIR / "candidates.json", encoding="utf-8") as f:
        candidates = json.load(f)
    for c in candidates:
        cid = c["candidate_id"]
        c["scope_status"] = scope_status[cid]
        c["dedup_status"] = dedup_status[cid]
    with open(WEB_EXPORT_DIR / "candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    audit = {
        "scope_status": scope_status,
        "dedup_status": dedup_status,
        "wastewater_suggestion_labels": sorted(WASTEWATER_SUGGESTION_LABELS),
        "duplicate_thresholds": {
            "iou_min": DUPLICATE_IOU_MIN,
            "iou_with_graph_min": DUPLICATE_IOU_WITH_GRAPH_MIN,
            "centroid_dist_max_pt": DUPLICATE_CENTROID_DIST_MAX_PT,
        },
        "out_of_scope_wastewater_count": sum(1 for v in scope_status.values() if v == "OUT_OF_SCOPE_WASTEWATER"),
        "duplicate_non_canonical_count": sum(1 for v in dedup_status.values() if v.startswith("DUPLICATE_OF")),
    }
    with open(DATASET_DIR / "dataset_cleanup_v1.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"scope_status: IN_SCOPE={sum(1 for v in scope_status.values() if v=='IN_SCOPE')}, "
          f"OUT_OF_SCOPE_WASTEWATER={audit['out_of_scope_wastewater_count']}")
    print(f"dedup_status: CANONICAL={sum(1 for v in dedup_status.values() if v=='CANONICAL')}, "
          f"DUPLICATE_OF=*={audit['duplicate_non_canonical_count']}")
    print(f"Wrote {WEB_EXPORT_DIR / 'candidates.json'} (+scope_status/dedup_status fields)")
    print(f"Wrote {DATASET_DIR / 'dataset_cleanup_v1.json'} (full audit record)")


if __name__ == "__main__":
    main()
