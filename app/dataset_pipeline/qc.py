"""v0.4 Phase 9 -- quality control over the generated annotation candidates.

Runs entirely over already-generated `AnnotationCandidate` objects
(candidates.py) -- it never re-derives anything from the PDFs, so it stays
cheap even for a large batch. Flags, rather than silently fixes or drops,
every issue: a human reviewer decides what to do with a flagged pair or
class, this module only prioritizes their queue.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Two candidates on the SAME plan/page whose tight bboxes overlap by at
# least this IoU fraction are very likely the same physical symbol detected
# twice (e.g. v0.1's own clustering split one icon into two overlapping
# candidates) -- exactly the "candidate generation overlaps" case the task
# calls out by name.
OVERLAP_IOU_DUPLICATE_THRESHOLD = 0.5
CLASS_IMBALANCE_RATIO_WARNING = 8.0  # largest-class-count / smallest-class-count


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 1e-6) * max(ay1 - ay0, 1e-6)
    area_b = max(bx1 - bx0, 1e-6) * max(by1 - by0, 1e-6)
    return inter / (area_a + area_b - inter)


@dataclass
class DuplicateCandidatePair:
    candidate_id_a: str
    candidate_id_b: str
    plan_id: str
    iou: float


def find_overlapping_candidates(candidates: list) -> list[DuplicateCandidatePair]:
    """candidates: list[AnnotationCandidate]. O(n^2) within each (plan,
    page) group only -- fine at this dataset's scale (tens to low hundreds
    of candidates per plan), and correct is more important than clever here
    since this runs once per batch, not in any hot path."""
    by_page: dict[tuple, list] = {}
    for c in candidates:
        by_page.setdefault((c.plan_id, c.page), []).append(c)

    pairs: list[DuplicateCandidatePair] = []
    for (plan_id, _page), group in by_page.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                iou = _iou(group[i].bbox, group[j].bbox)
                if iou >= OVERLAP_IOU_DUPLICATE_THRESHOLD:
                    pairs.append(DuplicateCandidatePair(
                        candidate_id_a=group[i].candidate_id, candidate_id_b=group[j].candidate_id,
                        plan_id=plan_id, iou=round(iou, 3),
                    ))
    return pairs


def class_imbalance_report(candidates: list) -> dict:
    """Reports on whatever labels currently exist -- either the engine's own
    suggested labels (before any human annotation) or `assigned_class` once
    annotation has started. Never invents a "true" distribution; this is
    descriptive only, for the human to weigh when deciding review priority
    or later sampling/augmentation."""
    suggested = Counter()
    for c in candidates:
        label = None
        if c.plan_specific_suggestion:
            label = c.plan_specific_suggestion["label"]
        elif c.generic_suggestion:
            label = c.generic_suggestion["label"]
        suggested[label or "NO_SUGGESTION"] += 1

    assigned = Counter(c.assigned_class for c in candidates if c.assigned_class)

    counts = list(suggested.values())
    ratio = (max(counts) / min(counts)) if counts and min(counts) > 0 else None
    return {
        "suggested_label_counts": dict(suggested.most_common()),
        "assigned_class_counts": dict(assigned.most_common()),
        "suggested_imbalance_ratio": round(ratio, 1) if ratio else None,
        "imbalance_warning": bool(ratio and ratio >= CLASS_IMBALANCE_RATIO_WARNING),
    }


def low_confidence_review_queue(candidates: list, max_items: int = 200) -> list[str]:
    """Candidate ids worth a human's attention first: no plan-specific
    suggestion AND no generic suggestion (engine has nothing to offer, so a
    human's raw judgement is the only source of truth), ranked with
    duplicates from `find_overlapping_candidates` implicitly de-prioritized
    (a human resolving an overlap pair effectively resolves both)."""
    unsuggested = [c.candidate_id for c in candidates
                   if not c.plan_specific_suggestion and not c.generic_suggestion]
    return unsuggested[:max_items]


@dataclass
class QCReport:
    duplicate_pairs: list  # list[DuplicateCandidatePair]
    class_imbalance: dict
    review_queue_no_suggestion: list  # list[candidate_id]

    def to_dict(self) -> dict:
        return {
            "duplicate_pairs": [
                {"candidate_id_a": p.candidate_id_a, "candidate_id_b": p.candidate_id_b,
                 "plan_id": p.plan_id, "iou": p.iou}
                for p in self.duplicate_pairs
            ],
            "class_imbalance": self.class_imbalance,
            "review_queue_no_suggestion_count": len(self.review_queue_no_suggestion),
            "review_queue_no_suggestion_sample": self.review_queue_no_suggestion[:20],
        }


def run_qc(candidates: list) -> QCReport:
    return QCReport(
        duplicate_pairs=find_overlapping_candidates(candidates),
        class_imbalance=class_imbalance_report(candidates),
        review_queue_no_suggestion=low_confidence_review_queue(candidates),
    )
