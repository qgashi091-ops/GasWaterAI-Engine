"""v0.4 Phase 7 -- train/validation/test split design, at PROJECT/STYLE-FAMILY
granularity, never at individual-plan granularity.

W-001..W-010 (the existing Golden holdout) are not inputs to this module at
all -- they are never assigned a split here, never touched, never counted.
This module only ever sees the 20 new DEV-xx plans grouped into families by
families.py.

ALGORITHM: a family (not a plan) is the unit of assignment, so two plans
that share a project can never land in different splits. Families are
sorted largest-first and each is greedily assigned to whichever split is
currently furthest below its TARGET plan-count share -- a simple,
deterministic bin-packing heuristic that keeps the realized split close to
the target ratio despite family sizes not dividing evenly (with only 20
plans and family sizes of 1-3, an exact ratio is not achievable regardless
of algorithm; greedy largest-first is the standard, auditable choice for
this scale).
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}


@dataclass
class SplitAssignment:
    plan_to_split: dict  # {plan_id: "train"|"validation"|"test"}
    family_to_split: dict  # {family_id: split}
    counts: dict  # {split: plan_count}

    def to_dict(self) -> dict:
        return {
            "plan_to_split": self.plan_to_split,
            "family_to_split": self.family_to_split,
            "counts": self.counts,
        }


def design_split(family_of: dict[str, str], ratios: dict[str, float] = None) -> SplitAssignment:
    ratios = ratios or DEFAULT_RATIOS
    families: dict[str, list[str]] = {}
    for plan_id, fam in sorted(family_of.items()):
        families.setdefault(fam, []).append(plan_id)

    total_plans = len(family_of)
    targets = {split: ratio * total_plans for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    family_to_split: dict[str, str] = {}

    # Largest families first, ties broken by family id for determinism.
    ordered = sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for fam, members in ordered:
        # Assign to whichever split is furthest below its target share.
        best_split = max(ratios, key=lambda s: targets[s] - counts[s])
        family_to_split[fam] = best_split
        counts[best_split] += len(members)

    plan_to_split = {pid: family_to_split[fam] for pid, fam in family_of.items()}
    return SplitAssignment(plan_to_split=plan_to_split, family_to_split=family_to_split, counts=counts)
