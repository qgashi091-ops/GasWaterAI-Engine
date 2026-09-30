"""v0.4 Phase 7 -- builds the legend-symbol-only dataset ("ACTIVE ANNOTATION
v2") across all 20 real dev plans. Replaces the candidate-based active
annotation workflow entirely; see app/legend_symbol_dataset/__init__.py.

Does NOT touch the prior 837-candidate dataset or the 387 human
annotations in the hosted tool's database -- both remain exactly where
they are, as historical/audit data.

Usage: python3 scripts/build_legend_symbol_dataset.py
Writes:
  data/dev_plans_v04/legend_symbol_dataset/manifest.json
  data/dev_plans_v04/legend_symbol_dataset/crops/<legend_symbol_id>_symbol.png
  data/dev_plans_v04/legend_symbol_dataset/crops/<legend_symbol_id>_context.png
  data/dev_plans_v04/legend_symbol_dataset/build_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset_pipeline.families import group_families
from app.legend_symbol_dataset.extract import extract_plan
from app.plan_analysis.pipeline import analyze_pdf_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "raw"
ID_MAPPING_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "id_mapping.LOCAL_ONLY.json"
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "legend_symbol_dataset"
CROPS_DIR = OUT_DIR / "crops"

DEDUP_DISTANCE_PT = 20.0


def _bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def dedup_same_plan_page(records: list) -> tuple[list, int]:
    """Suppresses only a LITERAL duplicate extraction of the same legend
    symbol on the same plan+page (identical normalized label text within
    DEDUP_DISTANCE_PT of each other) -- never a different plan, and never a
    genuinely different-looking instance of the same component TYPE, both
    of which are explicitly wanted as real training-data diversity."""
    by_key: dict[tuple, list] = {}
    for r in records:
        by_key.setdefault((r.plan_id, r.page), []).append(r)

    kept = []
    removed = 0
    for group in by_key.values():
        used = [False] * len(group)
        for i, r in enumerate(group):
            if used[i]:
                continue
            kept.append(r)
            ci = _bbox_center(r.bbox)
            norm_i = r.raw_legend_text.strip().casefold()
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if group[j].raw_legend_text.strip().casefold() == norm_i and _dist(ci, _bbox_center(group[j].bbox)) <= DEDUP_DISTANCE_PT:
                    used[j] = True
                    removed += 1
            used[i] = True
    return kept, removed


def main() -> None:
    with open(ID_MAPPING_PATH, encoding="utf-8") as f:
        id_mapping = json.load(f)
    original_filenames = {pid: entry["original_filename"] for pid, entry in id_mapping.items()}
    family_of = group_families(original_filenames)

    plan_ids = sorted(id_mapping.keys(), key=lambda p: int(p.split("-")[1]))
    assert len(plan_ids) == 20, f"expected 20 dev plans, found {len(plan_ids)}"

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    per_plan_stats = {}
    plans_with_usable_legend = 0
    dedup_removed_count = 0

    # Written/flushed after EVERY plan (not only at the very end) so a
    # long run that gets interrupted (DEV-07 alone turned out to hold 1196
    # legend rows -- far more than any other plan) never loses already-
    # completed plans' work.
    for plan_id in plan_ids:
        pdf_path = str(RAW_DIR / f"{plan_id}.pdf")
        print(f"Processing {plan_id}...", flush=True)
        doc = analyze_pdf_file(pdf_path)
        records, stats = extract_plan(plan_id, pdf_path, doc, family_of.get(plan_id))
        # Dedup is always within one (plan, page) by construction -- safe
        # to apply per-plan rather than needing every plan's records held
        # in memory at once.
        deduped, removed = dedup_same_plan_page(records)
        dedup_removed_count += removed

        for r in deduped:
            symbol_fn = f"{r.legend_symbol_id}_symbol.png"
            context_fn = f"{r.legend_symbol_id}_context.png"
            cv2.imwrite(str(CROPS_DIR / symbol_fn), cv2.cvtColor(r.symbol_png, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(CROPS_DIR / context_fn), cv2.cvtColor(r.context_png, cv2.COLOR_RGB2BGR))
            d = r.to_dict()
            d["symbol_crop_filename"] = symbol_fn
            d["context_crop_filename"] = context_fn
            manifest.append(d)

        per_plan_stats[plan_id] = {
            "has_usable_legend": stats.has_usable_legend,
            "legend_entries_total": stats.legend_entries_total,
            "excluded_not_symbol": stats.excluded_not_symbol,
            "excluded_privacy": stats.excluded_privacy,
            "excluded_non_potable": stats.excluded_non_potable,
            "included": stats.included,
        }
        if stats.has_usable_legend:
            plans_with_usable_legend += 1
        print(f"  {plan_id}: usable_legend={stats.has_usable_legend} total_entries={stats.legend_entries_total} "
              f"included={stats.included} excluded_non_potable={sum(stats.excluded_non_potable.values())}")

        with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_entries = sum(s["legend_entries_total"] for s in per_plan_stats.values())
    total_excluded_not_symbol = sum(s["excluded_not_symbol"] for s in per_plan_stats.values())
    total_excluded_privacy = sum(s["excluded_privacy"] for s in per_plan_stats.values())
    excluded_non_potable_totals: Counter = Counter()
    for s in per_plan_stats.values():
        excluded_non_potable_totals.update(s["excluded_non_potable"])
    total_excluded_non_potable = sum(excluded_non_potable_totals.values())

    proposed_names = [d["proposed_component_name"] for d in manifest]
    unique_proposed_names = sorted(set(n.casefold() for n in proposed_names))

    report = {
        "plans_total": len(plan_ids),
        "plans_with_usable_legend": plans_with_usable_legend,
        "total_legend_entries_extracted": total_entries,
        "excluded_not_a_real_symbol_row": total_excluded_not_symbol,
        "excluded_privacy": total_excluded_privacy,
        "excluded_non_potable_water": total_excluded_non_potable,
        "excluded_non_potable_water_by_reason": dict(excluded_non_potable_totals.most_common()),
        "same_plan_page_duplicates_removed": dedup_removed_count,
        "remaining_potable_water_symbol_entries": len(manifest),
        "unique_proposed_component_types": len(unique_proposed_names),
        "proposed_component_types": unique_proposed_names,
        "per_plan_stats": per_plan_stats,
    }
    with open(OUT_DIR / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in report.items() if k != "per_plan_stats"}, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(manifest)} entries to {OUT_DIR / 'manifest.json'}")
    print(f"Wrote crops to {CROPS_DIR}")
    print(f"Wrote report to {OUT_DIR / 'build_report.json'}")


if __name__ == "__main__":
    main()
