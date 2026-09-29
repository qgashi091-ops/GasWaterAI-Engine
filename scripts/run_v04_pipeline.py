"""v0.4 orchestrator: runs every phase (audit, privacy, candidate
generation, taxonomy, families/splits, export manifest, QC) over the 20 raw
dev plans and writes every artifact the annotation tool and the final
report need, under data/dev_plans_v04/annotation_dataset/ (pseudonymous,
safe to commit) -- never touching data/dev_plans_v04/raw/ or the local
id-mapping file (never committed, see .gitignore).

Each plan's expensive engine run happens in its OWN subprocess with a hard
timeout (_v04_plan_worker.py), so one pathological plan can't hang the
whole batch -- see that script's docstring.

Usage: python3 scripts/run_v04_pipeline.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.dataset_pipeline.audit import PlanAudit, summarize_batch  # noqa: E402
from app.dataset_pipeline.candidates import AnnotationCandidate  # noqa: E402
from app.dataset_pipeline.export import write_class_list, write_manifest  # noqa: E402
from app.dataset_pipeline.families import detect_near_duplicates, group_families  # noqa: E402
from app.dataset_pipeline.privacy import scan_filename  # noqa: E402
from app.dataset_pipeline.qc import run_qc  # noqa: E402
from app.dataset_pipeline.splits import design_split  # noqa: E402
from app.dataset_pipeline.taxonomy import (  # noqa: E402
    aggregate_evidence, is_plausible_component_label, normalize_legend_label, rank_classes,
)

RAW_DIR = ROOT / "data" / "dev_plans_v04" / "raw"
DATASET_DIR = ROOT / "data" / "dev_plans_v04" / "annotation_dataset"
CROPS_DIR = DATASET_DIR / "crops"
MAPPING_PATH = ROOT / "data" / "dev_plans_v04" / "id_mapping.LOCAL_ONLY.json"
# 240s was the initial budget; two of this batch's 20 real plans (DEV-07,
# DEV-10) hit it and were re-diagnosed by hand with a generous one-off
# timeout: both are genuinely large, unusually complex vector drawings
# (300k+ raw drawing primitives vs. a few thousand for a typical plan in
# this batch -- DEV-07's page is also exceptionally wide, ~14287pt, a
# multi-riser schema), not a hang -- both complete deterministically and
# correctly (parser_outcome "ok") in ~475-520s with the SAME unchanged
# engine and thresholds. Raising the wall-clock budget here is a batch
# operational setting applied uniformly to all 20 plans, not a per-plan
# threshold tuned to their content -- see docs/v04-dataset-report.md's
# Performance section for the measured runtimes this value is based on.
PER_PLAN_TIMEOUT_SECONDS = 700


def _run_worker_isolated(cmd: list, timeout_s: int, cwd: str) -> subprocess.CompletedProcess:
    """Runs a worker in its own process GROUP (start_new_session) so that on
    a timeout we can kill it AND every descendant it spawned -- specifically
    tesseract, invoked by v0.1's text.py OCR fallback for a scanned plan.
    Found the hard way on this batch: a single OCR tile can make tesseract
    hang for minutes on pathological input, and plain subprocess.run(...,
    timeout=...) only ever signals the direct child, silently leaving a
    runaway tesseract process (and its CPU/memory) behind for every such
    plan -- across a 20-plan batch that accumulates fast."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()  # reap
        raise


def _dict_to_audit(d: dict) -> PlanAudit:
    return PlanAudit(**{k: d.get(k) for k in PlanAudit.__dataclass_fields__})


def _dict_to_candidate(d: dict) -> AnnotationCandidate:
    return AnnotationCandidate(
        candidate_id=d["candidate_id"], plan_id=d["plan_id"], page=d["page"],
        bbox=tuple(d["bbox"]), crop_bbox=tuple(d["crop_bbox"]), bbox_in_crop_px=tuple(d["bbox_in_crop_px"]),
        crop_path=d["crop_path"], nearby_text=d["nearby_text"], graph_association=d["graph_association"],
        plan_specific_suggestion=d["plan_specific_suggestion"], generic_suggestion=d["generic_suggestion"],
        hybrid_kind=d["hybrid_kind"], status=d["status"], assigned_class=d["assigned_class"],
        corrected_bbox=tuple(d["corrected_bbox"]) if d["corrected_bbox"] else None,
        annotator_note=d["annotator_note"],
    )


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    plan_ids = sorted(p.stem for p in RAW_DIR.glob("DEV-*.pdf"))
    print(f"{len(plan_ids)} plans found")

    with open(MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)

    audits: list[PlanAudit] = []
    all_candidates: list[AnnotationCandidate] = []
    privacy_reports: list[dict] = []
    taxonomy_evidence_per_plan: list[dict] = []
    filename_privacy: dict[str, dict] = {}

    for plan_id in plan_ids:
        flagged, reason = scan_filename(mapping[plan_id]["original_filename"])
        filename_privacy[plan_id] = {"flagged": flagged, "reason": reason}

        pdf_path = RAW_DIR / f"{plan_id}.pdf"
        out_json = DATASET_DIR / f"_worker_{plan_id}.tmp.json"
        t0 = time.time()
        try:
            proc = _run_worker_isolated(
                [sys.executable, str(ROOT / "scripts" / "_v04_plan_worker.py"), plan_id, str(pdf_path), str(CROPS_DIR), str(out_json)],
                timeout_s=PER_PLAN_TIMEOUT_SECONDS, cwd=str(ROOT),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"worker exited {proc.returncode}: {proc.stderr[-1500:]}")
            with open(out_json, encoding="utf-8") as f:
                result = json.load(f)
            out_json.unlink(missing_ok=True)
            audit = _dict_to_audit(result["audit"])
            all_candidates.extend(_dict_to_candidate(c) for c in result["candidates"])
            privacy_reports.append(result["privacy"])
            taxonomy_evidence_per_plan.append(result["taxonomy_evidence"])
        except subprocess.TimeoutExpired:
            audit = PlanAudit(plan_id=plan_id, page_count=0, parser_outcome="timeout",
                               error=f"exceeded {PER_PLAN_TIMEOUT_SECONDS}s wall-clock budget",
                               runtime_seconds=round(time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001
            audit = PlanAudit(plan_id=plan_id, page_count=0, parser_outcome="subprocess_crashed",
                               error=str(exc), runtime_seconds=round(time.time() - t0, 1))
        audits.append(audit)
        print(f"{plan_id}: {audit.parser_outcome} ({audit.runtime_seconds}s, "
              f"symbols={audit.symbol_candidates}, legend={audit.legend_detected})")

    # Families / near-duplicates / splits (pseudonymous output only)
    original_filenames = {pid: v["original_filename"] for pid, v in mapping.items()}
    families = group_families(original_filenames)
    plans_bytes = {pid: (RAW_DIR / f"{pid}.pdf").read_bytes() for pid in plan_ids}
    near_dupes = detect_near_duplicates(plans_bytes)
    split = design_split(families)

    # Taxonomy: each worker subprocess already extracted the minimal
    # evidence needed (legend entries, safety-device codes, generic
    # suggestions) from its own plan, avoiding a third re-parse here --
    # aggregate_evidence/rank_classes apply the SAME support/ranking rule
    # propose_taxonomy uses, just fed from these pre-extracted counters.
    legend_counts, safety_counts, generic_counts = Counter(), Counter(), Counter()
    for ev in taxonomy_evidence_per_plan:
        for entry in ev.get("legend_entries", []):
            if entry["classification"] == "USABLE_TEMPLATE" and not entry["is_line_style_swatch"]:
                label = entry["normalized_label"].strip()
                if label and is_plausible_component_label(label):
                    legend_counts[normalize_legend_label(label)] += 1
        for code in ev.get("safety_device_codes", []):
            safety_counts[code] += 1
        for fact in ev.get("component_facts", []):
            generic_ev = fact.get("generic_evidence")
            if generic_ev and generic_ev.get("symbol_name"):
                generic_counts[generic_ev["symbol_name"].strip().casefold()] += 1

    evidence_counts = aggregate_evidence(legend_counts, safety_counts, generic_counts)
    taxonomy_classes = rank_classes(evidence_counts)

    qc_report = run_qc(all_candidates)

    manifest_info = write_manifest(all_candidates, families, split.plan_to_split, str(DATASET_DIR / "manifest.json"))
    write_class_list(taxonomy_classes, str(DATASET_DIR / "classes.json"))

    summary = {
        "batch_summary": summarize_batch(audits),
        "audits": [a.to_dict() for a in audits],
        "families": families,
        "near_duplicate_pairs": [{"plan_id_a": p.plan_id_a, "plan_id_b": p.plan_id_b, "hamming_distance": p.hamming_distance} for p in near_dupes],
        "split": split.to_dict(),
        "privacy": {"per_plan_text_scan": privacy_reports, "filename_scan": filename_privacy},
        "taxonomy": {"classes": taxonomy_classes, "evidence_counts": evidence_counts},
        "qc": qc_report.to_dict(),
        "manifest": manifest_info,
        "total_candidates": len(all_candidates),
    }
    with open(DATASET_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nTotal candidates: {len(all_candidates)}")
    print(f"Wrote {DATASET_DIR / 'pipeline_summary.json'}, manifest.json, classes.json, {len(list(CROPS_DIR.glob('*.png')))} crop PNGs")


if __name__ == "__main__":
    main()
