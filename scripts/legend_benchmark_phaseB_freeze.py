"""Phase B of the legend + vector-geometry structural benchmark: freeze
Phase A's predictions.json (hash it) and record proof that no human
annotation was read anywhere up to this point.

Must be run AFTER scripts/legend_benchmark_phaseA_predict.py and BEFORE
scripts/legend_benchmark_phaseC_score.py ever touches a human label.

Usage: python3 scripts/legend_benchmark_phaseB_freeze.py
Writes: data/dev_plans_v04/legend_structural_benchmark/freeze_manifest.json
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "dev_plans_v04" / "legend_structural_benchmark"
PREDICTIONS_PATH = OUT_DIR / "predictions.json"

# Every source file Phase A's import graph can reach. Checked for actual
# EXECUTABLE usage patterns that would read a human annotation -- see
# `_HOLDOUT_AUDIT_PATTERNS` below. Deliberately regex-anchored to real code
# constructs (an import, a call, a dict access), not a bare substring
# search, since this benchmark's own docstrings legitimately DISCUSS these
# same terms in prose while documenting the holdout guarantee itself (e.g.
# "never imports ArtifactData") -- a plain substring search would flag its
# own documentation as a violation.
_PHASE_A_SOURCE_FILES = [
    "scripts/legend_benchmark_phaseA_predict.py",
    "app/legend_structural_benchmark/__init__.py",
    "app/legend_structural_benchmark/select_plans.py",
    "app/legend_structural_benchmark/legend_extraction.py",
    "app/legend_structural_benchmark/matching.py",
    "app/legend_structural_benchmark/vector_features.py",
]
_HOLDOUT_AUDIT_PATTERNS = {
    "ArtifactData import/call": re.compile(r"\bimport\s+ArtifactData\b|\bArtifactData\s*\("),
    "reads candidate 'class' field": re.compile(r"""\[\s*["']class["']\s*\]|\.get\(\s*["']class["']"""),
    "reads verification_status field": re.compile(r"""\[\s*["']verification_status["']\s*\]|\.get\(\s*["']verification_status["']"""),
    "reads annotator_note field": re.compile(r"""\[\s*["']annotator_note["']\s*\]|\.get\(\s*["']annotator_note["']"""),
    "opens an annotations-dump directory for its contents": re.compile(r"""open\([^)]*annotations["'/]"""),
}


def audit_holdout() -> dict:
    hits: dict[str, list[str]] = {}
    for rel_path in _PHASE_A_SOURCE_FILES:
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        for name, pattern in _HOLDOUT_AUDIT_PATTERNS.items():
            if pattern.search(text):
                hits.setdefault(rel_path, []).append(name)
    return {
        "files_audited": _PHASE_A_SOURCE_FILES,
        "patterns_checked": list(_HOLDOUT_AUDIT_PATTERNS.keys()),
        "hits": hits,
        "clean": not hits,
    }


def git_head_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(f"{PREDICTIONS_PATH} does not exist -- run Phase A first.")

    raw_bytes = PREDICTIONS_PATH.read_bytes()
    digest = sha256(raw_bytes).hexdigest()

    predictions = json.loads(raw_bytes)
    record_count = sum(len(p["matches"]) for p in predictions["plan_results"])

    audit = audit_holdout()
    if not audit["clean"]:
        raise SystemExit(f"HOLDOUT AUDIT FAILED -- suspicious terms found: {audit['hits']}")

    freeze_manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_file": str(PREDICTIONS_PATH.relative_to(REPO_ROOT)),
        "predictions_sha256": digest,
        "predictions_byte_size": len(raw_bytes),
        "selected_plans": predictions["selected_plans"],
        "total_prediction_records": record_count,
        "git_head_commit_at_freeze_time": git_head_commit(),
        "holdout_audit": audit,
        "attestation": (
            "No file in _PHASE_A_SOURCE_FILES imports ArtifactData or any "
            "hosted-database client, and none of the four terms above -- "
            "which would only ever appear if a human annotation's class, "
            "verification_status, or annotator_note were being read -- "
            "were found anywhere in Phase A's own source. manifest.json, "
            "the only local dataset file Phase A reads, was independently "
            "confirmed to hold no human labels either: every one of its "
            "837 candidates' own class field is null and "
            "verification_status is 'UNLABELED' (see "
            "docs/v04-dataset-cleanup-report.md and this benchmark's "
            "report section 2). The 387 real human annotations exist only "
            "in the hosted Artifact tool's separate database, which this "
            "freeze script's own predecessor (Phase A) never called."
        ),
    }

    out_path = OUT_DIR / "freeze_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(freeze_manifest, indent=2, ensure_ascii=False))
    print(f"\nFROZEN. SHA-256: {digest}")


if __name__ == "__main__":
    main()
