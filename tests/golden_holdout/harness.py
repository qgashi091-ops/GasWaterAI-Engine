"""Golden holdout harness -- see this package's __init__.py for the
isolation contract. Python analogue of the Base44 app's
`goldenTestHarness.ts`, for the same 10 real reference plans.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS_DIR = HERE / "plans"  # gitignored -- real client PDFs, never committed (see repo .gitignore)
RAW_FINDINGS_PATH = HERE / "raw_findings_113.json"  # committed -- PII-scrubbed at extraction time (see provenance block inside)
CURATED_BENCHMARK_PATH = HERE / "golden_findings_38.json"  # DOES NOT EXIST YET -- see status function below

GOLDEN_PLAN_IDS = [f"W-{i:03d}" for i in range(1, 11)]

# Confirmed by direct content comparison (identical extracted text, 2604
# chars, and identical vector drawing-item count, 20308) at import time:
# W-010 is the SAME underlying plan as W-002, re-exported/re-saved under a
# different filename and timestamp -- not an 11th independent case. Anyone
# curating the 38-finding benchmark denominator from these 10 "cases" needs
# to decide how W-010 counts (a genuine 10th independent finding source, or
# a determinism/reproducibility check on W-002) -- not decided here.
KNOWN_CONTENT_DUPLICATES = {"W-010": "W-002"}


@dataclass
class GoldenPlan:
    plan_id: str
    path: Path
    pdf_bytes: bytes


def available_plan_ids() -> list[str]:
    """Which golden plan PDFs actually exist on disk right now (this
    directory is gitignored, so a fresh checkout has none until they are
    supplied locally -- mirrors the existing `tests/fixtures/` real-plan
    gating convention already used elsewhere in this repo)."""
    if not PLANS_DIR.exists():
        return []
    found = []
    for plan_id in GOLDEN_PLAN_IDS:
        if list(PLANS_DIR.glob(f"{plan_id}_*.pdf")):
            found.append(plan_id)
    return found


def load_golden_plan(plan_id: str) -> GoldenPlan | None:
    if not PLANS_DIR.exists():
        return None
    matches = list(PLANS_DIR.glob(f"{plan_id}_*.pdf"))
    if not matches:
        return None
    path = matches[0]
    return GoldenPlan(plan_id=plan_id, path=path, pdf_bytes=path.read_bytes())


def load_raw_findings(plan_id: str) -> dict | None:
    """The 113 RAW extracted Fachbericht bullets -- NOT the 38-finding
    benchmark denominator. See raw_findings_113.json's own `_provenance`
    block. Useful only as provenance/context, never as a scoring input
    until a real curated mapping exists."""
    if not RAW_FINDINGS_PATH.exists():
        return None
    data = json.loads(RAW_FINDINGS_PATH.read_text())
    return data["fachberichte"].get(plan_id)


def curated_benchmark_available() -> bool:
    return CURATED_BENCHMARK_PATH.exists()


def load_curated_benchmark() -> dict:
    """The actual 38-golden-finding benchmark denominator. Raises
    explicitly rather than silently falling back to the 113 raw bullets --
    per instruction, no caller may treat the raw bullets as this mapping."""
    if not CURATED_BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            "The curated 38-golden-finding benchmark mapping does not exist "
            f"at {CURATED_BENCHMARK_PATH}. It was not found in any of the "
            "repositories available in this session (see "
            "docs/v1-golden-holdout-import-report.md) and must be supplied "
            "or curated before a real benchmark run -- it must never be "
            "fabricated from the 113 raw findings or the Fachbericht text."
        )
    return json.loads(CURATED_BENCHMARK_PATH.read_text())
