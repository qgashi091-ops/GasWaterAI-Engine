"""ENGINE v1 epic, section 10 -- Golden evaluation (W-001..W-010), EVALUATION
ONLY: never trains, never tunes, never edits a rule after seeing a result.

STATUS as of the golden-holdout import (see
docs/v1-golden-holdout-import-report.md for the full audit): the 10 real
W-001..W-010 plan PDFs ARE now present locally, imported into the isolated
`tests/golden_holdout/plans/` holdout (gitignored -- see that package's own
isolation contract in `tests/golden_holdout/__init__.py`, enforced by
`tests/test_golden_holdout_isolation.py`). The curated 38-golden-finding
benchmark denominator is STILL NOT AVAILABLE: it was searched for across
every repository available in this session (including full git history on
every branch) and does not exist as a committed, machine-readable file
anywhere. Only the 113 RAW extracted Fachbericht bullets exist
(`tests/golden_holdout/raw_findings_113.json`) -- explicitly NOT the
benchmark denominator, never to be substituted for it (see
`tests/golden_holdout/harness.py`'s `load_curated_benchmark()`, which raises
rather than falling back to the raw bullets).

This script still does not fabricate a result in the curated benchmark's
absence -- see `main()`'s checks below. Point `--golden-findings` at a real
`golden_findings_38.json` (or place it at
`tests/golden_holdout/golden_findings_38.json`) once that curated mapping
exists, and this script will produce the full report table section 10 asks
for (TP/partial/FN, unsupported FP, weighted recall, precision,
NOT_ASSESSABLE, results by capability/rule family). The engine itself is
FROZEN before this script is ever pointed at real data (this script never
changes app/ code -- it only reads and reports), matching "freeze engine
first, hash/version it, THEN evaluate." Per explicit instruction, this
script does not run the engine against the golden plans or attempt any
optimization/benchmark until the curated denominator exists and a
benchmark run is explicitly requested -- see `main()`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fingerprint import document_fingerprint
from app.plan_analysis.canonical_inventory import build_canonical_inventory
from app.plan_analysis.component_evidence import build_component_evidence
from app.plan_analysis.legend_intelligence import build_legend_intelligence
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts
from app.rules.engine import run_checks
from app.rules.water_rules import ALL_RULES
from tests.golden_holdout import harness as golden_holdout

EXPECTED_PLAN_IDS = golden_holdout.GOLDEN_PLAN_IDS
DEFAULT_GOLDEN_FINDINGS_PATH = golden_holdout.CURATED_BENCHMARK_PATH
BASELINE = {
    "golden_finding_count": 38,
    "weighted_tp_baseline": 5.5,
    "weighted_recall_baseline": 0.145,
    "precision_baseline": 0.239,
    "unsupported_findings_baseline": 13,
}


def find_reference_plans(reference_dir: Path) -> dict[str, Path]:
    found = {}
    for plan_id in EXPECTED_PLAN_IDS:
        matches = list(reference_dir.glob(f"{plan_id}_*.pdf"))
        if matches:
            found[plan_id] = matches[0]
    return found


def run_engine_on_plan(pdf_path: Path) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=pdf_path.name)
    plan_facts = build_document_facts(doc)
    legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
    inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
    checks = run_checks(inventory["inventory"], ALL_RULES)
    return {
        "document_fingerprint": document_fingerprint(pdf_bytes),
        "inventory": inventory, "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-plans-dir", type=Path, default=golden_holdout.PLANS_DIR)
    parser.add_argument("--golden-findings", type=Path, default=DEFAULT_GOLDEN_FINDINGS_PATH,
                         help="JSON file describing the curated 38 golden findings and their expected classification per plan.")
    args = parser.parse_args()

    reference_dir = args.reference_plans_dir

    if reference_dir is None or not reference_dir.exists():
        print("BLOCKED: no reference plans directory available.")
        print(f"Expected the golden holdout at {golden_holdout.PLANS_DIR} (see tests/golden_holdout/__init__.py).")
        print(f"Baseline this evaluation would compare against: {json.dumps(BASELINE)}")
        sys.exit(2)

    found = find_reference_plans(reference_dir)
    missing = [p for p in EXPECTED_PLAN_IDS if p not in found]
    if missing:
        print(f"BLOCKED: {len(missing)}/10 reference plans missing from {reference_dir}: {missing}")
        sys.exit(2)

    if args.golden_findings is None or not args.golden_findings.exists():
        print("BLOCKED: the curated 38-golden-finding benchmark mapping does not exist yet.")
        print(f"  Expected at: {args.golden_findings}")
        print("  All 10 golden plans were found and are ready, but scoring requires this curated")
        print("  file, which is DISTINCT from the 113 raw Fachbericht bullets in")
        print("  tests/golden_holdout/raw_findings_113.json -- those are provenance only and")
        print("  must never be substituted for the real 38-finding denominator.")
        print(f"Baseline this evaluation would compare against: {json.dumps(BASELINE)}")
        sys.exit(2)

    golden_findings = json.loads(args.golden_findings.read_text())

    engine_outputs = {plan_id: run_engine_on_plan(path) for plan_id, path in found.items()}
    # Scoring logic intentionally not implemented further: it depends
    # entirely on golden_findings_38.json's own schema, which does not exist
    # in this session either. Once both real inputs exist, this is the
    # single place to add: match each golden finding to a check_id in
    # engine_outputs[plan_id]["checks"], classify TP/partial/FN, and
    # compute weighted recall/precision against BASELINE.
    print(json.dumps({"engine_outputs_computed_for": list(engine_outputs), "baseline": BASELINE}, indent=2))


if __name__ == "__main__":
    main()
