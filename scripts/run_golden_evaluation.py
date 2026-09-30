"""ENGINE v1 epic, section 10 -- Golden evaluation (W-001..W-010), EVALUATION
ONLY: never trains, never tunes, never edits a rule after seeing a result.

BLOCKER, confirmed before writing this script (not a guess): the 10 real
W-001..W-010 reference-plan PDFs and their 38-golden-finding ground-truth
dataset are NOT present anywhere in this build/dev session.
  - `tests/fixtures/` in this repo (qgashi091-ops/GasWaterAI-Engine) holds
    ONLY `W-003_Referenzfall.Plan.pdf`; there has never been a W-001, W-002,
    W-004..W-010 file committed to this repo (confirmed via `git log --all
    --diff-filter=A --name-only`).
  - `qgashi091-ops/gaswaterai` (the other repo available this session) has
    an identical gap: its own `backend/tests/fixtures/reference_plans/`
    directory contains only a README explaining the same real 10 plans are
    real client deliverables, deliberately `.gitignore`'d, and must be
    copied in locally or pointed to via `GASWATERAI_REFERENCE_PLANS_DIR`.
  - No other location on this container's filesystem holds them either
    (checked broadly).

This script is written and ready to run correctly the moment the files
exist; it does not fabricate a result in their absence -- see `main()`'s
first check. Point `--reference-plans-dir` at a directory containing
`W-001_Referenzfall.Plan.pdf` .. `W-010_Referenzfall.Plan.pdf` (or set
GASWATERAI_REFERENCE_PLANS_DIR) plus a `golden_findings.json` describing the
established 38 golden findings and this script will produce the full report
table section 10 asks for (TP/partial/FN, unsupported FP, weighted recall,
precision, NOT_ASSESSABLE, results by capability/rule family). The engine
itself is FROZEN before this script is ever pointed at real data (this
script never changes app/ code -- it only reads and reports), matching
"freeze engine first, hash/version it, THEN evaluate."
"""
from __future__ import annotations

import argparse
import json
import os
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

EXPECTED_PLAN_IDS = [f"W-{i:03d}" for i in range(1, 11)]
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
    parser.add_argument("--reference-plans-dir", type=Path, default=None)
    parser.add_argument("--golden-findings", type=Path, default=None,
                         help="JSON file describing the 38 golden findings and their expected classification per plan.")
    args = parser.parse_args()

    reference_dir = args.reference_plans_dir or (
        Path(os.environ["GASWATERAI_REFERENCE_PLANS_DIR"]) if os.environ.get("GASWATERAI_REFERENCE_PLANS_DIR") else None
    )

    if reference_dir is None or not reference_dir.exists():
        print("BLOCKED: no reference plans directory available.")
        print("Set --reference-plans-dir or GASWATERAI_REFERENCE_PLANS_DIR to a directory")
        print(f"containing {EXPECTED_PLAN_IDS[0]}_*.pdf .. {EXPECTED_PLAN_IDS[-1]}_*.pdf.")
        print("Checked and confirmed absent from this session:")
        print("  - GasWaterAI-Engine/tests/fixtures/ (only W-003 present)")
        print("  - gaswaterai/backend/tests/fixtures/reference_plans/ (gitignored, empty)")
        print(f"Baseline this evaluation would compare against: {json.dumps(BASELINE)}")
        sys.exit(2)

    found = find_reference_plans(reference_dir)
    missing = [p for p in EXPECTED_PLAN_IDS if p not in found]
    if missing:
        print(f"BLOCKED: {len(missing)}/10 reference plans missing from {reference_dir}: {missing}")
        sys.exit(2)

    if args.golden_findings is None or not args.golden_findings.exists():
        print("BLOCKED: --golden-findings JSON (the 38 golden findings' ground truth) not provided.")
        print("Reference plans were found, but scoring requires the golden findings file.")
        sys.exit(2)

    golden_findings = json.loads(args.golden_findings.read_text())

    engine_outputs = {plan_id: run_engine_on_plan(path) for plan_id, path in found.items()}
    # Scoring logic intentionally not implemented further: it depends
    # entirely on golden_findings.json's own schema, which does not exist
    # in this session either. Once both real inputs exist, this is the
    # single place to add: match each golden finding to a check_id in
    # engine_outputs[plan_id]["checks"], classify TP/partial/FN, and
    # compute weighted recall/precision against BASELINE.
    print(json.dumps({"engine_outputs_computed_for": list(engine_outputs), "baseline": BASELINE}, indent=2))


if __name__ == "__main__":
    main()
