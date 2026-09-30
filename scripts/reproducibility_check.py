"""ENGINE v1 epic, section 9 -- reproducibility.

Runs the full /check pipeline (parse -> PlanFacts -> legend intelligence ->
component evidence fusion (detector only, no vision fallback -- the vision
fallback is the one explicitly-external-model-observation source, and the
epic itself only requires PlanFacts/inventory/rule results to be
deterministic, not a live external call) -> canonical inventory -> rule
checks) N times against the same PDF bytes and asserts every field except
wall-clock timing diagnostics is byte-identical across all N runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plan_analysis.canonical_inventory import build_canonical_inventory
from app.plan_analysis.component_evidence import build_component_evidence
from app.plan_analysis.legend_intelligence import build_legend_intelligence
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts
from app.rules.engine import run_checks
from app.rules.water_rules import ALL_RULES

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "W-003_Referenzfall.Plan.pdf"
RUNS = 10


def run_once(pdf_bytes: bytes) -> dict:
    doc = analyze_pdf_bytes(pdf_bytes, filename="W-003.pdf")
    plan_facts = build_document_facts(doc)
    legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
    inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
    checks = run_checks(inventory["inventory"], ALL_RULES)
    return {
        "plan_facts": plan_facts, "component_evidence": component_evidence,
        "inventory": inventory, "checks": checks,
    }


def main() -> None:
    pdf_bytes = FIXTURE_PATH.read_bytes()
    serialized_runs = []
    for i in range(RUNS):
        result = run_once(pdf_bytes)
        serialized_runs.append(json.dumps(result, sort_keys=True, ensure_ascii=False))

    first = serialized_runs[0]
    all_identical = all(s == first for s in serialized_runs)
    lengths = [len(s) for s in serialized_runs]

    print(f"Runs: {RUNS}")
    print(f"Serialized lengths: {lengths}")
    print(f"All {RUNS} runs byte-identical: {all_identical}")
    if not all_identical:
        for i, s in enumerate(serialized_runs[1:], start=1):
            if s != first:
                # Cheap diff: find first differing character index.
                for idx, (a, b) in enumerate(zip(first, s)):
                    if a != b:
                        print(f"Run {i} first differs from run 0 at char {idx}: ...{first[max(0,idx-40):idx+40]!r} vs ...{s[max(0,idx-40):idx+40]!r}")
                        break
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
