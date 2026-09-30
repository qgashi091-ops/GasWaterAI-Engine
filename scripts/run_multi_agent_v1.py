"""CLI test path for the Multi-Agent Architecture v1 -- separate from the
existing /analyze and /check flows, per this epic's "separaten API-/CLI-
Testpfad multi_agent_v1" requirement.

Usage:
    PYTHONPATH=. python3 scripts/run_multi_agent_v1.py <plan.pdf> [--out result.json]

Runs the exact same deterministic chain /multi_agent_v1/analyze does
(analyze_pdf_bytes -> plan_facts -> legend_intelligence -> component_evidence
-> canonical_inventory -> rule checks -> run_multi_agent_v1), so a plan can
be exercised end-to-end from the command line with no server running. Uses
AnthropicAgentModelProvider -- with no GASWATERAI_VISION_API_KEY configured
(none exists in this environment), every model-dependent claim reports
`available: false` honestly; every deterministic-first agent path still
produces real results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.multi_agent_v1.pipeline import run_multi_agent_v1  # noqa: E402
from app.multi_agent_v1.provider import AnthropicAgentModelProvider  # noqa: E402
from app.plan_analysis.canonical_inventory import build_canonical_inventory  # noqa: E402
from app.plan_analysis.component_evidence import build_component_evidence  # noqa: E402
from app.plan_analysis.legend_intelligence import build_legend_intelligence  # noqa: E402
from app.plan_analysis.pipeline import analyze_pdf_bytes  # noqa: E402
from app.plan_analysis.plan_facts import build_document_facts  # noqa: E402
from app.rules.engine import run_checks  # noqa: E402
from app.rules.water_rules import ALL_RULES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write the full JSON result here (default: print a summary only).")
    args = parser.parse_args()

    pdf_bytes = args.pdf_path.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=args.pdf_path.name)
    plan_facts = build_document_facts(doc)

    try:
        legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
    except Exception as exc:  # noqa: BLE001
        legend_intelligence = {"legend_candidates": [], "legend_entries": [], "component_facts": [], "stats": {"error": str(exc)}}

    try:
        component_evidence = build_component_evidence(pdf_bytes, doc, legend_intelligence, vision_provider=None)
    except Exception as exc:  # noqa: BLE001
        component_evidence = {"evidence": [], "stats": {"error": str(exc)}}

    inventory = build_canonical_inventory(doc, plan_facts, component_evidence)
    check_results = run_checks(inventory["inventory"], ALL_RULES)

    t0 = time.perf_counter()
    result = run_multi_agent_v1(
        doc=doc, provider=AnthropicAgentModelProvider(), pdf_bytes=pdf_bytes, plan_facts=plan_facts,
        component_evidence=component_evidence.get("evidence", []), inventory=inventory["inventory"],
        rule_checks=check_results["checks"],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    payload = result.to_dict()
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"Wrote {args.out}")

    diag = payload["canonical_plan_understanding"]["diagnostics"]
    print(f"engine_version={payload['engine_version']}  elapsed_ms={elapsed_ms:.1f}")
    print(f"agents routed: {diag['totals']['agents_routed']}  skipped: {diag['totals']['agents_skipped']} ({', '.join(diag['totals']['skipped_agent_ids'])})")
    print(f"model calls made: {diag['totals']['model_calls_made']}  cached: {diag['totals']['cached_hits']}")
    stats = payload["canonical_plan_understanding"]["stats"]
    print(f"claims: {stats['claims_total']}  by_resolution={stats['by_resolution']}  conflicts={stats['conflicts_total']}")


if __name__ == "__main__":
    main()
