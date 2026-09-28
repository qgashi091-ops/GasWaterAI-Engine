"""Locates the garden-valve ("Gartenventil") topology in W-003 by TEXT
VOCABULARY, not by hardcoded coordinates or filenames -- this technique
generalizes to any plan: search extracted text for domain vocabulary, follow
the existing text-to-geometry association, then read off that edge/symbol's
PlanFacts topology classification.

Run with: python3 scripts/w003_topology_probe.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plan_analysis.pipeline import analyze_pdf_file
from app.plan_analysis.plan_facts import build_document_facts

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "W-003_Referenzfall.Plan.pdf"
GARDEN_VOCABULARY = re.compile(r"gartenventil", re.IGNORECASE)


def main() -> None:
    doc = analyze_pdf_file(str(FIXTURE))
    page = doc.pages[0]
    plan_facts = build_document_facts(doc)

    matches = [t for t in page.text_spans if GARDEN_VOCABULARY.search(t.text)]
    assoc_by_text = {a.text_id: a for a in page.associations}
    symbol_by_id = {s.id: s for s in page.symbols}

    def topology_facts_for_edge(edge_id: str):
        return [f for f in plan_facts["facts"] if edge_id in f["supporting_edges"]
                and f["fact_type"] in ("cycle_membership", "dead_end_path", "bridge_continuity")]

    print(f"Found {len(matches)} garden-valve-vocabulary text spans on page {page.page_number}\n")
    for t in matches:
        print(f"label={t.text!r}")
        a = assoc_by_text.get(t.id)
        if a is None or a.target_type == "unassigned":
            print("  -> no direct geometric association (unassigned)\n")
            continue
        print(f"  associated to {a.target_type} {a.target_id} (distance={a.distance:.2f}pt)")
        edge_ids = [a.target_id] if a.target_type == "edge" else (
            [e.id for n in symbol_by_id[a.target_id].port_node_ids
             for e in page.graph.edges if e.source == n or e.target == n]
        )
        for eid in edge_ids:
            for f in topology_facts_for_edge(eid):
                print(f"    edge {eid}: {f['fact_type']}={f['value']} kind={f['kind']} evidence={f['evidence_status']}")
        print()


if __name__ == "__main__":
    main()
