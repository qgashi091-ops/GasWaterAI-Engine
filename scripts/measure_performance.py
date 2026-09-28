"""Measures analysis time and peak memory (RSS) for the real W-003 PDF,
covering v0.1 (parse + PlanFacts), v0.2 (component recognition), and v0.3
(legend intelligence).

Run with: python3 scripts/measure_performance.py
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plan_analysis.component_facts import build_component_facts
from app.plan_analysis.legend_intelligence import build_legend_intelligence
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "W-003_Referenzfall.Plan.pdf"
RUNS = 5


def main() -> None:
    pdf_bytes = FIXTURE.read_bytes()
    print(f"Fixture: {FIXTURE.name} ({len(pdf_bytes)} bytes)")

    parse_times, facts_times, component_times, legend_times, total_times = [], [], [], [], []
    for i in range(RUNS):
        t0 = time.perf_counter()
        doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
        t1 = time.perf_counter()
        plan_facts = build_document_facts(doc)
        t2 = time.perf_counter()
        component_facts = build_component_facts(pdf_bytes, doc)
        t3 = time.perf_counter()
        legend_intelligence = build_legend_intelligence(pdf_bytes, doc)
        t4 = time.perf_counter()
        parse_times.append(t1 - t0)
        facts_times.append(t2 - t1)
        component_times.append(t3 - t2)
        legend_times.append(t4 - t3)
        total_times.append(t4 - t0)
        if i == 0:
            print(f"plan_facts: fact_count={plan_facts['stats']['fact_count']} by_kind={plan_facts['stats']['by_kind']}")
            print(f"component_facts: {component_facts['stats']}")
            print(f"legend_intelligence: {legend_intelligence['stats']}")

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    def stats(label, values):
        print(f"{label}: min={min(values):.3f}s  max={max(values):.3f}s  avg={sum(values)/RUNS:.3f}s")

    print(f"\nRuns: {RUNS}")
    stats("Parse time (pipeline.analyze_pdf_bytes)               ", parse_times)
    stats("PlanFacts time (plan_facts.build_document_facts)      ", facts_times)
    stats("Component recognition (component_facts.build_component_facts)", component_times)
    stats("Legend intelligence (legend_intelligence.build_legend_intelligence)", legend_times)
    stats("Total (parse + plan_facts + component_facts + legend_intelligence)", total_times)
    print(f"Peak RSS for this process (cumulative over {RUNS} runs): {peak_rss_kb} KiB (~{peak_rss_kb/1024:.1f} MiB)")


if __name__ == "__main__":
    main()
