"""Measures analysis time and peak memory (RSS) for the real W-003 PDF.

Run with: python3 scripts/measure_performance.py
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "W-003_Referenzfall.Plan.pdf"
RUNS = 5


def main() -> None:
    pdf_bytes = FIXTURE.read_bytes()
    print(f"Fixture: {FIXTURE.name} ({len(pdf_bytes)} bytes)")

    parse_times = []
    facts_times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
        t1 = time.perf_counter()
        plan_facts = build_document_facts(doc)
        t2 = time.perf_counter()
        parse_times.append(t1 - t0)
        facts_times.append(t2 - t1)
        if i == 0:
            print(f"fact_count={plan_facts['stats']['fact_count']} by_kind={plan_facts['stats']['by_kind']}")

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    print(f"\nRuns: {RUNS}")
    print(f"Parse time (pipeline.analyze_pdf_bytes):  min={min(parse_times):.3f}s  max={max(parse_times):.3f}s  avg={sum(parse_times)/RUNS:.3f}s")
    print(f"PlanFacts time (plan_facts.build_document_facts): min={min(facts_times):.3f}s  max={max(facts_times):.3f}s  avg={sum(facts_times)/RUNS:.3f}s")
    print(f"Peak RSS for this process (cumulative over {RUNS} runs): {peak_rss_kb} KiB (~{peak_rss_kb/1024:.1f} MiB)")


if __name__ == "__main__":
    main()
