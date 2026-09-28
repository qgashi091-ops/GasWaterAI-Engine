"""The primary acceptance criterion for this POC: running the SAME real
W-003 PDF through the engine repeatedly must produce semantically identical
PlanFacts every time. Timing metadata is allowed to differ (it lives only
under diagnostics.timing_ms in the API response, never inside plan_facts
itself -- see app/main.py) -- the technical facts may not.
"""
from __future__ import annotations

from pathlib import Path

from app.fingerprint import canonical_hash, document_fingerprint
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.plan_facts import build_document_facts

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
RUNS = 10


def test_w003_document_fingerprint_is_stable():
    pdf_bytes = FIXTURE.read_bytes()
    fp1 = document_fingerprint(pdf_bytes)
    fp2 = document_fingerprint(pdf_bytes)
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex digest


def test_w003_ten_repeated_runs_produce_semantically_identical_plan_facts():
    pdf_bytes = FIXTURE.read_bytes()
    hashes = []
    fact_counts = []
    for _ in range(RUNS):
        doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
        plan_facts = build_document_facts(doc)
        hashes.append(canonical_hash(plan_facts))
        fact_counts.append(plan_facts["stats"]["fact_count"])

    assert len(set(hashes)) == 1, (
        f"plan_facts differed across {RUNS} runs of the identical PDF -- "
        f"got {len(set(hashes))} distinct canonical hashes, expected 1"
    )
    assert len(set(fact_counts)) == 1
    assert fact_counts[0] > 0, "the real W-003 plan must yield a non-trivial fact set"
