"""Unit + real-fixture tests for legend_detection.py (v0.3 Phase 1).

Covers the required test list: legend region detection (synthetic +
real), geometric separation from scattered drawing content, heading
corroboration, and 10-run reproducibility on the real W-003 fixture.
"""
from __future__ import annotations

from pathlib import Path

from app.fingerprint import canonical_hash
from app.plan_analysis import schema
from app.plan_analysis.legend_detection import detect_legend_candidates
from app.plan_analysis.pipeline import analyze_pdf_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"


def _span(id_, text, bbox, source="native"):
    return schema.TextSpan(id=id_, text=text, bbox=bbox, source=source)


def _page(spans, page_number=1, width=1000.0, height=1000.0):
    return schema.PageAnalysis(page_number=page_number, width=width, height=height,
                                text_source="native", text_spans=spans)


def test_a_dense_grid_of_short_rows_is_detected_as_a_legend_candidate():
    # 12 tightly-packed rows in a small area -- exactly what a real legend
    # column looks like -- vs. nothing else on the page.
    spans = [
        _span(f"t{i}", f"Symbol {i}", (50.0, 100.0 + i * 10.0, 140.0, 108.0 + i * 10.0))
        for i in range(12)
    ]
    page = _page(spans)
    candidates = detect_legend_candidates(page, rotation_matrix=None)
    assert len(candidates) == 1
    assert candidates[0].row_count == 12
    assert candidates[0].bbox[0] <= 50.0 and candidates[0].bbox[2] >= 140.0


def test_scattered_labels_across_a_busy_drawing_never_form_a_legend_candidate():
    # Same TOTAL number of spans as the positive case above, but spread far
    # apart across the page the way real pipe/component labels are --
    # must NOT cluster into a dense block.
    spans = [
        _span(f"t{i}", f"DN {i}", (i * 400.0, i * 350.0, i * 400.0 + 40.0, i * 350.0 + 10.0))
        for i in range(12)
    ]
    page = _page(spans, width=5000.0, height=5000.0)
    candidates = detect_legend_candidates(page, rotation_matrix=None)
    assert candidates == []


def test_a_heading_match_ranks_a_candidate_above_a_same_size_headingless_one():
    heading_spans = [_span("h0", "LEGENDE", (10.0, 0.0, 80.0, 10.0))] + [
        _span(f"a{i}", f"Item {i}", (10.0, 20.0 + i * 8.0, 90.0, 26.0 + i * 8.0)) for i in range(10)
    ]
    plain_spans = [
        _span(f"b{i}", f"Note {i}", (500.0, 20.0 + i * 8.0, 580.0, 26.0 + i * 8.0)) for i in range(10)
    ]
    page = _page(heading_spans + plain_spans, width=2000.0, height=2000.0)
    candidates = detect_legend_candidates(page, rotation_matrix=None)
    assert len(candidates) == 2
    assert candidates[0].heading_text == "LEGENDE"
    assert candidates[0].confidence > candidates[1].confidence


def test_w003_real_legend_is_detected_with_the_right_heading_and_extent():
    pdf_bytes = FIXTURE.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    page_model = doc.pages[0]

    import pymupdf
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = pdf[0]
        candidates = detect_legend_candidates(page_model, rotation_matrix=page.rotation_matrix)
    finally:
        pdf.close()

    assert len(candidates) >= 1
    top = candidates[0]
    assert top.heading_text == "LEGENDE SANITÄR"
    x0, y0, x1, y1 = top.bbox
    # The real legend sits roughly at display-space (42, 268)-(525, 502) --
    # allow generous slack rather than pinning exact pixels.
    assert 20 <= x0 <= 80
    assert 240 <= y0 <= 300
    assert x1 >= 400
    assert y1 <= 560
    assert top.row_count >= 25


def test_w003_legend_detection_is_reproducible_across_ten_runs():
    pdf_bytes = FIXTURE.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)

    import pymupdf
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = pdf[0]
        hashes = set()
        for _ in range(10):
            candidates = detect_legend_candidates(doc.pages[0], rotation_matrix=page.rotation_matrix)
            hashes.add(canonical_hash([c.to_dict() for c in candidates]))
    finally:
        pdf.close()

    assert len(hashes) == 1, "legend detection must be bit-identical across repeated runs of the same PDF"
