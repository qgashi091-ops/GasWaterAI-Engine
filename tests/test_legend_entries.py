"""Unit + real-fixture tests for legend_entries.py (v0.3 Phase 2).

Covers the required test list: symbol/text row association, table-border
contamination, line-style-swatch exclusion, ambiguous label association,
and rejection of a coded lookup table (W-003's real "Legende
Verteilbatterie" lookalike).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.plan_analysis import schema
from app.plan_analysis.legend_detection import LegendCandidate, LegendRow, detect_legend_candidates
from app.plan_analysis.legend_entries import (
    CODE_ASSIGNMENT_PATTERN,
    _internal_gap_reason,
    _looks_like_border,
    _looks_like_line_style_swatch,
    classify_candidate_rows,
)
from app.plan_analysis.pipeline import analyze_pdf_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"


def _span(id_, text, bbox, source="native"):
    return schema.TextSpan(id=id_, text=text, bbox=bbox, source=source)


def _row(bbox, spans):
    return LegendRow(bbox=bbox, text=" ".join(s.text for s, _ in spans), spans=spans)


def test_internal_gap_flags_a_row_as_ambiguous():
    spans = [
        (_span("a", "Filter", (10.0, 0.0, 40.0, 8.0)), (10.0, 0.0, 40.0, 8.0)),
        (_span("b", "Endabschluss", (200.0, 0.0, 260.0, 8.0)), (200.0, 0.0, 260.0, 8.0)),
    ]
    row = _row((10.0, 0.0, 260.0, 8.0), spans)
    reason = _internal_gap_reason(row)
    assert reason is not None and "merged entries" in reason


def test_a_tight_single_label_row_has_no_ambiguity_reason():
    spans = [(_span("a", "Ventil", (10.0, 0.0, 40.0, 8.0)), (10.0, 0.0, 40.0, 8.0))]
    row = _row((10.0, 0.0, 40.0, 8.0), spans)
    assert _internal_gap_reason(row) is None


def _mask_image(pattern: np.ndarray) -> np.ndarray:
    # pattern: 1 == ink. Converts to a grayscale image the same way a real
    # render would look (white background, dark ink) so the real threshold
    # in _looks_like_border/_looks_like_line_style_swatch applies unchanged.
    gray = np.full(pattern.shape, 255, dtype=np.uint8)
    gray[pattern == 1] = 0
    return gray


def test_a_full_width_thin_line_is_flagged_as_a_table_border():
    pattern = np.zeros((60, 200), dtype=np.uint8)
    pattern[2:4, :] = 1  # a thin horizontal rule spanning the full zone width
    assert bool(_looks_like_border(_mask_image(pattern))) is True


def test_a_compact_icon_shaped_blob_is_not_flagged_as_a_border():
    pattern = np.zeros((60, 200), dtype=np.uint8)
    pattern[15:45, 80:120] = 1  # a centered, margin-on-all-sides blob
    assert bool(_looks_like_border(_mask_image(pattern))) is False


def test_a_solid_thin_line_is_a_line_style_swatch():
    pattern = np.zeros((60, 200), dtype=np.uint8)
    pattern[29:31, 10:190] = 1  # 2px-tall solid line
    assert bool(_looks_like_line_style_swatch(_mask_image(pattern))) is True


def test_a_dashed_line_with_many_islands_is_a_line_style_swatch():
    pattern = np.zeros((60, 200), dtype=np.uint8)
    for x in range(10, 190, 15):
        pattern[27:33, x:x + 8] = 1  # many short dashes along a thin band
    assert bool(_looks_like_line_style_swatch(_mask_image(pattern))) is True


def test_a_multi_stroke_icon_is_not_a_line_style_swatch():
    pattern = np.zeros((60, 200), dtype=np.uint8)
    pattern[10:50, 60:70] = 1     # vertical stroke
    pattern[10:50, 130:140] = 1   # another vertical stroke
    pattern[28:32, 10:190] = 1    # connecting horizontal line
    pattern[5:15, 90:110] = 1     # a small extra blob (e.g. a filled arrowhead)
    assert bool(_looks_like_line_style_swatch(_mask_image(pattern))) is False


def test_code_assignment_pattern_matches_a_lettered_lookup_row():
    assert CODE_ASSIGNMENT_PATTERN.match('a = Schrägsitzventil 5/4"')
    assert CODE_ASSIGNMENT_PATTERN.match("b = Wasserzähler (Lieferung EWL)")
    assert not CODE_ASSIGNMENT_PATTERN.match("Wasserzähler")
    assert not CODE_ASSIGNMENT_PATTERN.match("Kaltwasser Netzdruck")


def _w003_page_and_candidates():
    pdf_bytes = FIXTURE.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    import pymupdf
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = pdf[0]
    candidates = detect_legend_candidates(doc.pages[0], rotation_matrix=page.rotation_matrix)
    return pdf, page, candidates


def test_w003_real_symbol_legend_produces_many_usable_icon_templates():
    pdf, page, candidates = _w003_page_and_candidates()
    try:
        legend = next(c for c in candidates if c.heading_text == "LEGENDE SANITÄR")
        entries, is_symbol_legend = classify_candidate_rows(page, legend)
    finally:
        pdf.close()

    assert is_symbol_legend is True
    usable = [e for e in entries if e.classification == "USABLE_TEMPLATE"]
    icons = [e for e in usable if not e.is_line_style_swatch]
    swatches = [e for e in usable if e.is_line_style_swatch]
    assert len(icons) >= 25, "the real legend should yield many real icon templates, not just a handful"
    assert len(swatches) >= 5, "the real legend's pipe-type line swatches must be detected"
    labels = {e.normalized_label for e in icons}
    assert "wasserzähler" in labels
    assert "ventil" in labels


def test_w003_coded_lookup_table_legend_is_rejected_as_a_whole():
    pdf, page, candidates = _w003_page_and_candidates()
    try:
        # "Legende Verteilbatterie": a real plan-labelled legend, but a
        # lettered lookup table (a = ..., b = ..., ...) with no symbol
        # column -- must never contribute usable templates.
        lookup_candidates = [
            c for c in candidates
            if any("schrägsitzventil" in r.text.lower() or "wasserzähler (lieferung" in r.text.lower() for r in (c.rows or []))
        ]
        assert lookup_candidates, "expected to find the Verteilbatterie block among the candidates"
        entries, is_symbol_legend = classify_candidate_rows(page, lookup_candidates[0])
    finally:
        pdf.close()

    assert is_symbol_legend is False
    assert all(e.classification == "REJECTED" for e in entries)


def test_w003_dsmmungslegende_text_block_is_rejected_as_non_symbol():
    pdf, page, candidates = _w003_page_and_candidates()
    try:
        insulation_candidates = [
            c for c in candidates
            if any("dämmst" in r.text.lower() or "dämmungen" in r.text.lower() for r in (c.rows or []))
        ]
        assert insulation_candidates, "expected to find the insulation-notes block among the candidates"
        entries, is_symbol_legend = classify_candidate_rows(page, insulation_candidates[0])
    finally:
        pdf.close()

    assert is_symbol_legend is False
