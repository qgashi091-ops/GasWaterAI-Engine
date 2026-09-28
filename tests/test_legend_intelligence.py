"""Unit + real-fixture tests for symbol_templates.py (Phase 3) and
legend_intelligence.py (Phases 4-6 + orchestration).

Covers the required test list: legend self-match exclusion, plan-specific
match beating a weaker generic match, explicit conflict recording,
nearby-but-unconnected graph objects, stable component_fact_ids, and
10-run reproducibility of the full v0.3 pipeline on the real W-003 fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from app.fingerprint import canonical_hash
from app.plan_analysis import schema
from app.plan_analysis.legend_entries import LegendEntry
from app.plan_analysis.legend_intelligence import (
    PlanMatchScore,
    _decide_hybrid,
    _symbol_in_any_legend,
    build_legend_intelligence,
)
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.symbol_library import get_library
from app.plan_analysis.symbol_templates import build_templates

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
LIB = get_library()


def _symbol(id_="sym0", bbox=(0.0, 0.0, 10.0, 10.0), centroid=(5.0, 5.0), ports=None):
    return schema.SymbolCandidate(id=id_, bbox=bbox, centroid=centroid, primitive_count=3,
                                   has_curves=True, area=100.0, port_node_ids=ports or [])


def test_a_symbol_bbox_inside_the_legend_is_excluded():
    page = pymupdf.open()
    p = page.new_page(width=1000, height=1000)
    legend_bboxes = [(0.0, 0.0, 200.0, 200.0)]
    assert _symbol_in_any_legend(p, (10.0, 10.0, 20.0, 20.0), legend_bboxes) is True
    assert _symbol_in_any_legend(p, (500.0, 500.0, 520.0, 520.0), legend_bboxes) is False
    page.close()


def _canon_blob(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    canvas = np.zeros((128, 128), dtype=np.uint8)
    canvas[40:88, 40:88] = 255
    canvas[40:88, 40:88][rng.rand(48, 48) < 0.1] = 0
    return canvas


def _entry(label, canon, line_swatch=False):
    return LegendEntry(
        legend_entry_id=f"LE_{label}", legend_id="LG1_0", page=1,
        bbox=(0.0, 0.0, 10.0, 10.0), symbol_bbox=(0.0, 0.0, 10.0, 10.0),
        label_text=label, normalized_label=label, label_source="native",
        classification="USABLE_TEMPLATE", ambiguity_reason=None,
        symbol_ink_present=True, is_line_style_swatch=line_swatch,
        symbol_canonical=canon,
    )


def test_an_unambiguous_plan_specific_match_becomes_a_component_fact_and_is_never_overridden():
    entries = [_entry("meineinrichtung", _canon_blob(1))]
    templates = build_templates(entries)
    top_score = PlanMatchScore(templates[0].template_id, templates[0].legend_entry_id, "meineinrichtung", 0.95, 0)
    plan_scores = [top_score]
    sym = _symbol()
    fact = _decide_hybrid(sym, page_number=1, plan_scores=plan_scores, generic_scores=[],
                           library=LIB, graph_assoc={"relation": "on_edge", "node_ids": [], "edge_ids": ["e1"]},
                           text_corroborated=False)
    assert fact.kind == "COMPONENT_FACT"
    assert fact.symbol_label == "meineinrichtung"
    assert fact.recognition_method == "plan_specific_legend_template_match"


def test_conflicting_generic_identity_is_recorded_but_never_overrides_a_clear_plan_specific_fact():
    entries = [_entry("meineinrichtung", _canon_blob(2))]
    templates = build_templates(entries)
    plan_scores = [PlanMatchScore(templates[0].template_id, templates[0].legend_entry_id, "meineinrichtung", 0.96, 0)]

    import cv2, os
    from app.plan_analysis.symbol_library import LIBRARY_DIR, canonicalize
    gray = cv2.imread(os.path.join(LIBRARY_DIR, "templates", "SYM-038.png"), cv2.IMREAD_GRAYSCALE)
    generic_scores = LIB.match(canonicalize(gray))  # a confident, different, real identity

    sym = _symbol()
    fact = _decide_hybrid(sym, page_number=1, plan_scores=plan_scores, generic_scores=generic_scores,
                           library=LIB, graph_assoc={"relation": "on_edge", "node_ids": [], "edge_ids": ["e1"]},
                           text_corroborated=False)
    assert fact.kind == "COMPONENT_FACT"
    assert fact.symbol_label == "meineinrichtung", "the unambiguous plan-specific match must win"
    assert fact.hybrid_conflict is True, "the disagreement with the generic library must be recorded, not hidden"


def test_weak_plan_specific_score_alone_never_forces_a_fact():
    entries = [_entry("weaklabel", _canon_blob(3))]
    templates = build_templates(entries)
    plan_scores = [PlanMatchScore(templates[0].template_id, templates[0].legend_entry_id, "weaklabel", 0.80, 0)]
    sym = _symbol()
    fact = _decide_hybrid(sym, page_number=1, plan_scores=plan_scores, generic_scores=[],
                           library=LIB, graph_assoc={"relation": "near_not_connected", "node_ids": [], "edge_ids": []},
                           text_corroborated=False)
    assert fact.kind == "COMPONENT_CANDIDATE"


def test_a_symbol_near_but_not_touching_any_edge_is_reported_unconnected_not_invented():
    sym = _symbol(centroid=(9999.0, 9999.0), ports=[])
    fact = _decide_hybrid(sym, page_number=1, plan_scores=[], generic_scores=[], library=LIB,
                           graph_assoc={"relation": "near_not_connected", "node_ids": [], "edge_ids": []},
                           text_corroborated=False)
    assert fact.kind == "UNRESOLVED"
    assert fact.graph_association["relation"] == "near_not_connected"


def test_component_fact_ids_are_stable_across_identical_repeated_calls():
    entries = [_entry("stableid", _canon_blob(4))]
    templates = build_templates(entries)
    plan_scores = [PlanMatchScore(templates[0].template_id, templates[0].legend_entry_id, "stableid", 0.95, 0)]
    sym = _symbol()
    graph_assoc = {"relation": "on_edge", "node_ids": [], "edge_ids": ["e1"]}
    fact_a = _decide_hybrid(sym, 1, plan_scores, [], LIB, graph_assoc, False)
    fact_b = _decide_hybrid(sym, 1, plan_scores, [], LIB, graph_assoc, False)
    assert fact_a.component_fact_id == fact_b.component_fact_id


def test_w003_full_pipeline_runs_and_excludes_legend_interior_symbols():
    pdf_bytes = FIXTURE.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    result = build_legend_intelligence(pdf_bytes, doc)

    assert result["stats"]["legend_candidates_detected"] >= 1
    assert result["stats"]["legend_entries_extracted"] >= 1
    assert result["stats"]["plan_specific_templates_built"] >= 10
    total_evaluated = result["stats"]["components_evaluated"]
    # v0.2 evaluated every v0.1 symbol candidate including ones sitting
    # inside the legend block itself; v0.3 must evaluate strictly fewer,
    # since legend-interior candidates are now excluded from Phase 4 search.
    from app.plan_analysis.component_facts import build_component_facts
    v02_result = build_component_facts(pdf_bytes, doc)
    assert total_evaluated < v02_result["stats"]["components_evaluated"]
    assert total_evaluated > 0


def test_w003_legend_intelligence_is_reproducible_across_ten_runs():
    pdf_bytes = FIXTURE.read_bytes()
    hashes = set()
    for _ in range(10):
        doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
        result = build_legend_intelligence(pdf_bytes, doc)
        hashes.add(canonical_hash(result))
    assert len(hashes) == 1, "legend intelligence output must be bit-identical across repeated runs of the same PDF"


def test_v01_topology_is_unaffected_by_running_v03_afterward():
    from app.plan_analysis.plan_facts import build_document_facts
    pdf_bytes = FIXTURE.read_bytes()
    doc_a = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    facts_before = build_document_facts(doc_a)

    doc_b = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    build_legend_intelligence(pdf_bytes, doc_b)  # runs v0.3, must not mutate shared state
    facts_after = build_document_facts(doc_b)

    assert canonical_hash(facts_before) == canonical_hash(facts_after)
