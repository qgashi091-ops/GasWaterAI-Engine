"""Unit tests for component_facts.py: graph association (never invented from
proximity alone), the ambiguous-recognition-stays-unresolved rule, duplicate
suppression, and graph-order independence. Graph-association tests are pure
(no PDF rendering); the rest use the real, committed W-003 fixture (the only
plan this module is evaluated against -- W-001..W-010 remain untouched
holdout data).
"""
from __future__ import annotations

from pathlib import Path

from app.plan_analysis import schema
from app.plan_analysis.component_facts import _decide, _graph_association, build_component_facts
from app.plan_analysis.pipeline import analyze_pdf_bytes
from app.plan_analysis.symbol_library import get_library

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
LIB = get_library()


def _symbol(id_="sym0", bbox=(0, 0, 10, 10), centroid=(5, 5), ports=None):
    return schema.SymbolCandidate(
        id=id_, bbox=bbox, centroid=centroid, primitive_count=3, has_curves=True, area=100.0,
        port_node_ids=ports or [],
    )


def _edge(id_="e1", source="n1", target="n2", polyline=((0, 0), (100, 0))):
    return schema.GraphEdge(id=id_, source=source, target=target, length=100.0, color_rgb=None,
                             stroke_width=None, polyline=list(polyline), is_bridge=False)


def test_component_at_a_proven_endpoint_uses_the_existing_v01_port_evidence():
    sym = _symbol(ports=["n5"])
    assoc = _graph_association(sym, [], page_diagonal=1000)
    assert assoc["relation"] == "at_endpoint"
    assert assoc["node_ids"] == ["n5"]


def test_component_at_a_branch_with_multiple_proven_ports():
    sym = _symbol(ports=["n5", "n6"])
    assoc = _graph_association(sym, [], page_diagonal=1000)
    assert assoc["relation"] == "at_branch"


def test_component_on_a_pipe_is_found_by_proximity_to_the_drawn_polyline():
    sym = _symbol(centroid=(50, 1), ports=[])
    edge = _edge(polyline=((0, 0), (100, 0)))
    assoc = _graph_association(sym, [edge], page_diagonal=1000)
    assert assoc["relation"] == "on_edge"
    assert assoc["edge_ids"] == ["e1"]


def test_component_near_a_pipe_but_far_enough_away_is_never_invented_as_connected():
    sym = _symbol(centroid=(50, 500), ports=[])  # far from the edge below
    edge = _edge(polyline=((0, 0), (100, 0)))
    assoc = _graph_association(sym, [edge], page_diagonal=1000)
    assert assoc["relation"] == "near_not_connected"
    assert assoc["edge_ids"] == []


def test_ambiguous_recognition_stays_unresolved_never_a_confident_wrong_type():
    # A candidate that matches two members of a known ambiguous family
    # almost equally must never become a COMPONENT_FACT.
    import cv2
    import os
    from app.plan_analysis.symbol_library import LIBRARY_DIR, canonicalize
    gray = cv2.imread(os.path.join(LIBRARY_DIR, "templates", "SYM-008.png"), cv2.IMREAD_GRAYSCALE)
    canon = canonicalize(gray)
    scores = LIB.match(canon)
    sym = _symbol()
    fact = _decide(sym, page_number=1, scores=scores, library=LIB, graph_assoc={"relation": "on_edge", "node_ids": [], "edge_ids": ["e1"]})
    assert fact.kind != "COMPONENT_FACT"
    assert fact.corroboration_status in ("ambiguous_family", "text_or_legend_required", "below_fact_threshold")


def test_unrecognizable_crop_is_unresolved_not_a_forced_guess():
    sym = _symbol()
    fact = _decide(sym, page_number=1, scores=[], library=LIB, graph_assoc={"relation": "near_not_connected", "node_ids": [], "edge_ids": []})
    assert fact.kind == "UNRESOLVED"
    assert fact.symbol_type is None


def test_a_clean_self_match_of_a_non_ambiguous_symbol_reaches_component_fact():
    import cv2
    import os
    from app.plan_analysis.symbol_library import LIBRARY_DIR, canonicalize
    gray = cv2.imread(os.path.join(LIBRARY_DIR, "templates", "SYM-038.png"), cv2.IMREAD_GRAYSCALE)  # Wasserzaehler
    canon = canonicalize(gray)
    scores = LIB.match(canon)
    sym = _symbol()
    fact = _decide(sym, page_number=1, scores=scores, library=LIB, graph_assoc={"relation": "on_edge", "node_ids": [], "edge_ids": ["e1"]})
    assert fact.kind == "COMPONENT_FACT"
    assert fact.symbol_type == "SYM-038"


def test_w003_duplicate_symbol_entry_is_never_reported_twice():
    pdf_bytes = FIXTURE.read_bytes()
    doc = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    original_count = len(doc.pages[0].symbols)
    duplicated = doc.pages[0].symbols + [doc.pages[0].symbols[0]]
    doc.pages[0].symbols = duplicated
    result = build_component_facts(pdf_bytes, doc)
    assert result["stats"]["components_evaluated"] == original_count, "the duplicated entry must collapse into one component fact"


def test_w003_graph_order_independence():
    pdf_bytes = FIXTURE.read_bytes()
    doc_a = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    doc_b = analyze_pdf_bytes(pdf_bytes, filename=FIXTURE.name)
    doc_b.pages[0].symbols = list(reversed(doc_b.pages[0].symbols))
    doc_b.pages[0].graph.edges = list(reversed(doc_b.pages[0].graph.edges))

    result_a = build_component_facts(pdf_bytes, doc_a)
    result_b = build_component_facts(pdf_bytes, doc_b)

    key = lambda f: f["component_fact_id"]
    assert sorted(result_a["facts"], key=key) == sorted(result_b["facts"], key=key)
