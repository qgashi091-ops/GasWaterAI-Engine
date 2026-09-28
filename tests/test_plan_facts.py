"""Unit tests for the deterministic cycle/dead-end algorithm in
app/plan_analysis/plan_facts.py. Pure, synthetic graph fixtures shaped like
the parser's own schema.PageAnalysis output -- no PDF, no I/O.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.plan_analysis import schema
from app.plan_analysis.plan_facts import build_page_facts


def page(nodes, edges, symbols=None, text_spans=None, associations=None):
    return schema.PageAnalysis(
        page_number=1, width=1000, height=1000, rotation=0, text_source="native",
        text_spans=text_spans or [], graph=schema.PipeGraph(nodes=nodes, edges=edges),
        symbols=symbols or [], associations=associations or [],
        dangling_pipe_ends=[], warnings=[], stats={},
    )


def n(id_, point=(0.0, 0.0), degree=1):
    return schema.GraphNode(id=id_, point=point, degree=degree, is_endpoint=degree == 1)


def e(id_, source, target, is_bridge=False, bridge_reason=None, length=1.0):
    return schema.GraphEdge(id=id_, source=source, target=target, length=length,
                             color_rgb=None, stroke_width=None, polyline=[(0, 0), (1, 1)],
                             is_bridge=is_bridge, bridge_reason=bridge_reason)


def facts_of_type(facts, fact_type):
    return [f for f in facts if f.fact_type == fact_type]


def test_a_simple_chain_is_acyclic_with_confirmed_dead_ends():
    p = page(
        nodes=[n("n1"), n("n2", degree=2), n("n3", degree=2), n("n4", degree=2), n("n5")],
        edges=[e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n4"), e("e4", "n4", "n5")],
    )
    facts, summary = build_page_facts(p)
    assert summary["components"][0]["topology_status"] == "acyclic_tree"
    cycle_facts = facts_of_type(facts, "cycle_membership")
    assert len(cycle_facts) == 1 and cycle_facts[0].value is False
    dead_ends = facts_of_type(facts, "dead_end_path")
    assert len(dead_ends) == 2, "both real termini of the chain are reported"
    assert all(f.kind == "DERIVED_FACT" for f in dead_ends)
    assert all(f.value == "PHYSICAL_END" for f in dead_ends)


def test_b_closed_triangle_is_a_proven_cycle_with_no_dead_ends():
    p = page(
        nodes=[n("n1", degree=2), n("n2", degree=2), n("n3", degree=2)],
        edges=[e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n1")],
    )
    facts, summary = build_page_facts(p)
    assert summary["components"][0]["topology_status"] == "cycle_present"
    cycle_facts = facts_of_type(facts, "cycle_membership")
    assert len(cycle_facts) == 1 and cycle_facts[0].value is True
    assert facts_of_type(facts, "dead_end_path") == []


def test_c_branch_off_a_cycle_reports_both_the_cycle_and_the_dead_end_separately():
    p = page(
        nodes=[n("n1", degree=2), n("n2", degree=3), n("n3", degree=2), n("n4")],
        edges=[e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n1"), e("e4", "n2", "n4")],
    )
    facts, summary = build_page_facts(p)
    assert summary["components"][0]["topology_status"] == "cycle_present"
    branches = facts_of_type(facts, "branch")
    assert len(branches) == 1 and branches[0].supporting_nodes == ["n2"]
    dead_ends = facts_of_type(facts, "dead_end_path")
    assert len(dead_ends) == 1
    assert dead_ends[0].value == "PHYSICAL_END"
    assert dead_ends[0].supporting_edges == ["e4"]


def test_d_bridge_required_closure_never_becomes_a_cycle_fact():
    # X = {n1,n2} direct edge, Y = {n3,n4} direct edge; two bridges would
    # close a full ring IF trusted -- they must not be.
    p = page(
        nodes=[n("n1"), n("n2"), n("n3"), n("n4")],
        edges=[
            e("e1", "n1", "n2"),
            e("e2", "n3", "n4"),
            e("b1", "n2", "n3", is_bridge=True, bridge_reason="collinear_gap"),
            e("b2", "n4", "n1", is_bridge=True, bridge_reason="corner_convergence"),
        ],
    )
    facts, summary = build_page_facts(p)
    assert len(summary["components"]) == 2
    assert all(c["topology_status"] == "acyclic_tree" for c in summary["components"])
    bridge_facts = facts_of_type(facts, "bridge_continuity")
    assert len(bridge_facts) == 2
    assert all(f.kind == "UNRESOLVED" for f in bridge_facts)
    dead_ends = facts_of_type(facts, "dead_end_path")
    assert len(dead_ends) == 4, "all four nodes are degree-1 leaves in their 2-node components"
    assert all(f.kind == "UNRESOLVED" for f in dead_ends), "every leaf here is bridge-adjacent"


def test_e_unresolved_endpoint_mixed_with_a_proven_one():
    p = page(
        nodes=[n("n1"), n("n2"), n("n3")],
        edges=[e("e1", "n1", "n2"), e("b1", "n2", "n3", is_bridge=True)],
    )
    facts, _ = build_page_facts(p)
    dead_ends = {f.detail["terminal_node"]: f for f in facts_of_type(facts, "dead_end_path")}
    assert dead_ends["n1"].kind == "DERIVED_FACT"
    assert dead_ends["n1"].value == "PHYSICAL_END"
    assert dead_ends["n2"].kind == "UNRESOLVED"
    assert dead_ends["n2"].value is None


def test_f_input_order_independence():
    nodes_a = [n("n1"), n("n2", degree=3), n("n3", degree=2), n("n4")]
    edges_a = [e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n1"), e("e4", "n2", "n4")]
    facts_a, summary_a = build_page_facts(page(nodes_a, edges_a))

    nodes_b = list(reversed(nodes_a))
    edges_b = [edges_a[3], edges_a[1], edges_a[0], edges_a[2]]
    facts_b, summary_b = build_page_facts(page(nodes_b, edges_b))

    to_dicts = lambda fs: sorted((f.to_dict() for f in fs), key=lambda d: d["fact_id"])
    assert to_dicts(facts_a) == to_dicts(facts_b)
    assert summary_a == summary_b


def test_g_small_coordinate_jitter_does_not_change_topology_facts():
    edges = [e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n1")]
    p1 = page([n("n1", (0, 0), 2), n("n2", (10, 0), 2), n("n3", (10, 10), 2)], edges)
    p2 = page([n("n1", (0.3, -0.2), 2), n("n2", (10.1, 0.05), 2), n("n3", (9.8, 10.4), 2)], edges)
    facts1, summary1 = build_page_facts(p1)
    facts2, summary2 = build_page_facts(p2)
    strip = lambda fs: sorted(({k: v for k, v in f.to_dict().items() if k != "detail"} for f in fs), key=lambda d: d["fact_id"])
    assert strip(facts1) == strip(facts2)
    assert [c["topology_status"] for c in summary1["components"]] == [c["topology_status"] for c in summary2["components"]]


def test_stable_ids_are_content_derived_not_counters():
    p = page(
        nodes=[n("n1", degree=2), n("n2", degree=2), n("n3", degree=2)],
        edges=[e("e1", "n1", "n2"), e("e2", "n2", "n3"), e("e3", "n3", "n1")],
    )
    facts_a, _ = build_page_facts(p)
    facts_b, _ = build_page_facts(p)
    ids_a = sorted(f.fact_id for f in facts_a)
    ids_b = sorted(f.fact_id for f in facts_b)
    assert ids_a == ids_b
    assert all(f.fact_id.startswith("PF") for f in facts_a)


def test_a_self_loop_is_a_cycle_without_inflating_degree():
    p = page(
        nodes=[n("n1"), n("n2")],
        edges=[e("e1", "n1", "n2"), e("e2", "n1", "n1")],
    )
    facts, _ = build_page_facts(p)
    degree_facts = {f.supporting_nodes[0]: f for f in facts_of_type(facts, "node_degree")}
    assert degree_facts["n1"].value == 1
    assert degree_facts["n2"].value == 1


def test_kind_is_always_one_of_the_three_allowed_values():
    p = page(
        nodes=[n("n1"), n("n2", degree=2), n("n3")],
        edges=[e("e1", "n1", "n2"), e("b1", "n2", "n3", is_bridge=True)],
    )
    facts, _ = build_page_facts(p)
    assert all(f.kind in ("FACT", "DERIVED_FACT", "UNRESOLVED") for f in facts)


def test_dimension_evidence_is_a_fact_only_when_directly_associated_to_an_edge():
    span = schema.TextSpan(id="t1", text="DN25", bbox=(0, 0, 1, 1), source="native", label_hint="nominal_diameter")
    assoc = schema.TextAssociation(text_id="t1", target_type="edge", target_id="e1", distance=1.0)
    p = page(
        nodes=[n("n1"), n("n2")], edges=[e("e1", "n1", "n2")],
        text_spans=[span], associations=[assoc],
    )
    facts, _ = build_page_facts(p)
    dims = facts_of_type(facts, "dimension_evidence")
    assert len(dims) == 1
    assert dims[0].kind == "FACT"
    assert dims[0].value == "DN25"
