"""Router: deterministic skip logic, matching this epic's own worked
examples verbatim (sicherer Dead-End/Cycle -> SchlaufungsAgent SKIP; keine
Zirkulationssituation -> Skip; keine Sicherungssituation -> Skip; keine
Probenahmesituation -> Skip)."""
from __future__ import annotations

from app.multi_agent_v1.context import PlanAgentContext
from app.multi_agent_v1.router import AgentRouter

from .conftest import analyze


def _context(pdf_bytes: bytes, inventory=None, rule_checks=None) -> PlanAgentContext:
    doc, facts = analyze(pdf_bytes)
    return PlanAgentContext(doc=doc, pdf_bytes=pdf_bytes, plan_facts=facts, inventory=inventory or [], rule_checks=rule_checks or [])


def test_schlaufungs_agent_skipped_when_all_dead_ends_resolved(legend_and_riser_pdf_bytes):
    context = _context(legend_and_riser_pdf_bytes)
    plan = AgentRouter().route(context)
    decision = plan.decisions["schlaufungs_agent"]
    # The synthetic riser is a simple drawn segment with no bridging gap --
    # plan_facts resolves both its ends directly, so nothing is left for
    # SchlaufungsAgent to do.
    assert decision.run is False
    assert "already resolved" in decision.reason


def test_zirkulation_agent_skipped_with_no_circulation_medium(legend_and_riser_pdf_bytes):
    context = _context(legend_and_riser_pdf_bytes, inventory=[])
    plan = AgentRouter().route(context)
    assert plan.decisions["zirkulations_hydraulik_agent"].run is False


def test_zirkulation_agent_routed_when_a_circulation_item_exists(legend_and_riser_pdf_bytes):
    inventory = [{
        "inventory_id": "INV1", "component_type": "zirkulationspumpe", "resolution": "COMPONENT_FACT",
        "page": 1, "bbox": [600, 1090, 700, 1110], "graph_node_ids": [], "graph_edge_ids": [],
        "medium": "Zirkulation", "dimension": None, "topology_status": None, "cycle_or_dead_end": None,
        "reserve_evidence": False, "safety_device_evidence": False, "provenance": [],
    }]
    context = _context(legend_and_riser_pdf_bytes, inventory=inventory)
    plan = AgentRouter().route(context)
    decision = plan.decisions["zirkulations_hydraulik_agent"]
    assert decision.run is True
    assert plan.subjects["zirkulations_hydraulik_agent"][0]["subject_id"] == "INV1"


def test_sicherungs_and_rueckfluss_skipped_with_no_safety_device(legend_and_riser_pdf_bytes):
    context = _context(legend_and_riser_pdf_bytes, inventory=[])
    plan = AgentRouter().route(context)
    assert plan.decisions["sicherungs_agent"].run is False
    assert plan.decisions["rueckfluss_agent"].run is False


def test_sicherungs_routed_when_safety_device_evidence_present(legend_and_riser_pdf_bytes):
    inventory = [{
        "inventory_id": "INV2", "component_type": "sicherheitsventil", "resolution": "COMPONENT_FACT",
        "page": 1, "bbox": [600, 990, 700, 1010], "graph_node_ids": [], "graph_edge_ids": [],
        "medium": None, "dimension": None, "topology_status": None, "cycle_or_dead_end": None,
        "reserve_evidence": False, "safety_device_evidence": True, "provenance": [],
    }]
    context = _context(legend_and_riser_pdf_bytes, inventory=inventory)
    plan = AgentRouter().route(context)
    assert plan.decisions["sicherungs_agent"].run is True
    assert plan.decisions["rueckfluss_agent"].run is True
    subj = plan.subjects["sicherungs_agent"][0]
    assert "Kategorie 2" in "\n".join(subj["nearby_text"])


def test_probenahme_agent_skipped_with_no_sampling_text(blank_pdf_bytes):
    context = _context(blank_pdf_bytes)
    plan = AgentRouter().route(context)
    assert plan.decisions["probenahme_agent"].run is False


def test_probenahme_agent_routed_when_sampling_text_present(legend_and_riser_pdf_bytes):
    context = _context(legend_and_riser_pdf_bytes)
    plan = AgentRouter().route(context)
    assert plan.decisions["probenahme_agent"].run is True


def test_symbol_agent_skipped_when_every_item_is_already_a_component_fact(legend_and_riser_pdf_bytes):
    inventory = [{
        "inventory_id": "INV3", "component_type": "wasserzaehler", "resolution": "COMPONENT_FACT",
        "page": 1, "bbox": [1, 1, 2, 2], "graph_node_ids": ["n1"], "graph_edge_ids": ["e1"],
        "medium": None, "dimension": None, "topology_status": None, "cycle_or_dead_end": None,
        "reserve_evidence": False, "safety_device_evidence": False, "provenance": [],
    }]
    context = _context(legend_and_riser_pdf_bytes, inventory=inventory)
    plan = AgentRouter().route(context)
    assert plan.decisions["symbol_agent"].run is False
    assert plan.decisions["anschluss_agent"].run is False


def test_route_nachweis_skips_when_no_gaps():
    decision, subjects = AgentRouter().route_nachweis([])
    assert decision.run is False
    assert subjects == []


def test_route_nachweis_runs_for_each_gap():
    decision, subjects = AgentRouter().route_nachweis([("s1:claim", "reason one"), ("s2:claim", "reason two")])
    assert decision.run is True
    assert len(subjects) == 2


def test_routing_is_deterministic_across_repeated_runs(legend_and_riser_pdf_bytes):
    context = _context(legend_and_riser_pdf_bytes)
    plan_a = AgentRouter().route(context)
    plan_b = AgentRouter().route(context)
    assert plan_a.to_dict() == plan_b.to_dict()
