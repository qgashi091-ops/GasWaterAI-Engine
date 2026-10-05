"""End-to-end pipeline integration: builds a real synthetic PDF, runs the
full multi-agent v1 pipeline against it with a FixtureAgentModelProvider
(no live credential needed or used), and checks the assembled
CanonicalPlanUnderstanding plus deterministic repeatability of the whole
merge/routing layer -- the one thing this epic explicitly asks every test
suite to prove."""
from __future__ import annotations

import json

from app.multi_agent_v1.pipeline import run_multi_agent_v1
from app.multi_agent_v1.provider import FixtureAgentModelProvider

from .conftest import analyze

_FIXTURE_RESPONSES = {
    "classify_plan_area": {"plan_area_kind": "SCHEMA", "evidence": [], "confidence": "uncertain"},
    "identify_symbol": {"component_type": "UNKNOWN", "evidence": [], "ambiguity": None, "confidence": "uncertain"},
    "associate_text": {"target_type": "unassigned", "evidence": [], "confidence": "uncertain"},
    "classify_medium": {"medium": "UNKNOWN", "evidence": [], "confidence": "uncertain"},
    "classify_connection": {"role": "UNKNOWN", "evidence": [], "confidence": "uncertain"},
    "classify_schlaufung": {"classification": "UNRESOLVED", "evidence": [], "confidence": "uncertain"},
    "classify_position": {"position": "UPSTREAM_OF_CONSUMER", "zuordnung": "test fixture", "evidence": [], "confidence": "supported"},
    "classify_zirkulation": {"classification": "CIRCULATION_SYSTEM", "evidence": [], "confidence": "uncertain"},
    "classify_probenahme": {"present": True, "position": "at distributor", "zuordnung": "test fixture", "evidence": [], "confidence": "supported"},
}


def _inventory():
    return [
        {
            "inventory_id": "INV-SICHERUNG", "component_type": "sicherheitsventil", "resolution": "COMPONENT_FACT",
            "page": 1, "bbox": [598, 990, 780, 1012], "graph_node_ids": ["n1"], "graph_edge_ids": [],
            "medium": None, "dimension": None, "topology_status": None, "cycle_or_dead_end": None,
            "reserve_evidence": False, "safety_device_evidence": True, "provenance": [],
        },
        {
            "inventory_id": "INV-ZIRKULATION", "component_type": "zirkulationspumpe", "resolution": "COMPONENT_FACT",
            "page": 1, "bbox": [598, 1090, 780, 1112], "graph_node_ids": ["n2"], "graph_edge_ids": [],
            "medium": "Zirkulation", "dimension": None, "topology_status": None, "cycle_or_dead_end": None,
            "reserve_evidence": False, "safety_device_evidence": False, "provenance": [],
        },
    ]


def _run(pdf_bytes):
    doc, facts = analyze(pdf_bytes)
    provider = FixtureAgentModelProvider(responses=dict(_FIXTURE_RESPONSES))
    result = run_multi_agent_v1(
        doc=doc, provider=provider, pdf_bytes=pdf_bytes, plan_facts=facts,
        inventory=_inventory(), rule_checks=[],
    )
    return result, provider


def test_pipeline_produces_a_complete_canonical_structure(legend_and_riser_pdf_bytes):
    result, _ = _run(legend_and_riser_pdf_bytes)
    cpu = result.canonical_plan_understanding
    assert cpu["engine"] == "multi_agent_v1"
    for key in ("sections", "conflicts", "routing", "diagnostics", "stats"):
        assert key in cpu


def test_plan_area_classification_present_for_the_only_page(legend_and_riser_pdf_bytes):
    result, _ = _run(legend_and_riser_pdf_bytes)
    plan_areas = result.canonical_plan_understanding["sections"].get("plan_areas", [])
    assert any(c["subject_id"] == "p1" for c in plan_areas)
    p1 = next(c for c in plan_areas if c["subject_id"] == "p1")
    # "Legende" heading text resolves this deterministically (keyword tier),
    # with zero model calls for this claim.
    assert p1["value"] == "LEGENDE"


def test_sicherung_category_is_read_from_plan_text_not_guessed(legend_and_riser_pdf_bytes):
    result, provider = _run(legend_and_riser_pdf_bytes)
    sicherung = result.canonical_plan_understanding["sections"].get("sicherung", [])
    claim = next(c for c in sicherung if c["subject_id"] == "INV-SICHERUNG")
    assert claim["value"]["liquid_category"] == 2
    assert claim["value"]["category_source"] == "explicit_text_on_plan"
    # position/zuordnung came from the fixture model call, category did not --
    # batching (performance-optimization epic) sends "classify_position_batch".
    assert any(r.tool_name == "classify_position_batch" for r in provider.calls)


def test_rueckfluss_is_not_assessable_never_fabricated(legend_and_riser_pdf_bytes):
    result, provider = _run(legend_and_riser_pdf_bytes)
    rueckfluss = result.canonical_plan_understanding["sections"].get("rueckfluss", [])
    claim = next(c for c in rueckfluss if c["subject_id"] == "INV-SICHERUNG")
    assert claim["value"]["result"] == "NOT_ASSESSABLE"
    assert not any(r.tool_name.startswith("classify_rueckfluss") for r in provider.calls)


def test_zirkulation_classified_from_text_with_zero_model_calls(legend_and_riser_pdf_bytes):
    result, provider = _run(legend_and_riser_pdf_bytes)
    zirkulation = result.canonical_plan_understanding["sections"].get("zirkulation", [])
    claim = next(c for c in zirkulation if c["subject_id"] == "INV-ZIRKULATION")
    assert claim["value"] == "CIRCULATION_SYSTEM"
    assert not any(r.tool_name == "classify_zirkulation" for r in provider.calls)


def test_probenahme_section_present(legend_and_riser_pdf_bytes):
    result, _ = _run(legend_and_riser_pdf_bytes)
    probenahme = result.canonical_plan_understanding["sections"].get("probenahme", [])
    assert len(probenahme) >= 1
    assert probenahme[0]["value"]["required"] == "NOT_ASSESSABLE"


def test_diagnostics_report_skipped_agents_with_reasons(legend_and_riser_pdf_bytes):
    result, _ = _run(legend_and_riser_pdf_bytes)
    diagnostics = result.canonical_plan_understanding["diagnostics"]
    assert diagnostics["totals"]["agents_skipped"] >= 1
    schlaufungs = diagnostics["by_agent"]["schlaufungs_agent"]
    assert schlaufungs["routed"] is False


def test_pipeline_is_deterministic_across_repeated_runs(legend_and_riser_pdf_bytes):
    result_a, _ = _run(legend_and_riser_pdf_bytes)
    result_b, _ = _run(legend_and_riser_pdf_bytes)
    a = json.dumps(result_a.canonical_plan_understanding, sort_keys=True)
    b = json.dumps(result_b.canonical_plan_understanding, sort_keys=True)
    assert a == b
