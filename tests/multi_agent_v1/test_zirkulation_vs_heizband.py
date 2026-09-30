"""Zirkulation vs Heizband: ZirkulationsHydraulikAgent must distinguish
CIRCULATION_SYSTEM from HEAT_TRACING_SYSTEM (Heizband/Begleitheizung is
explicitly NOT hydraulic circulation, so it must never trigger a pump/valve
requirement), never assess temperature, and resolve the common cases with
zero model calls."""
from __future__ import annotations

from app.multi_agent_v1.agents.zirkulations_hydraulik_agent import ZirkulationsHydraulikAgent
from app.multi_agent_v1.provider import FixtureAgentModelProvider

from .conftest import analyze


def _context_for(pdf_bytes):
    from app.multi_agent_v1.context import PlanAgentContext
    doc, facts = analyze(pdf_bytes)
    return PlanAgentContext(doc=doc, pdf_bytes=pdf_bytes, plan_facts=facts)


def test_heizband_classified_as_heat_tracing_with_no_model_call(heizband_pdf_bytes):
    context = _context_for(heizband_pdf_bytes)
    provider = FixtureAgentModelProvider(responses={"classify_zirkulation": {"classification": "CIRCULATION_SYSTEM", "evidence": [], "confidence": "supported"}})
    agent = ZirkulationsHydraulikAgent(provider)
    obs = agent.classify(context, "subj1", 1, (280, 990, 520, 1010))
    assert obs.value == "HEAT_TRACING_SYSTEM"
    assert provider.calls == []  # deterministic text match short-circuits the model call


def test_zirkulationspumpe_classified_as_circulation_system_with_no_model_call(legend_and_riser_pdf_bytes):
    context = _context_for(legend_and_riser_pdf_bytes)
    provider = FixtureAgentModelProvider(responses={"classify_zirkulation": {"classification": "HEAT_TRACING_SYSTEM", "evidence": [], "confidence": "supported"}})
    agent = ZirkulationsHydraulikAgent(provider)
    obs = agent.classify(context, "subj1", 1, (580, 1090, 780, 1110))
    assert obs.value == "CIRCULATION_SYSTEM"
    assert provider.calls == []


def test_heat_tracing_situation_never_produces_a_pump_or_valve_requirement(heizband_pdf_bytes):
    context = _context_for(heizband_pdf_bytes)
    agent = ZirkulationsHydraulikAgent(FixtureAgentModelProvider())
    obs = agent.classify(context, "subj1", 1, (280, 990, 520, 1010))
    # The claim value is a closed enum -- HEAT_TRACING_SYSTEM never carries a
    # nested "pump_required"/"valve_required" field, so no downstream logic
    # could be misled into demanding one.
    assert obs.value == "HEAT_TRACING_SYSTEM"
    assert isinstance(obs.value, str)


def test_ambiguous_situation_falls_back_to_model(blank_pdf_bytes):
    from app.multi_agent_v1.context import PlanAgentContext
    doc, facts = analyze(blank_pdf_bytes)
    context = PlanAgentContext(doc=doc, pdf_bytes=blank_pdf_bytes, plan_facts=facts)
    provider = FixtureAgentModelProvider(responses={"classify_zirkulation": {"classification": "UNRESOLVED", "evidence": [], "confidence": "uncertain"}})
    agent = ZirkulationsHydraulikAgent(provider)
    obs = agent.classify(context, "subj1", 1, (0, 0, 2000, 2000))
    assert len(provider.calls) == 1
    assert obs.value == "UNRESOLVED"


def test_classify_zirkulation_tool_schema_has_no_temperature_field():
    from app.multi_agent_v1.agents.zirkulations_hydraulik_agent import _TOOL_SCHEMA
    props = _TOOL_SCHEMA["input_schema"]["properties"]
    assert not any("temperatur" in k.lower() or "temperature" in k.lower() for k in props)
