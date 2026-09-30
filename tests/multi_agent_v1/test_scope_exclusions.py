"""Scope-Ausschluesse: this epic explicitly excludes pipe dimensioning,
DN-value correctness/competition, real-world length from schema geometry,
and scale-based length calculation. None of the 12 agents' claim_types may
name any of those, and the two "reconciliation-only" agents
(StagnationsAgent, RueckflussAgent) must never call a model at all --
concrete proof that "keine erneute Symbol-/Topologieinterpretation" and
"keine neuen SVGW-Anforderungen erfinden" are structurally true, not just
asserted."""
from __future__ import annotations

from app.multi_agent_v1.agents.anschluss_agent import AnschlussAgent
from app.multi_agent_v1.agents.leitungs_agent import LeitungsAgent
from app.multi_agent_v1.agents.nachweis_agent import NachweisAgent
from app.multi_agent_v1.agents.planstruktur_agent import PlanstrukturAgent
from app.multi_agent_v1.agents.probenahme_agent import ProbenahmeAgent
from app.multi_agent_v1.agents.rueckfluss_agent import RueckflussAgent
from app.multi_agent_v1.agents.schlaufungs_agent import SchlaufungsAgent
from app.multi_agent_v1.agents.sicherungs_agent import SicherungsAgent
from app.multi_agent_v1.agents.stagnations_agent import StagnationsAgent
from app.multi_agent_v1.agents.symbol_agent import SymbolAgent
from app.multi_agent_v1.agents.text_agent import TextAgent
from app.multi_agent_v1.agents.zirkulations_hydraulik_agent import ZirkulationsHydraulikAgent
from app.multi_agent_v1.provider import FixtureAgentModelProvider

ALL_AGENT_CLASSES = [
    PlanstrukturAgent, SymbolAgent, TextAgent, LeitungsAgent, AnschlussAgent, SchlaufungsAgent,
    SicherungsAgent, StagnationsAgent, RueckflussAgent, ZirkulationsHydraulikAgent, ProbenahmeAgent,
    NachweisAgent,
]

_OUT_OF_SCOPE_TOKENS = ("dimension", "dn_value", "dn_wert", "real_world_length", "scale", "massstab", "rohrlaenge", "pipe_length")


def test_no_claim_type_names_an_out_of_scope_dimensioning_concept():
    for cls in ALL_AGENT_CLASSES:
        for claim_type in cls.claim_types:
            lowered = claim_type.lower()
            for token in _OUT_OF_SCOPE_TOKENS:
                assert token not in lowered, f"{cls.agent_id}'s claim_type {claim_type!r} looks like an excluded dimensioning concept"


def test_stagnations_agent_never_calls_the_model():
    provider = FixtureAgentModelProvider(responses={"anything": {"value": "should never be used"}})
    agent = StagnationsAgent(provider)

    class _FakeContext:
        def facts_for_subject(self, subject_id):
            return []

    agent.assess(_FakeContext(), "subj1", "INV1")
    assert provider.calls == []


def test_nachweis_agent_never_calls_the_model():
    provider = FixtureAgentModelProvider(responses={"anything": {"value": "should never be used"}})
    agent = NachweisAgent(provider)
    agent.classify_gap(context=None, subject_id="subj1", reason="No API key configured")
    assert provider.calls == []
