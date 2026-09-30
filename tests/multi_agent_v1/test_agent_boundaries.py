"""Agentengrenzen: every agent can only ever emit its own declared
claim_types -- checked structurally (base_agent.py raises), not just
asserted about the code's intent. Also verifies the 12 agents' claim_types
are pairwise disjoint (the "one agent = one task" design assumption
canonical.py's section mapping and evidence_merger.py's per-(claim_type,
subject_id) grouping both rely on)."""
from __future__ import annotations

import pytest

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
from app.multi_agent_v1.schema import AGENT_IDS

ALL_AGENT_CLASSES = [
    PlanstrukturAgent, SymbolAgent, TextAgent, LeitungsAgent, AnschlussAgent, SchlaufungsAgent,
    SicherungsAgent, StagnationsAgent, RueckflussAgent, ZirkulationsHydraulikAgent, ProbenahmeAgent,
    NachweisAgent,
]


def test_every_agent_id_is_declared_and_unique():
    ids = [cls.agent_id for cls in ALL_AGENT_CLASSES]
    assert set(ids) == set(AGENT_IDS)
    assert len(ids) == len(set(ids))


def test_claim_types_are_pairwise_disjoint_across_agents():
    seen: dict[str, str] = {}
    for cls in ALL_AGENT_CLASSES:
        for ct in cls.claim_types:
            assert ct not in seen, f"claim_type {ct!r} declared by both {seen.get(ct)} and {cls.agent_id}"
            seen[ct] = cls.agent_id


@pytest.mark.parametrize("cls", ALL_AGENT_CLASSES)
def test_agent_cannot_emit_outside_its_declared_claim_types(cls):
    agent = cls(FixtureAgentModelProvider())
    with pytest.raises(ValueError, match="outside its declared claim_types"):
        agent._observation("definitely_not_a_real_claim_type", "subject1", value=None, confidence=None)


def test_symbol_agent_never_declares_topology_or_compliance_claims():
    # SymbolAgent is explicitly "keine Fachbewertung" -- it must not be able
    # to emit a compliance/topology-shaped claim under any name overlapping
    # the topology/compliance agents' own claim_types.
    forbidden = {"schlaufung_topology", "anschluss_topology", "leitung_medium", "stagnation_risk", "rueckfluss_protection"}
    assert forbidden.isdisjoint(SymbolAgent.claim_types)


def test_text_agent_cannot_create_a_component():
    # "Text allein darf kein Bauteil erzeugen" -- structurally, TextAgent's
    # claim_types must never include component_type.
    assert "component_type" not in TextAgent.claim_types
