"""Schema-Validierung: AgentObservation enforces kind/agent_id at
construction time -- an invalid instance cannot exist, not just fail a
later check."""
from __future__ import annotations

import pytest

from app.multi_agent_v1.schema import AgentObservation, DeterministicFactRef, MergedClaim


def test_agent_observation_rejects_wrong_kind():
    with pytest.raises(ValueError, match="AGENT_OBSERVATION"):
        AgentObservation(
            kind="PLAN_FACT", agent_id="planstruktur_agent", claim_type="plan_area_kind",
            subject_id="p1", value="SCHEMA", confidence="supported",
        )


def test_agent_observation_rejects_unknown_agent_id():
    with pytest.raises(ValueError, match="Unknown agent_id"):
        AgentObservation(
            kind="AGENT_OBSERVATION", agent_id="chief_agent", claim_type="plan_area_kind",
            subject_id="p1", value="SCHEMA", confidence="supported",
        )


def test_agent_observation_accepts_valid_construction():
    obs = AgentObservation(
        kind="AGENT_OBSERVATION", agent_id="planstruktur_agent", claim_type="plan_area_kind",
        subject_id="p1", value="SCHEMA", confidence="supported",
    )
    d = obs.to_dict()
    assert d["kind"] == "AGENT_OBSERVATION"
    assert d["agent_id"] == "planstruktur_agent"


def test_deterministic_fact_ref_to_dict_has_stable_kind_label():
    ref = DeterministicFactRef(
        fact_id="PF1", claim_type="schlaufung_topology", subject_id="p1:node:n1",
        value="PHYSICAL_END", resolved=True, source_module="app.plan_analysis.plan_facts",
    )
    assert ref.to_dict()["kind"] == "DETERMINISTIC_FACT"


def test_merged_claim_to_dict_round_trips_conflicts():
    claim = MergedClaim("plan_area_kind", "p1", "SCHEMA", "AGENT_SINGLE", provenance=[{"kind": "AGENT_OBSERVATION", "source": "planstruktur_agent"}])
    d = claim.to_dict()
    assert d["resolution"] == "AGENT_SINGLE"
    assert d["conflicts"] == []
