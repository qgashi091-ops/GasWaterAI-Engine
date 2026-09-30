"""EvidenceMerger: PlanFacts-Prioritaet (deterministic always wins),
Konflikte (never silently hidden), Evidence Merge (agreement/single/
conflict/unresolved), and deterministic repeatability of the merge layer."""
from __future__ import annotations

from app.multi_agent_v1.context import PlanAgentContext
from app.multi_agent_v1.evidence_merger import EvidenceMerger
from app.multi_agent_v1.schema import AgentObservation, DeterministicFactRef, MergedClaim

from .conftest import analyze


def _blank_context(pdf_bytes) -> PlanAgentContext:
    doc, facts = analyze(pdf_bytes)
    return PlanAgentContext(doc=doc, pdf_bytes=pdf_bytes, plan_facts=facts)


def _obs(agent_id, claim_type, subject_id, value, confidence="supported", available=True, error=None):
    return AgentObservation(
        kind="AGENT_OBSERVATION", agent_id=agent_id, claim_type=claim_type, subject_id=subject_id,
        value=value, confidence=confidence, available=available, error=error,
    )


def test_deterministic_fact_wins_over_disagreeing_agent_observation(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    context._facts_by_subject["p1:node:n1"] = [DeterministicFactRef(
        fact_id="PF1", claim_type="schlaufung_topology", subject_id="p1:node:n1",
        value="PHYSICAL_END", resolved=True, source_module="app.plan_analysis.plan_facts",
    )]
    context._facts_by_claim_type["schlaufung_topology"] = context._facts_by_subject["p1:node:n1"]

    observations = [_obs("schlaufungs_agent", "schlaufung_topology", "p1:node:n1", "SCHLAUFE")]
    claims = EvidenceMerger().merge(context, observations)
    claim = next(c for c in claims if c.subject_id == "p1:node:n1")
    assert claim.resolution == "DETERMINISTIC"
    assert claim.value == "PHYSICAL_END"  # the agent's disagreeing "SCHLAUFE" never wins
    assert len(claim.conflicts) == 1
    assert claim.conflicts[0].competing_values[0]["value"] == "PHYSICAL_END"


def test_single_available_observation_resolves_agent_single(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    claims = EvidenceMerger().merge(context, [_obs("planstruktur_agent", "plan_area_kind", "p1", "SCHEMA")])
    claim = claims[0]
    assert claim.resolution == "AGENT_SINGLE"
    assert claim.value == "SCHEMA"


def test_agreeing_observations_resolve_agent_agreed(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    observations = [
        _obs("symbol_agent", "component_type", "INV1", "wasserzaehler"),
        _obs("symbol_agent", "component_type", "INV1", "wasserzaehler"),
    ]
    claim = EvidenceMerger().merge(context, observations)[0]
    assert claim.resolution == "AGENT_AGREED"
    assert claim.value == "wasserzaehler"


def test_disagreeing_observations_produce_conflict_never_silently_resolved(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    observations = [
        _obs("symbol_agent", "component_type", "INV1", "wasserzaehler", confidence="uncertain"),
        _obs("symbol_agent", "component_type", "INV1", "absperrarmatur", confidence="uncertain"),
    ]
    claim = EvidenceMerger().merge(context, observations)[0]
    assert claim.resolution == "AGENT_CONFLICT"
    assert len(claim.conflicts) == 1
    values = {c["value"] for c in claim.conflicts[0].competing_values}
    assert values == {"wasserzaehler", "absperrarmatur"}


def test_unavailable_observation_yields_unresolved(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    observations = [_obs("symbol_agent", "component_type", "INV1", None, confidence=None, available=False, error="No API key configured")]
    claim = EvidenceMerger().merge(context, observations)[0]
    assert claim.resolution == "UNRESOLVED"
    assert claim.value is None


def test_merge_is_deterministic_across_repeated_runs(blank_pdf_bytes):
    context = _blank_context(blank_pdf_bytes)
    observations = [
        _obs("symbol_agent", "component_type", "INV1", "wasserzaehler", confidence="uncertain"),
        _obs("symbol_agent", "component_type", "INV1", "absperrarmatur", confidence="uncertain"),
        _obs("planstruktur_agent", "plan_area_kind", "p1", "SCHEMA"),
    ]
    merger = EvidenceMerger()
    claims_a = [c.to_dict() for c in merger.merge(context, list(observations))]
    claims_b = [c.to_dict() for c in merger.merge(context, list(reversed(observations)))]
    assert claims_a == claims_b
