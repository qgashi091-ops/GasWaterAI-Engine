"""Multi-Agent Architecture v1 -- pipeline orchestration.

This module is deliberately thin and contains NO judgement of its own: it
(1) builds a PlanAgentContext from already-computed deterministic outputs,
(2) asks AgentRouter which agents to run and on which subjects,
(3) dispatches each routed agent's own narrow method per subject,
(4) collects every gap left behind and asks NachweisAgent to classify its
    source,
(5) merges everything through EvidenceMerger, and
(6) assembles CanonicalPlanUnderstanding.

There is no super-/chief-agent here: this function calls agents in a fixed,
documented order purely for DATA-DEPENDENCY reasons (e.g. RueckflussAgent
needs SicherungsAgent's already-produced value for the same subject), never
because one agent's judgement gates or overrides another's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.plan_analysis import schema as plan_schema

from .agents.anschluss_agent import AnschlussAgent
from .agents.leitungs_agent import LeitungsAgent
from .agents.nachweis_agent import NachweisAgent
from .agents.planstruktur_agent import PlanstrukturAgent
from .agents.probenahme_agent import ProbenahmeAgent
from .agents.rueckfluss_agent import RueckflussAgent
from .agents.schlaufungs_agent import SchlaufungsAgent
from .agents.sicherungs_agent import SicherungsAgent
from .agents.stagnations_agent import StagnationsAgent
from .agents.symbol_agent import SymbolAgent
from .agents.text_agent import TextAgent
from .agents.zirkulations_hydraulik_agent import ZirkulationsHydraulikAgent
from . import canonical
from .context import PlanAgentContext
from .evidence_merger import EvidenceMerger
from .provider import AgentModelProvider
from .router import AgentRouter
from .schema import AgentObservation, CallDiagnostics

ENGINE_VERSION = "multi-agent-v1.0.0"


@dataclass
class PipelineResult:
    canonical_plan_understanding: dict
    observations: list = field(default_factory=list)  # list[AgentObservation.to_dict()]

    def to_dict(self) -> dict:
        return {
            "engine_version": ENGINE_VERSION,
            "canonical_plan_understanding": self.canonical_plan_understanding,
            "observations": self.observations,
        }


def _diag(agent_id: str, subject_id: str, obs: AgentObservation) -> CallDiagnostics:
    called_model = obs.model is not None
    return CallDiagnostics(
        agent_id=agent_id, subject_id=subject_id, called_model=called_model,
        cached=obs.cached, available=obs.available, latency_ms=obs.latency_ms, error=obs.error,
    )


def run_multi_agent_v1(
    doc: plan_schema.DocumentAnalysis,
    provider: AgentModelProvider,
    pdf_bytes: bytes | None = None,
    plan_facts: dict | None = None,
    component_evidence: list[dict] | None = None,
    inventory: list[dict] | None = None,
    rule_checks: list[dict] | None = None,
    candidate_labels: list[str] | None = None,
) -> PipelineResult:
    context = PlanAgentContext(
        doc=doc, pdf_bytes=pdf_bytes, plan_facts=plan_facts or {}, component_evidence=component_evidence or [],
        inventory=inventory or [], rule_checks=rule_checks or [],
    )
    router = AgentRouter()
    routing_plan = router.route(context, candidate_labels=candidate_labels)

    observations: list[AgentObservation] = []
    call_diagnostics: list[CallDiagnostics] = []

    def _record(agent_id: str, subject_id: str, obs: AgentObservation) -> None:
        observations.append(obs)
        call_diagnostics.append(_diag(agent_id, subject_id, obs))

    # 1. PlanstrukturAgent
    if routing_plan.decisions["planstruktur_agent"].run:
        agent = PlanstrukturAgent(provider)
        for s in routing_plan.subjects["planstruktur_agent"]:
            obs = agent.classify_page(context, s["page"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 2. SymbolAgent
    if routing_plan.decisions["symbol_agent"].run:
        agent = SymbolAgent(provider)
        for s in routing_plan.subjects["symbol_agent"]:
            obs = agent.identify(context, s["subject_id"], s["page"], s["bbox"], s["candidate_labels"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 3. TextAgent
    if routing_plan.decisions["text_agent"].run:
        agent = TextAgent(provider)
        for s in routing_plan.subjects["text_agent"]:
            obs = agent.associate(context, s["subject_id"], s["page"], s["bbox"], s["text"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 4. LeitungsAgent
    if routing_plan.decisions["leitungs_agent"].run:
        agent = LeitungsAgent(provider)
        for s in routing_plan.subjects["leitungs_agent"]:
            obs = agent.classify_medium(context, s["subject_id"], s["page"], s["bbox"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 5. AnschlussAgent
    if routing_plan.decisions["anschluss_agent"].run:
        agent = AnschlussAgent(provider)
        for s in routing_plan.subjects["anschluss_agent"]:
            obs = agent.classify_connection(context, s["subject_id"], s["page"], s["bbox"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 6. SchlaufungsAgent (only ever called for plan_facts-UNRESOLVED subjects -- see router.py)
    if routing_plan.decisions["schlaufungs_agent"].run:
        agent = SchlaufungsAgent(provider)
        for s in routing_plan.subjects["schlaufungs_agent"]:
            obs = agent.classify(context, s["subject_id"], s["page"], s["bbox"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 7. SicherungsAgent -- must run before 9 (RueckflussAgent needs its value)
    sicherung_values: dict[str, dict] = {}
    if routing_plan.decisions["sicherungs_agent"].run:
        agent = SicherungsAgent(provider)
        for s in routing_plan.subjects["sicherungs_agent"]:
            obs = agent.assess(context, s["subject_id"], s["page"], s["bbox"], s["nearby_text"], s["component_type"])
            _record(agent.agent_id, s["subject_id"], obs)
            if obs.available:
                sicherung_values[s["subject_id"]] = obs.value

    # 8. StagnationsAgent (no model calls)
    if routing_plan.decisions["stagnations_agent"].run:
        agent = StagnationsAgent(provider)
        for s in routing_plan.subjects["stagnations_agent"]:
            obs = agent.assess(context, s["subject_id"], s["inventory_id"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 9. RueckflussAgent (no model calls; reads SicherungsAgent's value for the same subject)
    if routing_plan.decisions["rueckfluss_agent"].run:
        agent = RueckflussAgent(provider)
        for s in routing_plan.subjects["rueckfluss_agent"]:
            sicherung = sicherung_values.get(s["subject_id"], {"liquid_category": None, "category_source": "no_explicit_category_text", "device_type": None})
            obs = agent.assess(context, s["subject_id"], sicherung)
            _record(agent.agent_id, s["subject_id"], obs)

    # 10. ZirkulationsHydraulikAgent
    if routing_plan.decisions["zirkulations_hydraulik_agent"].run:
        agent = ZirkulationsHydraulikAgent(provider)
        for s in routing_plan.subjects["zirkulations_hydraulik_agent"]:
            obs = agent.classify(context, s["subject_id"], s["page"], s["bbox"])
            _record(agent.agent_id, s["subject_id"], obs)

    # 11. ProbenahmeAgent
    if routing_plan.decisions["probenahme_agent"].run:
        agent = ProbenahmeAgent(provider)
        for s in routing_plan.subjects["probenahme_agent"]:
            obs = agent.assess(context, s["subject_id"], s["page"], s["bbox"], s["nearby_text"])
            _record(agent.agent_id, s["subject_id"], obs)

    # ---- collect gaps for NachweisAgent (agent 12) ----
    gaps: list[tuple[str, str]] = []
    for obs in observations:
        if not obs.available and obs.error:
            gaps.append((f"{obs.subject_id}:{obs.claim_type}", obs.error))
        elif isinstance(obs.value, dict):
            reason = obs.value.get("reason") or obs.value.get("required_reason")
            if reason:
                gaps.append((f"{obs.subject_id}:{obs.claim_type}", reason))

    nachweis_decision, nachweis_subjects = router.route_nachweis(gaps)
    routing_plan.decisions["nachweis_agent"] = nachweis_decision
    routing_plan.subjects["nachweis_agent"] = nachweis_subjects
    if nachweis_decision.run:
        agent = NachweisAgent(provider)
        for s in nachweis_subjects:
            obs = agent.classify_gap(context, s["subject_id"], s["reason"])
            _record(agent.agent_id, s["subject_id"], obs)

    merger = EvidenceMerger()
    merged_claims = merger.merge(context, observations)
    plan_understanding = canonical.build(merged_claims, routing_plan, call_diagnostics)

    return PipelineResult(
        canonical_plan_understanding=plan_understanding,
        observations=[o.to_dict() for o in observations],
    )
