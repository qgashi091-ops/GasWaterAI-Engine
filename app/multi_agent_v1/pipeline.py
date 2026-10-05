"""Multi-Agent Architecture v1 -- pipeline orchestration.

This module is deliberately thin and contains NO judgement of its own: it
(1) builds a PlanAgentContext from already-computed deterministic outputs,
(2) asks AgentRouter which agents to run and on which subjects,
(3) dispatches each routed agent's own batched method per subject,
(4) collects every gap left behind and asks NachweisAgent to classify its
    source,
(5) merges everything through EvidenceMerger, and
(6) assembles CanonicalPlanUnderstanding.

DEPENDENCY GRAPH (derived from the actual code, not assumed -- see the
performance-optimization epic's own baseline report in
docs/multi-agent-v1-performance-baseline.md for the full per-agent trace):
every agent except two reads ONLY already-computed deterministic data
(context.inventory/plan_facts/text_spans, all built BEFORE this function
is even called) -- none of them reads another AGENT's observation. The two
real, code-verified exceptions:
  - RueckflussAgent reads SicherungsAgent's own observation VALUE for the
    SAME subject (`sicherung_values`, populated while Stage A runs) --
    it cannot start before SicherungsAgent's result for that subject
    exists.
  - NachweisAgent reads the full `observations` list every OTHER agent
    produced (to find gaps) -- it cannot start before everything else is
    done.
This gives exactly three stages, run in this order:
  STAGE A -- planstruktur, symbol, text, leitungs, anschluss, schlaufungs,
             sicherungs, stagnations, zirkulations_hydraulik, probenahme:
             10 agents, provably independent of each other's output,
             dispatched onto a bounded ThreadPoolExecutor
             (GASWATERAI_AGENT_MAX_CONCURRENCY, default 3) so their
             (already-batched) provider calls overlap instead of
             running one agent fully before the next starts.
  STAGE B -- rueckfluss_agent (needs Stage A's sicherungs_agent output).
  STAGE C -- nachweis_agent (needs every Stage A/B observation).

Regardless of which Stage-A agent's thread finishes first, observations
are appended to the final list in the SAME fixed agent+subject order this
pipeline has always used (see `_STAGE_A_AGENT_ORDER` below) -- the merge
order, and therefore CanonicalPlanUnderstanding, is deterministic
independent of completion order.
"""
from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.plan_analysis import schema as plan_schema

from .agent_cache import AgentResponseCache
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
from . import timing_diagnostics
from .context import PlanAgentContext
from .evidence_merger import EvidenceMerger
from .provider import AgentModelProvider
from .router import AgentRouter
from .schema import AgentObservation, CallDiagnostics

ENGINE_VERSION = "multi-agent-v1.1.0"

# Fixed order Stage A's observations are appended in, REGARDLESS of which
# agent's thread actually finished first -- this is what keeps the merge
# (and therefore CanonicalPlanUnderstanding) deterministic under
# concurrency. Matches this pipeline's original, pre-parallelization order.
_STAGE_A_AGENT_ORDER = (
    "planstruktur_agent", "symbol_agent", "text_agent", "leitungs_agent", "anschluss_agent",
    "schlaufungs_agent", "sicherungs_agent", "stagnations_agent", "zirkulations_hydraulik_agent",
    "probenahme_agent",
)


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


def _stage_a_tasks(context: PlanAgentContext, routing_plan, provider, cache) -> dict:
    """Builds one zero-arg callable per Stage-A agent -- each returns
    `{subject_id: AgentObservation}` for exactly that agent, using its own
    batched method. Agents routed SKIP get a trivial callable that does no
    work at all (never constructs the agent, never touches the provider)."""
    decisions = routing_plan.decisions
    subjects = routing_plan.subjects
    tasks: dict[str, "callable"] = {}

    def _skip() -> dict:
        return {}

    # Every task below uses functools.partial, never a lambda closing over
    # a loop/if-chain-reused local -- `partial` binds `agent`/`items` by
    # VALUE at construction time, so reusing the name `agent` (or `items`)
    # across these independent if-blocks can never let one agent's lambda
    # accidentally capture a LATER block's agent instance (the classic
    # Python late-binding closure bug a naive `lambda: agent...` here would
    # hit, since every block would otherwise share the same enclosing-scope
    # variable).

    if decisions["planstruktur_agent"].run:
        agent = PlanstrukturAgent(provider, cache=cache)
        pages = [s["page"] for s in subjects["planstruktur_agent"]]
        tasks["planstruktur_agent"] = functools.partial(agent.classify_pages_batch, context, pages)
    else:
        tasks["planstruktur_agent"] = _skip

    if decisions["symbol_agent"].run:
        agent = SymbolAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"], s["candidate_labels"]) for s in subjects["symbol_agent"]]
        tasks["symbol_agent"] = functools.partial(agent.identify_batch, context, items)
    else:
        tasks["symbol_agent"] = _skip

    if decisions["text_agent"].run:
        agent = TextAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"], s["text"]) for s in subjects["text_agent"]]
        tasks["text_agent"] = functools.partial(agent.associate_batch, context, items)
    else:
        tasks["text_agent"] = _skip

    if decisions["leitungs_agent"].run:
        agent = LeitungsAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"]) for s in subjects["leitungs_agent"]]
        tasks["leitungs_agent"] = functools.partial(agent.classify_media_batch, context, items)
    else:
        tasks["leitungs_agent"] = _skip

    if decisions["anschluss_agent"].run:
        agent = AnschlussAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"]) for s in subjects["anschluss_agent"]]
        tasks["anschluss_agent"] = functools.partial(agent.classify_connections_batch, context, items)
    else:
        tasks["anschluss_agent"] = _skip

    if decisions["schlaufungs_agent"].run:
        agent = SchlaufungsAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"]) for s in subjects["schlaufungs_agent"]]
        tasks["schlaufungs_agent"] = functools.partial(agent.classify_batch, context, items)
    else:
        tasks["schlaufungs_agent"] = _skip

    if decisions["sicherungs_agent"].run:
        agent = SicherungsAgent(provider, cache=cache)
        items = [
            (s["subject_id"], s["page"], s["bbox"], s["nearby_text"], s["component_type"])
            for s in subjects["sicherungs_agent"]
        ]
        tasks["sicherungs_agent"] = functools.partial(agent.assess_batch, context, items)
    else:
        tasks["sicherungs_agent"] = _skip

    if decisions["stagnations_agent"].run:
        agent = StagnationsAgent(provider, cache=cache)
        items = list(subjects["stagnations_agent"])

        def _run_stagnations(agent=agent, items=items) -> dict:
            return {s["subject_id"]: agent.assess(context, s["subject_id"], s["inventory_id"]) for s in items}

        tasks["stagnations_agent"] = _run_stagnations
    else:
        tasks["stagnations_agent"] = _skip

    if decisions["zirkulations_hydraulik_agent"].run:
        agent = ZirkulationsHydraulikAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"]) for s in subjects["zirkulations_hydraulik_agent"]]
        tasks["zirkulations_hydraulik_agent"] = functools.partial(agent.classify_batch, context, items)
    else:
        tasks["zirkulations_hydraulik_agent"] = _skip

    if decisions["probenahme_agent"].run:
        agent = ProbenahmeAgent(provider, cache=cache)
        items = [(s["subject_id"], s["page"], s["bbox"], s["nearby_text"]) for s in subjects["probenahme_agent"]]
        tasks["probenahme_agent"] = functools.partial(agent.assess_batch, context, items)
    else:
        tasks["probenahme_agent"] = _skip

    return tasks


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
    with timing_diagnostics.phase("deterministic_preprocessing"):
        context = PlanAgentContext(
            doc=doc, pdf_bytes=pdf_bytes, plan_facts=plan_facts or {}, component_evidence=component_evidence or [],
            inventory=inventory or [], rule_checks=rule_checks or [],
        )
    router = AgentRouter()
    with timing_diagnostics.phase("router"):
        routing_plan = router.route(context, candidate_labels=candidate_labels)

    # Request-scoped only (see agent_cache.py's own docstring) -- never
    # persisted, never shared across plans or requests.
    cache = AgentResponseCache()

    observations: list[AgentObservation] = []
    call_diagnostics: list[CallDiagnostics] = []

    def _record(agent_id: str, subject_id: str, obs: AgentObservation) -> None:
        observations.append(obs)
        call_diagnostics.append(_diag(agent_id, subject_id, obs))

    # ---- STAGE A: 10 mutually-independent agents, bounded concurrency ----
    tasks = _stage_a_tasks(context, routing_plan, provider, cache)
    for agent_id in _STAGE_A_AGENT_ORDER:
        timing_diagnostics.begin_agent(
            agent_id, run=routing_plan.decisions[agent_id].run,
            subjects_total=len(routing_plan.subjects[agent_id]),
        )

    max_workers = timing_diagnostics.max_concurrency()
    stage_a_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            agent_id: executor.submit(timing_diagnostics.run_in_current_context(tasks[agent_id]))
            for agent_id in _STAGE_A_AGENT_ORDER
        }
        for agent_id, future in futures.items():
            stage_a_results[agent_id] = future.result()

    for agent_id in _STAGE_A_AGENT_ORDER:
        timing_diagnostics.end_agent(agent_id)

    sicherung_values: dict[str, dict] = {}
    for agent_id in _STAGE_A_AGENT_ORDER:
        results = stage_a_results[agent_id]
        for s in routing_plan.subjects[agent_id]:
            obs = results.get(s["subject_id"])
            if obs is None:
                continue
            _record(agent_id, s["subject_id"], obs)
            if agent_id == "sicherungs_agent" and obs.available:
                sicherung_values[s["subject_id"]] = obs.value

    # ---- STAGE B: rueckfluss_agent (needs Stage A's sicherungs_agent output) ----
    with timing_diagnostics.agent_scope("rueckfluss_agent", routing_plan.decisions["rueckfluss_agent"].run):
        if routing_plan.decisions["rueckfluss_agent"].run:
            agent = RueckflussAgent(provider, cache=cache)
            for s in routing_plan.subjects["rueckfluss_agent"]:
                sicherung = sicherung_values.get(
                    s["subject_id"],
                    {"liquid_category": None, "category_source": "no_explicit_category_text", "device_type": None},
                )
                obs = agent.assess(context, s["subject_id"], sicherung)
                _record(agent.agent_id, s["subject_id"], obs)

    # ---- collect gaps for NachweisAgent (Stage C) ----
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

    # ---- STAGE C: nachweis_agent (needs every other observation) ----
    with timing_diagnostics.agent_scope("nachweis_agent", nachweis_decision.run):
        if nachweis_decision.run:
            agent = NachweisAgent(provider, cache=cache)
            for s in nachweis_subjects:
                obs = agent.classify_gap(context, s["subject_id"], s["reason"])
                _record(agent.agent_id, s["subject_id"], obs)

    merger = EvidenceMerger()
    with timing_diagnostics.phase("evidence_merger"):
        merged_claims = merger.merge(context, observations)
    with timing_diagnostics.phase("canonical_output"):
        plan_understanding = canonical.build(merged_claims, routing_plan, call_diagnostics)

    return PipelineResult(
        canonical_plan_understanding=plan_understanding,
        observations=[o.to_dict() for o in observations],
    )
