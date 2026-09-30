"""AgentRouter -- a plain deterministic function module, not an agent and
not a chief/super-agent (it never calls a model and never produces a claim
of its own). Its only job is: for each of the 12 agents, decide RUN or SKIP,
and if RUN, exactly which subjects to ask about -- "Ziel: minimale
KI-Aufrufe". Every skip decision here is because the situation genuinely
does not apply or is already resolved by existing deterministic evidence,
matching this epic's own worked examples verbatim (sicherer Dead-End/Cycle
-> SchlaufungsAgent SKIP; keine Zirkulationssituation -> Skip; keine
Sicherungssituation -> Skip; keine Probenahmesituation -> Skip).

NachweisAgent is deliberately NOT routed here: its subjects (gaps found by
OTHER agents/facts) do not exist until after the rest of the pipeline has
run, so pipeline.py calls `route_nachweis()` separately, after collecting
every gap. This keeps the router honest about what it can actually decide
ahead of time, rather than special-casing a 13th "meta" decision path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.multi_agent_v1.agents.probenahme_agent import _PROBENAHME_PATTERN

from .context import PlanAgentContext
from .schema import RoutingDecision

DEFAULT_CANDIDATE_LABELS = ["kueche", "wc_up"]


def _edge_bbox(edge) -> tuple:
    xs = [p[0] for p in edge.polyline] or [0.0]
    ys = [p[1] for p in edge.polyline] or [0.0]
    return (min(xs), min(ys), max(xs), max(ys))


def _node_bbox(node, half: float = 30.0) -> tuple:
    x, y = node.point
    return (x - half, y - half, x + half, y + half)


@dataclass
class RoutingPlan:
    decisions: dict = field(default_factory=dict)   # agent_id -> RoutingDecision
    subjects: dict = field(default_factory=dict)     # agent_id -> list[dict] (per-subject descriptors)

    def to_dict(self) -> dict:
        return {
            "decisions": {k: v.to_dict() for k, v in self.decisions.items()},
            "call_budget_by_agent": {k: len(v) for k, v in self.subjects.items()},
        }


class AgentRouter:
    def route(self, context: PlanAgentContext, candidate_labels: list[str] | None = None) -> RoutingPlan:
        plan = RoutingPlan()
        self._route_planstruktur(context, plan)
        self._route_symbol(context, plan, candidate_labels or DEFAULT_CANDIDATE_LABELS)
        self._route_text(context, plan)
        self._route_leitungs(context, plan)
        self._route_anschluss(context, plan)
        self._route_schlaufungs(context, plan)
        self._route_sicherungs(context, plan)
        self._route_stagnations(context, plan)
        self._route_rueckfluss(context, plan)
        self._route_zirkulation(context, plan)
        self._route_probenahme(context, plan)
        return plan

    # ---- 1. PlanstrukturAgent: always relevant if the document has pages ----
    def _route_planstruktur(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        pages = context.all_pages()
        agent_id = "planstruktur_agent"
        if not pages:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "document has no pages")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, "every page needs a plan-area classification", [f"p{p.page_number}" for p in pages])
        plan.subjects[agent_id] = [{"subject_id": f"p{p.page_number}", "page": p.page_number} for p in pages]

    # ---- 2. SymbolAgent: only inventory items not already a COMPONENT_FACT ----
    def _route_symbol(self, context: PlanAgentContext, plan: RoutingPlan, candidate_labels: list[str]) -> None:
        agent_id = "symbol_agent"
        unresolved = [it for it in context.inventory if it.get("resolution") != "COMPONENT_FACT"]
        if not unresolved:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "every component identity is already a resolved COMPONENT_FACT")
            plan.subjects[agent_id] = []
            return
        labels = sorted({it["component_type"] for it in context.inventory if it.get("component_type")}) or list(candidate_labels)
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(unresolved)} component(s) lack a resolved identity", [it["inventory_id"] for it in unresolved])
        plan.subjects[agent_id] = [
            {"subject_id": it["inventory_id"], "page": it["page"], "bbox": tuple(it["bbox"]), "candidate_labels": labels}
            for it in unresolved
        ]

    # ---- 3. TextAgent: only text spans association.py left "unassigned" ----
    def _route_text(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "text_agent"
        subjects = []
        for page in context.all_pages():
            span_by_id = {s.id: s for s in page.text_spans}
            for assoc in page.associations:
                if assoc.target_type != "unassigned":
                    continue
                span = span_by_id.get(assoc.text_id)
                if span is None:
                    continue
                subjects.append({
                    "subject_id": f"p{page.page_number}:text:{span.id}", "page": page.page_number,
                    "bbox": tuple(span.bbox), "text": span.text,
                })
        if not subjects:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "no unassigned text spans -- association.py already resolved every one")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(subjects)} text span(s) left unassigned by deterministic association", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 4. LeitungsAgent: only edges that have some nearby dimension-label text to reconcile ----
    def _route_leitungs(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "leitungs_agent"
        refs = context.facts_for_claim_type("leitung_dimension_evidence")
        subjects = []
        for ref in refs:
            page_number = ref.detail.get("page")
            page = context.page(page_number) if page_number else None
            edge_ids = ref.detail.get("supporting_edges") or []
            if page is None or not edge_ids:
                continue
            edge = next((e for e in page.graph.edges if e.id == edge_ids[0]), None)
            if edge is None:
                continue
            subjects.append({"subject_id": ref.subject_id, "page": page_number, "bbox": _edge_bbox(edge)})
        if not subjects:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "no edges with nearby dimension-label text to reconcile a medium for")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(subjects)} edge(s) have text evidence worth classifying", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 5. AnschlussAgent: only components with no resolved graph association ----
    def _route_anschluss(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "anschluss_agent"
        unresolved = [
            it for it in context.inventory
            if not (it.get("graph_node_ids") or it.get("graph_edge_ids")) or it.get("resolution") != "COMPONENT_FACT"
        ]
        if not unresolved:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "every component already has a resolved graph association")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(unresolved)} component(s) lack a resolved connection role", [it["inventory_id"] for it in unresolved])
        plan.subjects[agent_id] = [
            {"subject_id": it["inventory_id"], "page": it["page"], "bbox": tuple(it["bbox"])} for it in unresolved
        ]

    # ---- 6. SchlaufungsAgent: ONLY subjects plan_facts left genuinely UNRESOLVED ----
    def _route_schlaufungs(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "schlaufungs_agent"
        refs = [r for r in context.facts_for_claim_type("schlaufung_topology") if not r.resolved]
        subjects = []
        for ref in refs:
            node_id = ref.subject_id.rsplit(":node:", 1)[-1]
            page_number = ref.detail.get("page")
            page = context.page(page_number) if page_number else None
            if page is None:
                continue
            node = next((n for n in page.graph.nodes if n.id == node_id), None)
            if node is None:
                continue
            subjects.append({"subject_id": ref.subject_id, "page": page_number, "bbox": _node_bbox(node)})
        if not subjects:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "every dead-end/cycle/terminal case is already resolved by plan_facts -- SKIP (sicherer Dead-End/Cycle)")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(subjects)} case(s) left UNRESOLVED by plan_facts", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 7. SicherungsAgent: only where safety_device_evidence is True ----
    def _route_sicherungs(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "sicherungs_agent"
        relevant = [it for it in context.inventory if it.get("safety_device_evidence")]
        if not relevant:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "no safety-device situation on this plan -- SKIP")
            plan.subjects[agent_id] = []
            return
        subjects = [
            {
                "subject_id": it["inventory_id"], "page": it["page"], "bbox": tuple(it["bbox"]),
                "nearby_text": context.nearby_text(it["page"], tuple(it["bbox"]), margin_pt=80.0),
                "component_type": it.get("component_type"),
            }
            for it in relevant
        ]
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(relevant)} safety-device situation(s) found", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 8. StagnationsAgent: only where a cycle or dead-end situation exists ----
    def _route_stagnations(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "stagnations_agent"
        relevant = [it for it in context.inventory if it.get("cycle_or_dead_end") in ("cycle", "dead_end")]
        if not relevant:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "no cycle/dead-end situation to assess stagnation risk for -- SKIP")
            plan.subjects[agent_id] = []
            return
        subjects = [{"subject_id": it["inventory_id"], "inventory_id": it["inventory_id"]} for it in relevant]
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(relevant)} cycle/dead-end situation(s) found", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 9. RueckflussAgent: only where a safety-device situation exists (mirrors SicherungsAgent) ----
    def _route_rueckfluss(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "rueckfluss_agent"
        relevant = [it for it in context.inventory if it.get("safety_device_evidence")]
        if not relevant:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "no safety-device situation on this plan -- SKIP")
            plan.subjects[agent_id] = []
            return
        subjects = [{"subject_id": it["inventory_id"]} for it in relevant]
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(relevant)} safety-device situation(s) to check backflow protection for", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 10. ZirkulationsHydraulikAgent: only where medium == "Zirkulation" ----
    def _route_zirkulation(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "zirkulations_hydraulik_agent"
        relevant = [it for it in context.inventory if it.get("medium") == "Zirkulation"]
        if not relevant:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "keine Zirkulationssituation -- SKIP")
            plan.subjects[agent_id] = []
            return
        subjects = [{"subject_id": it["inventory_id"], "page": it["page"], "bbox": tuple(it["bbox"])} for it in relevant]
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(relevant)} circulation-labeled situation(s) found", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 11. ProbenahmeAgent: only where sampling-point text exists on a page ----
    def _route_probenahme(self, context: PlanAgentContext, plan: RoutingPlan) -> None:
        agent_id = "probenahme_agent"
        subjects = []
        for page in context.all_pages():
            for span in page.text_spans:
                if _PROBENAHME_PATTERN.search(span.text):
                    subjects.append({
                        "subject_id": f"p{page.page_number}:text:{span.id}", "page": page.page_number,
                        "bbox": tuple(span.bbox),
                        "nearby_text": context.nearby_text(page.page_number, tuple(span.bbox), margin_pt=80.0),
                    })
        if not subjects:
            plan.decisions[agent_id] = RoutingDecision(agent_id, False, "keine Probenahmesituation -- SKIP")
            plan.subjects[agent_id] = []
            return
        plan.decisions[agent_id] = RoutingDecision(agent_id, True, f"{len(subjects)} sampling-point mention(s) found", [s["subject_id"] for s in subjects])
        plan.subjects[agent_id] = subjects

    # ---- 12. NachweisAgent: routed separately, after the rest of the pipeline ran ----
    def route_nachweis(self, gaps: list[tuple[str, str]]) -> tuple[RoutingDecision, list[dict]]:
        agent_id = "nachweis_agent"
        if not gaps:
            return RoutingDecision(agent_id, False, "no unresolved gap was left by any other agent/fact"), []
        subjects = [{"subject_id": subject_id, "reason": reason} for subject_id, reason in gaps]
        return (
            RoutingDecision(agent_id, True, f"{len(gaps)} unresolved gap(s) need source classification", [s["subject_id"] for s in subjects]),
            subjects,
        )
