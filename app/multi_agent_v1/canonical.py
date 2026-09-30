"""CanonicalPlanUnderstanding -- the multi-agent pipeline's single output
structure. Purely assembled from MergedClaims (evidence_merger.py) plus the
routing plan and call diagnostics -- this module computes nothing new
either, it only groups and serializes.
"""
from __future__ import annotations

from collections import defaultdict

from .diagnostics import summarize as summarize_diagnostics
from .router import RoutingPlan
from .schema import CallDiagnostics, MergedClaim

# claim_type -> the section name it is grouped under in the output. Every
# claim_type any of the 12 agents (or plan_facts/component_evidence/rules
# wrapped as DeterministicFactRef) can produce is listed here explicitly, so
# an unmapped claim_type is a loud bug (KeyError), never silently dropped.
_SECTION_BY_CLAIM_TYPE = {
    "plan_area_kind": "plan_areas",
    "component_type": "components",
    "label_association": "label_associations",
    "leitung_medium": "leitungen",
    "leitung_dimension_evidence": "leitungen",
    "leitung_segment": "leitungen",
    "anschluss_topology": "anschluesse",
    "schlaufung_topology": "schlaufung",
    "sicherungseinrichtung": "sicherung",
    "stagnation_risk": "stagnation",
    "rueckfluss_protection": "rueckfluss",
    "zirkulation_classification": "zirkulation",
    "probenahme_status": "probenahme",
    "nachweis_gap": "nachweis",
}


def build(
    merged_claims: list[MergedClaim], routing_plan: RoutingPlan, call_diagnostics: list[CallDiagnostics],
) -> dict:
    sections: dict = defaultdict(list)
    unmapped: list[dict] = []
    for claim in merged_claims:
        section = _SECTION_BY_CLAIM_TYPE.get(claim.claim_type)
        if section is None:
            if claim.claim_type.startswith("rule:"):
                section = "rule_checks"
            else:
                unmapped.append(claim.to_dict())
                continue
        sections[section].append(claim.to_dict())

    all_conflicts = [c.to_dict() for claim in merged_claims for c in claim.conflicts]

    return {
        "engine": "multi_agent_v1",
        "sections": dict(sections),
        "unmapped_claims": unmapped,
        "conflicts": all_conflicts,
        "routing": routing_plan.to_dict(),
        "diagnostics": summarize_diagnostics(routing_plan.decisions, call_diagnostics),
        "stats": {
            "claims_total": len(merged_claims),
            "by_resolution": {
                res: sum(1 for c in merged_claims if c.resolution == res)
                for res in ("DETERMINISTIC", "AGENT_AGREED", "AGENT_SINGLE", "AGENT_CONFLICT", "UNRESOLVED")
            },
            "conflicts_total": len(all_conflicts),
        },
    }
