"""Agent 12/12 -- NachweisAgent.

Sole responsibility: classify the SOURCE of a gap that some other part of
the pipeline already flagged -- never discover new gaps by re-inspecting
the plan itself (no crop, no model call, pure text-pattern reconciliation
over an already-produced error/reason string). The one rule the epic states
by name: ENGINE_ERROR, PARSER_ERROR and RECOGNITION_GAP must never be
reported as a customer deficiency (Kundenmangel).

Classification order (first match wins -- most specific technical causes
checked before the general "genuinely missing plan content" fallback):
  1. ENGINE_ERROR: the reason names an infrastructure failure this engine's
     own code raised (no credential, HTTP failure, unparseable model
     response, could-not-render).
  2. PARSER_ERROR: the reason traces to a reconstructed/bridged geometry
     inference (app/plan_analysis/bridging.py's own vocabulary) rather than
     directly-drawn evidence -- an artifact of this engine's own parsing,
     not the plan's content.
  3. RECOGNITION_GAP: the reason names a capability this engine/ruleset does
     not yet cover (e.g. "no existing rule", "not accessible from this
     repository" -- the same honestly-documented corpus gap SicherungsAgent/
     RueckflussAgent/ProbenahmeAgent already report).
  4. CUSTOMER_DEFICIENCY: anything else -- a genuine, actionable absence of
     required information on the plan itself.
"""
from __future__ import annotations

import re

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..schema import AgentObservation

GAP_TYPES = ("ENGINE_ERROR", "PARSER_ERROR", "RECOGNITION_GAP", "CUSTOMER_DEFICIENCY")

_ENGINE_ERROR_PATTERN = re.compile(
    r"no api key|request failed|could not parse response|could not render|no such page|"
    r"fixtureagentmodelprovider has no response", re.IGNORECASE,
)
_PARSER_ERROR_PATTERN = re.compile(r"reconstructed|bridge[_\s]|bridging", re.IGNORECASE)
_RECOGNITION_GAP_PATTERN = re.compile(
    r"no existing rule|rule corpus|not accessible from this repository|"
    r"no explicit .*category|no_explicit_category|no_existing_category_requirement_rule|"
    r"no_existing_requirement_rule", re.IGNORECASE,
)


def classify_gap_reason(reason: str) -> str:
    """Pure text classifier, exposed as a module-level function so
    router.py/pipeline.py and tests can classify a reason string without
    constructing an agent instance."""
    if _ENGINE_ERROR_PATTERN.search(reason):
        return "ENGINE_ERROR"
    if _PARSER_ERROR_PATTERN.search(reason):
        return "PARSER_ERROR"
    if _RECOGNITION_GAP_PATTERN.search(reason):
        return "RECOGNITION_GAP"
    return "CUSTOMER_DEFICIENCY"


class NachweisAgent(BaseAgent):
    agent_id = "nachweis_agent"
    claim_types = ("nachweis_gap",)

    def classify_gap(self, context: PlanAgentContext, subject_id: str, reason: str) -> AgentObservation:
        gap_type = classify_gap_reason(reason)
        return self._observation(
            "nachweis_gap", subject_id, {"gap_type": gap_type, "description": reason},
            confidence="supported", evidence=[reason], detail={"method": "text_pattern_reconciliation"},
        )
