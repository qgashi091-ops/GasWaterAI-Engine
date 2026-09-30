"""Agent 8/12 -- StagnationsAgent.

Sole responsibility: report stagnation/water-renewal risk STRICTLY from
already-confirmed facts and the existing rule engine's own result --
"Keine erneute Symbol-/Topologieinterpretation": this agent never renders a
crop, never calls a model, and never looks at a symbol or the vector graph
directly. It only reads:
  - the existing RULE-001-unintended-loop check result (app/rules/water_rules.py),
    already computed deterministically upstream, wrapped as a
    DeterministicFactRef with claim_type "rule:RULE-001-unintended-loop";
  - the existing RULE-002-dead-end-stagnation-length result, which this
    codebase already, honestly, always reports NOT_ASSESSABLE (no scale
    calibration -- see that rule's own docstring); this agent reports that
    same honesty rather than inventing a risk level from nothing.

This makes StagnationsAgent's `provider` parameter genuinely unused -- a
concrete, testable illustration of this epic's "Ziel: minimale KI-Aufrufe":
some agents in this architecture never need a model call at all.
"""
from __future__ import annotations

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..schema import AgentObservation

STAGNATION_VALUES = ("LOW", "ELEVATED", "UNRESOLVED")


class StagnationsAgent(BaseAgent):
    agent_id = "stagnations_agent"
    claim_types = ("stagnation_risk",)

    def assess(self, context: PlanAgentContext, subject_id: str, inventory_id: str) -> AgentObservation:
        loop_facts = [
            f for f in context.facts_for_subject(inventory_id)
            if f.claim_type == "rule:RULE-001-unintended-loop"
        ]
        length_facts = [
            f for f in context.facts_for_subject(inventory_id)
            if f.claim_type == "rule:RULE-002-dead-end-stagnation-length"
        ]

        if loop_facts:
            result = loop_facts[0].value
            if result == "NON_COMPLIANT":
                return self._observation(
                    "stagnation_risk", subject_id, "ELEVATED", confidence="supported",
                    evidence=["RULE-001-unintended-loop reported NON_COMPLIANT (closed loop, non-circulation-labeled)"],
                    detail={"basis": "RULE-001-unintended-loop", "rule_result": result},
                )
            if result == "COMPLIANT":
                return self._observation(
                    "stagnation_risk", subject_id, "LOW", confidence="supported",
                    evidence=["RULE-001-unintended-loop reported COMPLIANT (labeled circulation loop)"],
                    detail={"basis": "RULE-001-unintended-loop", "rule_result": result},
                )

        if length_facts:
            return self._observation(
                "stagnation_risk", subject_id, "UNRESOLVED", confidence="uncertain",
                evidence=["RULE-002-dead-end-stagnation-length is NOT_ASSESSABLE in this engine (no scale calibration)"],
                detail={"basis": "RULE-002-dead-end-stagnation-length"},
            )

        return self._observation(
            "stagnation_risk", subject_id, "UNRESOLVED", confidence=None,
            evidence=["no existing rule result covers this subject"], detail={"basis": "none"},
        )
