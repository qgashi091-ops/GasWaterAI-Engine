"""Agent 9/12 -- RueckflussAgent.

Sole responsibility: compare a DOCUMENTED application/category (from
SicherungsAgent's observation -- never re-derived here) against the
present/required safety device. Like StagnationsAgent, this is a pure
reconciliation agent with NO model call: "Bestehende Fachregeln verwenden.
Keine neuen SVGW-Anforderungen erfinden" -- this codebase has no existing
rule mapping a liquid category to a required backflow-device type (the same
confirmed gap SicherungsAgent's own docstring documents), so this agent
NEVER invents that mapping. Its only two honest outcomes are:
  - NOT_ASSESSABLE, "no_explicit_category_evidence" -- when SicherungsAgent
    itself could not read an explicit category off the plan;
  - NOT_ASSESSABLE, "no_existing_category_requirement_rule" -- when a
    category IS known but no existing Rule (app/rules/water_rules.py) states
    what device that category requires.
A COMPLIANT/NON_COMPLIANT verdict would require exactly the kind of
fabricated SVGW requirement this whole epic explicitly forbids, so this
agent structurally cannot produce one until a real Rule for it is added to
app/rules/ by someone with access to the actual corpus.
"""
from __future__ import annotations

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..schema import AgentObservation

RUECKFLUSS_VALUES = ("COMPLIANT", "NON_COMPLIANT", "NOT_ASSESSABLE")


class RueckflussAgent(BaseAgent):
    agent_id = "rueckfluss_agent"
    claim_types = ("rueckfluss_protection",)

    def assess(
        self, context: PlanAgentContext, subject_id: str, sicherungseinrichtung: dict,
    ) -> AgentObservation:
        category = sicherungseinrichtung.get("liquid_category")
        category_source = sicherungseinrichtung.get("category_source")
        device_type = sicherungseinrichtung.get("device_type")

        if category is None or category_source != "explicit_text_on_plan":
            return self._observation(
                "rueckfluss_protection", subject_id,
                {"result": "NOT_ASSESSABLE", "reason": "no_explicit_category_evidence", "device_type": device_type},
                confidence="supported",
                evidence=["SicherungsAgent found no explicit liquid-category text on the plan; never guessed."],
            )

        # A category IS documented, but this codebase has no existing rule
        # stating which device that category requires -- see module docstring.
        return self._observation(
            "rueckfluss_protection", subject_id,
            {
                "result": "NOT_ASSESSABLE", "reason": "no_existing_category_requirement_rule",
                "device_type": device_type, "documented_category": category,
            },
            confidence="supported",
            evidence=[
                f"category {category} is documented on the plan, but no existing rule in "
                "app/rules/water_rules.py defines the required device for it -- not fabricated here.",
            ],
        )
