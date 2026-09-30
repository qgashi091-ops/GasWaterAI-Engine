"""Agent 3/12 -- TextAgent.

Sole responsibility: read relevant labels/notes/legend text and associate
them SPATIALLY (to a nearby symbol or edge) -- never to create or confirm a
component's identity by itself ("Text allein darf kein Bauteil erzeugen").
This is enforced structurally, not just by docstring: this agent's
`claim_types` is exactly `("label_association",)`, which does not include
"component_type" -- base_agent.py's `_observation()` raises if this agent
ever tried to emit one, so a text-only claim can never masquerade as a
component fact.

Router-level rule: only invoked for a text span whose existing deterministic
association (`app.plan_analysis.association.py`, already run upstream and
present as a `TextAssociation` on the page) is "unassigned" -- a text span
association.py already resolved to a symbol or edge is a DETERMINISTIC_FACT
this agent is never asked to redo.
"""
from __future__ import annotations

from app.microagents.hashing import hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan "
    "containing one piece of text and its immediate surroundings. Decide "
    "whether that text spatially belongs to a nearby drawn symbol, a nearby "
    "drawn pipe/edge, or neither. You are NOT identifying what the symbol "
    "is -- only where the text belongs. Respond only via the "
    "associate_text tool."
)

TOOL_NAME = "associate_text"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Decide what a piece of plan text is spatially associated with.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target_type": {"type": "string", "enum": ["symbol", "edge", "unassigned"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["target_type", "evidence", "confidence"],
    },
}


class TextAgent(BaseAgent):
    agent_id = "text_agent"
    claim_types = ("label_association",)

    def associate(
        self, context: PlanAgentContext, subject_id: str, page_number: int, text_bbox: tuple, text: str,
    ) -> AgentObservation:
        crop = context.render_crop_png(page_number, text_bbox, margin_fraction=1.5)
        prompt_hash = hash_text(SYSTEM_PROMPT)
        if crop is None:
            return self._observation(
                "label_association", subject_id, {"target_type": "unassigned", "target_id": None},
                confidence="uncertain", evidence=["could not render surrounding crop"], prompt_hash=prompt_hash,
            )
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_TOOL_SCHEMA,
            text=f"The text in question: {text!r}", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            return self._observation(
                "label_association", subject_id, {"target_type": "unassigned", "target_id": None},
                confidence=None, available=False, error=response.error, model=response.model,
                prompt_hash=prompt_hash,
            )
        parsed = response.tool_input or {}
        return self._observation(
            "label_association", subject_id,
            {"target_type": parsed.get("target_type", "unassigned"), "target_id": None},
            confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
            model=response.model, prompt_hash=prompt_hash, raw_response_id=response.raw_response_id,
            latency_ms=response.latency_ms, detail={"page": page_number, "text": text},
        )
