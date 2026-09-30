"""Agent 2/12 -- SymbolAgent.

Sole responsibility: recognize and localize trinkwasserrelevante (potable-
water-relevant) components/symbols. Output is strictly: type, page, bbox,
evidence, a categorical safety/confidence marker, and ambiguity -- never a
compliance judgement, never a topology claim (that is LeitungsAgent's and
AnschlussAgent's job entirely).

Router-level rule (see router.py): this agent is only ever asked about a
symbol candidate whose existing `component_evidence` resolution is NOT
already "COMPONENT_FACT" -- a confidently resolved component identity is a
DETERMINISTIC_FACT this agent is never even invoked for, let alone allowed
to override (enforced again at merge time in evidence_merger.py; this
agent-level skip exists purely to save calls, per "Ziel: minimale
KI-Aufrufe").
"""
from __future__ import annotations

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

SYSTEM_PROMPT = (
    "You identify a single potable-water plumbing component shown in an "
    "image crop from a technical plan. Choose exactly one label from the "
    "given candidate list, or UNKNOWN if the crop does not clearly show one "
    "of them. Base your answer only on what is visually present. A cautious "
    "UNKNOWN is better than a confident wrong guess. Never assess whether "
    "the component is correctly installed or compliant -- only identify it. "
    "Respond only via the identify_symbol tool."
)

TOOL_NAME = "identify_symbol"


def _tool_schema(candidate_labels: list[str]) -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Identify the potable-water component in the image crop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_type": {"type": "string", "enum": [*candidate_labels, "UNKNOWN"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "ambiguity": {"type": ["string", "null"]},
                "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
            },
            "required": ["component_type", "evidence", "ambiguity", "confidence"],
        },
    }


class SymbolAgent(BaseAgent):
    agent_id = "symbol_agent"
    claim_types = ("component_type",)

    def identify(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple,
        candidate_labels: list[str],
    ) -> AgentObservation:
        crop = context.render_crop_png(page_number, bbox)
        if crop is None:
            return self._unavailable("component_type", subject_id, "Could not render a crop for this bbox.")

        prompt_hash = hash_text(SYSTEM_PROMPT + "|" + ",".join(sorted(candidate_labels)))
        input_hash = hash_bytes(crop)
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_tool_schema(sorted(candidate_labels)),
            text=f"Candidate labels: {', '.join(sorted(candidate_labels))}, or UNKNOWN.", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            return self._observation(
                "component_type", subject_id, None, confidence=None, available=False,
                error=response.error, model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            )
        parsed = response.tool_input or {}
        component_type = parsed.get("component_type")
        return self._observation(
            "component_type", subject_id,
            None if component_type in (None, "UNKNOWN") else component_type,
            confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
            ambiguity=parsed.get("ambiguity"), model=response.model, prompt_hash=prompt_hash,
            input_hash=input_hash, raw_response_id=response.raw_response_id, latency_ms=response.latency_ms,
            detail={"page": page_number, "bbox": list(bbox)},
        )
