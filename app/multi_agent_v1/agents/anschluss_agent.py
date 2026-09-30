"""Agent 5/12 -- AnschlussAgent.

Sole responsibility: determine Leitung <-> Bauteil <-> Verbraucher <->
Verteiler relationships, and whether a component sits at a pipe's END.
Reuses the EXISTING graph_association a ComponentEvidence entry already
carries (built deterministically in component_evidence.py from the vector
graph -- see that module) as a DeterministicFactRef wherever it is present;
this agent is only invoked (see router.py) when that association is missing
or ambiguous, and even then never overrides it.
"""
from __future__ import annotations

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

ROLE_VALUES = ("VERBRAUCHER", "VERTEILER", "LEITUNGSENDE", "DURCHGANG", "UNKNOWN")

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan "
    "showing one component and its immediate pipe connections. Classify its "
    "connection role: VERBRAUCHER (a water-consuming fixture/appliance "
    "terminating the line), VERTEILER (a distributor/manifold feeding "
    "multiple further lines), LEITUNGSENDE (the component sits at the "
    "physical end of a pipe, none of the above), DURCHGANG (the pipe simply "
    "passes through/by it), or UNKNOWN. Base your answer only on what is "
    "visually present. Respond only via the classify_connection tool."
)

TOOL_NAME = "classify_connection"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify a component's pipe-connection role.",
    "input_schema": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": list(ROLE_VALUES)},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["role", "evidence", "confidence"],
    },
}


class AnschlussAgent(BaseAgent):
    agent_id = "anschluss_agent"
    claim_types = ("anschluss_topology",)

    def classify_connection(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple,
    ) -> AgentObservation:
        crop = context.render_crop_png(page_number, bbox, margin_fraction=1.0)
        prompt_hash = hash_text(SYSTEM_PROMPT)
        if crop is None:
            return self._observation(
                "anschluss_topology", subject_id, "UNKNOWN", confidence="uncertain",
                evidence=["could not render a crop"], prompt_hash=prompt_hash,
            )
        input_hash = hash_bytes(crop)
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_TOOL_SCHEMA,
            text="Classify the highlighted component's connection role.", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            return self._observation(
                "anschluss_topology", subject_id, "UNKNOWN", confidence=None, available=False,
                error=response.error, model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            )
        parsed = response.tool_input or {}
        return self._observation(
            "anschluss_topology", subject_id, parsed.get("role", "UNKNOWN"),
            confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
            model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            raw_response_id=response.raw_response_id, latency_ms=response.latency_ms,
        )
