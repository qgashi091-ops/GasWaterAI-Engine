"""Agent 4/12 -- LeitungsAgent.

Sole responsibility: follow KW (Kaltwasser) / WW (Warmwasser) /
Zirkulation pipe-run identity for one edge. "Bestehender Vektorgraph/
PlanFacts hat Vorrang": this agent NEVER re-derives connectivity, direction,
or topology itself -- the drawn edge and its connectivity are already a
plan_facts.py `pipe_segment`/DERIVED_FACT; this agent only opines on the
one thing plan_facts.py does not classify at all: which MEDIUM (KW/WW/
Zirkulation) a given drawn edge carries.

Deterministic-first, same discipline as the rest of this package: a small,
explicit regex over the deterministic text already found near the edge
(no model call at all when it is unambiguous) before ever falling back to
a cropped-image model call.
"""
from __future__ import annotations

import re

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

MEDIUM_VALUES = ("KW", "WW", "Zirkulation", "UNKNOWN")

_PATTERNS = {
    "Zirkulation": re.compile(r"\bzirk(ulation)?\b|\bwwz\b", re.IGNORECASE),
    "WW": re.compile(r"\bww\b|warmwasser", re.IGNORECASE),
    "KW": re.compile(r"\bkw\b|kaltwasser", re.IGNORECASE),
}
# Zirkulation checked before WW: a "WWZ"-style label matches both WW and
# Zirkulation patterns, and Zirkulation is the more specific medium.

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan "
    "showing one drawn pipe segment and its nearby labels/colors. Decide "
    "whether it carries cold water (KW), hot water (WW), hot-water "
    "circulation (Zirkulation), or is not determinable (UNKNOWN) from what "
    "is visible. Do not guess connectivity or routing -- only the medium. "
    "Respond only via the classify_medium tool."
)

TOOL_NAME = "classify_medium"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify which medium a drawn pipe segment carries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "medium": {"type": "string", "enum": list(MEDIUM_VALUES)},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["medium", "evidence", "confidence"],
    },
}


class LeitungsAgent(BaseAgent):
    agent_id = "leitungs_agent"
    claim_types = ("leitung_medium",)

    def classify_medium(
        self, context: PlanAgentContext, subject_id: str, page_number: int, edge_bbox: tuple,
    ) -> AgentObservation:
        nearby = context.nearby_text(page_number, edge_bbox, margin_pt=40.0)
        joined = "\n".join(nearby)
        for medium, pattern in _PATTERNS.items():
            if pattern.search(joined):
                return self._observation(
                    "leitung_medium", subject_id, medium, confidence="supported",
                    evidence=[f"nearby text matches {medium!r} pattern"], detail={"method": "text_pattern", "nearby_text": nearby},
                )

        crop = context.render_crop_png(page_number, edge_bbox, margin_fraction=0.8)
        prompt_hash = hash_text(SYSTEM_PROMPT)
        if crop is None:
            return self._observation(
                "leitung_medium", subject_id, "UNKNOWN", confidence="uncertain",
                evidence=["no nearby text match and no crop could be rendered"], prompt_hash=prompt_hash,
            )
        input_hash = hash_bytes(crop)
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_TOOL_SCHEMA,
            text="Classify the medium of the highlighted pipe segment.", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            return self._observation(
                "leitung_medium", subject_id, "UNKNOWN", confidence=None, available=False,
                error=response.error, model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            )
        parsed = response.tool_input or {}
        return self._observation(
            "leitung_medium", subject_id, parsed.get("medium", "UNKNOWN"),
            confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
            model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            raw_response_id=response.raw_response_id, latency_ms=response.latency_ms,
            detail={"method": "model"},
        )
