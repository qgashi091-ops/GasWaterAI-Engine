"""Agent 11/12 -- ProbenahmeAgent.

Sole responsibility: sampling points (Probenahmestellen) -- whether
required (only so far as an EXISTING rule states it), whether present, and
their position/assignment. "erforderlich soweit bestehende Regel dies
vorgibt": this codebase has no existing rule (app/rules/water_rules.py)
defining when a sampling point is required, so `required` is always
honestly "NOT_ASSESSABLE" with that reason -- never a fabricated SVGW
requirement, mirroring RueckflussAgent's identical honesty about the
missing category-requirement rule.

`present`/`position`/`zuordnung` ARE determinable from what is actually
drawn/labeled, so those use the same deterministic-text-first, model-
fallback-second pattern as the other perception agents.
"""
from __future__ import annotations

import re

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

_PROBENAHME_PATTERN = re.compile(r"probenahmestelle|probenahmehahn|entnahmestelle", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan near "
    "text that may reference a water sampling point (Probenahmestelle). "
    "Confirm whether a sampling point is actually drawn/present, and if so "
    "describe its position and what it is assigned to. Do not assess "
    "whether one is legally required. Respond only via the "
    "classify_probenahme tool."
)

TOOL_NAME = "classify_probenahme"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Confirm presence and position of a water sampling point.",
    "input_schema": {
        "type": "object",
        "properties": {
            "present": {"type": "boolean"},
            "position": {"type": ["string", "null"]},
            "zuordnung": {"type": ["string", "null"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["present", "position", "zuordnung", "evidence", "confidence"],
    },
}


class ProbenahmeAgent(BaseAgent):
    agent_id = "probenahme_agent"
    claim_types = ("probenahme_status",)

    def assess(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple, nearby_text: list[str],
    ) -> AgentObservation:
        base_value = {"required": "NOT_ASSESSABLE", "required_reason": "no_existing_requirement_rule"}
        joined = "\n".join(nearby_text)
        text_hit = bool(_PROBENAHME_PATTERN.search(joined))

        crop = context.render_crop_png(page_number, bbox, margin_fraction=1.0)
        prompt_hash = hash_text(SYSTEM_PROMPT)
        if crop is None:
            value = {**base_value, "present": text_hit, "position": None, "zuordnung": None}
            return self._observation(
                "probenahme_status", subject_id, value, confidence="uncertain" if text_hit else None,
                evidence=(["nearby text names a sampling point"] if text_hit else []), prompt_hash=prompt_hash,
            )
        input_hash = hash_bytes(crop)
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_TOOL_SCHEMA,
            text="Confirm the sampling point shown, if any.", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            value = {**base_value, "present": text_hit, "position": None, "zuordnung": None}
            return self._observation(
                "probenahme_status", subject_id, value, confidence=None, available=False,
                error=response.error, model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            )
        parsed = response.tool_input or {}
        value = {
            **base_value, "present": bool(parsed.get("present", text_hit)),
            "position": parsed.get("position"), "zuordnung": parsed.get("zuordnung"),
        }
        return self._observation(
            "probenahme_status", subject_id, value, confidence=parsed.get("confidence"),
            evidence=list(parsed.get("evidence") or []), model=response.model, prompt_hash=prompt_hash,
            input_hash=input_hash, raw_response_id=response.raw_response_id, latency_ms=response.latency_ms,
        )
