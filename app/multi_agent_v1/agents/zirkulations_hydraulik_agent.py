"""Agent 10/12 -- ZirkulationsHydraulikAgent.

Sole responsibility: classify a WW-circulation-relevant situation as
CIRCULATION_SYSTEM (classic hot-water circulation: pump, circulation
valves, hydraulic balancing) or HEAT_TRACING_SYSTEM (Heizband/
Begleitheizung -- explicitly NOT hydraulic circulation, so this agent must
never demand a pump or circulation valve for it), or UNRESOLVED. Explicitly
NEVER checks temperature -- "Keine Temperaturprüfung" -- no temperature
keyword or value is ever read or reasoned about here.

Deterministic-first, same discipline as LeitungsAgent: a small explicit
keyword check decides the common cases with zero model calls; the model is
only asked for genuinely ambiguous cases, and never asked about temperature.
"""
from __future__ import annotations

import re

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent
from ..context import PlanAgentContext
from ..provider import AgentModelRequest
from ..schema import AgentObservation

CLASSIFICATION_VALUES = ("CIRCULATION_SYSTEM", "HEAT_TRACING_SYSTEM", "UNRESOLVED")

_HEAT_TRACING_PATTERN = re.compile(r"heizband|begleitheizung|elektrische?\s+begleitheizung", re.IGNORECASE)
_CIRCULATION_PATTERN = re.compile(
    r"zirkulationspumpe|zirkulationsventil|r[uü]ckf[uü]hrung|hydraulisch(er|e)?\s+abgleich", re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan near a "
    "hot-water circulation situation. Decide whether what is drawn is a "
    "classic hydraulic circulation system (a circulation pump and/or "
    "circulation/balancing valves returning water to the source) or "
    "electric trace heating (Heizband/Begleitheizung -- a heating cable "
    "wrapped along the pipe, NOT hydraulic circulation), or UNRESOLVED if "
    "neither is clear. You are NEVER assessing temperature -- ignore any "
    "temperature values shown. Respond only via the classify_zirkulation tool."
)

TOOL_NAME = "classify_zirkulation"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify a WW-circulation-relevant drawing as hydraulic circulation or trace heating.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {"type": "string", "enum": list(CLASSIFICATION_VALUES)},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["classification", "evidence", "confidence"],
    },
}


class ZirkulationsHydraulikAgent(BaseAgent):
    agent_id = "zirkulations_hydraulik_agent"
    claim_types = ("zirkulation_classification",)

    def classify(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple,
    ) -> AgentObservation:
        nearby = context.nearby_text(page_number, bbox, margin_pt=60.0)
        joined = "\n".join(nearby)
        if _HEAT_TRACING_PATTERN.search(joined):
            return self._observation(
                "zirkulation_classification", subject_id, "HEAT_TRACING_SYSTEM", confidence="supported",
                evidence=["nearby text names Heizband/Begleitheizung"], detail={"method": "text_pattern"},
            )
        if _CIRCULATION_PATTERN.search(joined):
            return self._observation(
                "zirkulation_classification", subject_id, "CIRCULATION_SYSTEM", confidence="supported",
                evidence=["nearby text names a circulation pump/valve/hydraulic balancing"],
                detail={"method": "text_pattern"},
            )

        crop = context.render_crop_png(page_number, bbox, margin_fraction=0.9)
        prompt_hash = hash_text(SYSTEM_PROMPT)
        if crop is None:
            return self._observation(
                "zirkulation_classification", subject_id, "UNRESOLVED", confidence="uncertain",
                evidence=["no nearby text match and no crop could be rendered"], prompt_hash=prompt_hash,
            )
        input_hash = hash_bytes(crop)
        request = AgentModelRequest(
            system_prompt=SYSTEM_PROMPT, tool_name=TOOL_NAME, tool_schema=_TOOL_SCHEMA,
            text="Classify this circulation-relevant drawing (never assess temperature).", images=[crop],
        )
        response = self.call_model(request)
        if not response.available:
            return self._observation(
                "zirkulation_classification", subject_id, "UNRESOLVED", confidence=None, available=False,
                error=response.error, model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            )
        parsed = response.tool_input or {}
        return self._observation(
            "zirkulation_classification", subject_id, parsed.get("classification", "UNRESOLVED"),
            confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
            model=response.model, prompt_hash=prompt_hash, input_hash=input_hash,
            raw_response_id=response.raw_response_id, latency_ms=response.latency_ms, detail={"method": "model"},
        )
