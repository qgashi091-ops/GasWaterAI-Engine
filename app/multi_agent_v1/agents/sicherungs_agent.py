"""Agent 7/12 -- SicherungsAgent.

Sole responsibility: for an ALREADY-IDENTIFIED safety-relevant component
(component_type resolved elsewhere -- this agent never re-identifies the
symbol itself, that is SymbolAgent's exclusive claim), determine its
POSITION and ZUORDNUNG (what it protects), and the liquid category (1-5)
IF AND ONLY IF the plan itself states it explicitly.

"Ausschließlich die in den bestehenden Richtlinien definierten
Flüssigkeitskategorien 1-5. Niemals Kategorien erfinden oder raten": this
codebase has NO existing SVGW category-definition corpus (confirmed by
repo-wide search -- same finding already documented in
app/rules/__init__.py and app/rules/water_rules.py's own docstrings for the
COMPLIANT/NON_COMPLIANT rule set). So this agent NEVER calls a model to
guess a category from a symbol's appearance -- it can only read an EXPLICIT
category mention already written on the plan (e.g. "Kategorie 3",
"Flüssigkeitskategorie 2", "FLK 4") via a small deterministic regex, with NO
model call involved in that decision at all. When no such explicit text
exists, `liquid_category` is honestly None with `category_source` explaining
why -- never a fabricated best guess.
"""
from __future__ import annotations

import re

from app.microagents.hashing import hash_bytes, hash_text

from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Position/zuordnung is a slightly richer per-subject answer than a bare
# enum -- conservative batch size.
DEFAULT_BATCH_SIZE = 5

POSITION_VALUES = ("UPSTREAM_OF_CONSUMER", "AT_DISTRIBUTOR", "AT_CONNECTION_POINT", "UNKNOWN")

_CATEGORY_PATTERN = re.compile(
    r"(?:fl[uü]ssigkeitskategorie|flk|kategorie)\s*[:\-]?\s*([1-5])\b", re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan "
    "showing one already-identified safety/backflow-protection device and "
    "its immediate pipe connections. Classify ONLY its position relative to "
    "what it protects: UPSTREAM_OF_CONSUMER, AT_DISTRIBUTOR, "
    "AT_CONNECTION_POINT, or UNKNOWN. Do not classify what TYPE of device "
    "it is (already known) and never invent a liquid category. Respond only "
    "via the classify_position tool."
)

TOOL_NAME = "classify_position"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify a safety device's position relative to what it protects.",
    "input_schema": {
        "type": "object",
        "properties": {
            "position": {"type": "string", "enum": list(POSITION_VALUES)},
            "zuordnung": {"type": ["string", "null"], "description": "Short description of what it protects, if visible."},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["position", "zuordnung", "evidence", "confidence"],
    },
}


def extract_explicit_category(text_blocks: list[str]) -> tuple[int | None, str]:
    """Deterministic, model-free category extraction. Returns
    (category_or_None, category_source). Never called with intent to guess --
    a miss returns (None, 'no_explicit_category_text'), never a fallback
    heuristic that could be mistaken for professional judgement."""
    for text in text_blocks:
        m = _CATEGORY_PATTERN.search(text)
        if m:
            return int(m.group(1)), "explicit_text_on_plan"
    return None, "no_explicit_category_text"


class SicherungsAgent(BaseAgent):
    agent_id = "sicherungs_agent"
    claim_types = ("sicherungseinrichtung",)

    def assess(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple,
        nearby_text: list[str], component_type: str | None,
    ) -> AgentObservation:
        return self.assess_batch(context, [(subject_id, page_number, bbox, nearby_text, component_type)])[subject_id]

    def assess_batch(
        self, context: PlanAgentContext,
        subjects: list[tuple[str, int, tuple, list[str], str | None]],
    ) -> dict[str, AgentObservation]:
        results: dict[str, AgentObservation] = {}
        prompt_hash = hash_text(SYSTEM_PROMPT)
        batch_subjects: list[BatchSubjectInput] = []
        meta: dict[str, tuple[bytes, int | None, str, str | None]] = {}  # subject_id -> (crop, category, category_source, component_type)
        for subject_id, page_number, bbox, nearby_text, component_type in subjects:
            category, category_source = extract_explicit_category(nearby_text)
            crop = context.render_crop_png(page_number, bbox, margin_fraction=1.0)
            base_value = {
                "device_type": component_type, "liquid_category": category, "category_source": category_source,
                "position": "UNKNOWN", "zuordnung": None,
            }
            if crop is None:
                results[subject_id] = self._observation(
                    "sicherungseinrichtung", subject_id, base_value, confidence="uncertain",
                    evidence=["could not render a crop for position/zuordnung"], prompt_hash=prompt_hash,
                    detail={"category_source": category_source},
                )
                continue
            meta[subject_id] = (crop, category, category_source, component_type)
            batch_subjects.append(BatchSubjectInput(
                subject_id=subject_id, images=[crop],
                text=f"Known device type: {component_type or 'unknown'}.",
            ))

        if batch_subjects:
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, (crop, category, category_source, component_type) in meta.items():
                outcome = outcomes[subject_id]
                input_hash = hash_bytes(crop)
                base_value = {
                    "device_type": component_type, "liquid_category": category, "category_source": category_source,
                    "position": "UNKNOWN", "zuordnung": None,
                }
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "sicherungseinrichtung", subject_id, base_value, confidence=None, available=False,
                        error=outcome.error, model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                        detail={"category_source": category_source},
                    )
                    continue
                parsed = outcome.tool_input or {}
                value = {
                    "device_type": component_type, "liquid_category": category, "category_source": category_source,
                    "position": parsed.get("position", "UNKNOWN"), "zuordnung": parsed.get("zuordnung"),
                }
                results[subject_id] = self._observation(
                    "sicherungseinrichtung", subject_id, value, confidence=parsed.get("confidence"),
                    evidence=list(parsed.get("evidence") or []), model=outcome.model, prompt_hash=prompt_hash,
                    input_hash=input_hash, raw_response_id=outcome.raw_response_id, latency_ms=outcome.latency_ms,
                    cached=outcome.cached, detail={"category_source": category_source},
                )
        return results
