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

from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Small crops, short label-association decision -> the full 8-image cap.
DEFAULT_BATCH_SIZE = 8

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
        return self.associate_batch(context, [(subject_id, page_number, text_bbox, text)])[subject_id]

    def associate_batch(
        self, context: PlanAgentContext, subjects: list[tuple[str, int, tuple, str]],
    ) -> dict[str, AgentObservation]:
        results: dict[str, AgentObservation] = {}
        prompt_hash = hash_text(SYSTEM_PROMPT)
        batch_subjects: list[BatchSubjectInput] = []
        meta: dict[str, tuple[int, str]] = {}
        for subject_id, page_number, text_bbox, text in subjects:
            crop = context.render_crop_png(page_number, text_bbox, margin_fraction=1.5)
            if crop is None:
                results[subject_id] = self._observation(
                    "label_association", subject_id, {"target_type": "unassigned", "target_id": None},
                    confidence="uncertain", evidence=["could not render surrounding crop"], prompt_hash=prompt_hash,
                )
                continue
            meta[subject_id] = (page_number, text)
            batch_subjects.append(BatchSubjectInput(
                subject_id=subject_id, images=[crop], text=f"The text in question: {text!r}",
            ))

        if batch_subjects:
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, (page_number, text) in meta.items():
                outcome = outcomes[subject_id]
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "label_association", subject_id, {"target_type": "unassigned", "target_id": None},
                        confidence=None, available=False, error=outcome.error, model=outcome.model,
                        prompt_hash=prompt_hash,
                    )
                    continue
                parsed = outcome.tool_input or {}
                results[subject_id] = self._observation(
                    "label_association", subject_id,
                    {"target_type": parsed.get("target_type", "unassigned"), "target_id": None},
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    model=outcome.model, prompt_hash=prompt_hash, raw_response_id=outcome.raw_response_id,
                    latency_ms=outcome.latency_ms, cached=outcome.cached, detail={"page": page_number, "text": text},
                )
        return results
