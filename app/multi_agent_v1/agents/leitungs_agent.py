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

from .. import timing_diagnostics
from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Live baseline: 54 LeitungsAgent model calls on the first real plan --
# the single biggest contributor to total runtime. Full 8-image cap.
DEFAULT_BATCH_SIZE = 8

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
        return self.classify_media_batch(context, [(subject_id, page_number, edge_bbox)])[subject_id]

    def classify_media_batch(
        self, context: PlanAgentContext, subjects: list[tuple[str, int, tuple]],
    ) -> dict[str, AgentObservation]:
        """Section 7 (PlanFacts-Vorrang/deterministic-first): the
        text-pattern tier below is UNCHANGED and runs for every subject
        BEFORE any batching decision -- a subject resolved by nearby text
        never enters a batch or consumes a model call at all."""
        results: dict[str, AgentObservation] = {}
        batch_subjects: list[BatchSubjectInput] = []
        meta: dict[str, tuple[int, bytes]] = {}
        deterministic_skip = 0
        for subject_id, page_number, edge_bbox in subjects:
            nearby = context.nearby_text(page_number, edge_bbox, margin_pt=40.0)
            joined = "\n".join(nearby)
            matched_medium = None
            for medium, pattern in _PATTERNS.items():
                if pattern.search(joined):
                    matched_medium = medium
                    break
            if matched_medium is not None:
                results[subject_id] = self._observation(
                    "leitung_medium", subject_id, matched_medium, confidence="supported",
                    evidence=[f"nearby text matches {matched_medium!r} pattern"],
                    detail={"method": "text_pattern", "nearby_text": nearby},
                )
                deterministic_skip += 1
                continue

            crop = context.render_crop_png(page_number, edge_bbox, margin_fraction=0.8)
            prompt_hash = hash_text(SYSTEM_PROMPT)
            if crop is None:
                results[subject_id] = self._observation(
                    "leitung_medium", subject_id, "UNKNOWN", confidence="uncertain",
                    evidence=["no nearby text match and no crop could be rendered"], prompt_hash=prompt_hash,
                )
                continue
            meta[subject_id] = (page_number, crop)
            batch_subjects.append(BatchSubjectInput(
                subject_id=subject_id, images=[crop], text="Classify the medium of the highlighted pipe segment.",
            ))

        if deterministic_skip:
            timing_diagnostics.record_subjects_deterministic_skip(self.agent_id, deterministic_skip)
        if batch_subjects:
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, (page_number, crop) in meta.items():
                outcome = outcomes[subject_id]
                prompt_hash = hash_text(SYSTEM_PROMPT)
                input_hash = hash_bytes(crop)
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "leitung_medium", subject_id, "UNKNOWN", confidence=None, available=False,
                        error=outcome.error, model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    )
                    continue
                parsed = outcome.tool_input or {}
                results[subject_id] = self._observation(
                    "leitung_medium", subject_id, parsed.get("medium", "UNKNOWN"),
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    raw_response_id=outcome.raw_response_id, latency_ms=outcome.latency_ms,
                    cached=outcome.cached, detail={"method": "model"},
                )
        return results
