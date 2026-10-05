"""Agent 6/12 -- SchlaufungsAgent.

Sole responsibility: classify Schlaufe (loop) / Weiterführung (continuation)
/ Stichleitung (branch stub) / Blindende (blind end) -- ONLY for a node/path
plan_facts.py left UNRESOLVED. This is the agent the epic singles out by
name for the non-negotiable rule: "dead_end_confirmed, cycle_membership,
PHYSICAL_END usw. niemals überschreiben."

Enforcement is belt-and-suspenders, not just router-level:
  1. router.py never routes this agent to a subject whose plan_facts
     cycle_membership/dead_end_path/terminal_endpoint is already resolved
     (kind != "UNRESOLVED") -- see router.py's `schlaufungs_relevant_subjects`.
  2. This agent's own `classify()` method ALSO checks
     `context.is_resolved(subject_id, "schlaufung_topology")` itself and
     refuses (returns an unavailable observation, calls no model) if asked
     about an already-resolved subject -- so even a future router bug could
     not make this agent silently produce a competing claim.
  3. evidence_merger.py's priority rule is the final backstop: even if an
     observation somehow existed for a resolved subject, the merger always
     prefers the resolved DeterministicFactRef.
"""
from __future__ import annotations

from app.microagents.hashing import hash_bytes, hash_text

from .. import timing_diagnostics
from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Live baseline: 10 SchlaufungsAgent calls. Moderate crop complexity.
DEFAULT_BATCH_SIZE = 6

CLASSIFICATION_VALUES = ("SCHLAUFE", "WEITERFUEHRUNG", "STICHLEITUNG", "BLINDENDE", "UNRESOLVED")

SYSTEM_PROMPT = (
    "You look at a small cropped region of a technical plumbing plan "
    "showing a pipe end or junction that could not be classified purely "
    "from the drawn vector geometry (for example because a reconstructed "
    "'bridge' connection makes the true continuation uncertain). Decide, "
    "from what is visually drawn, whether this is: SCHLAUFE (the line "
    "visibly loops back into another drawn line), WEITERFUEHRUNG (the pipe "
    "visibly continues beyond what the vector graph captured), "
    "STICHLEITUNG (a branch stub feeding one fixture), BLINDENDE (a genuine "
    "capped/blind end), or UNRESOLVED if the image itself does not make it "
    "clear either. Respond only via the classify_schlaufung tool."
)

TOOL_NAME = "classify_schlaufung"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify an otherwise-unresolved pipe-end/junction situation.",
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


class SchlaufungsAgent(BaseAgent):
    agent_id = "schlaufungs_agent"
    claim_types = ("schlaufung_topology",)

    def classify(
        self, context: PlanAgentContext, subject_id: str, page_number: int, bbox: tuple,
    ) -> AgentObservation:
        return self.classify_batch(context, [(subject_id, page_number, bbox)])[subject_id]

    def classify_batch(
        self, context: PlanAgentContext, subjects: list[tuple[str, int, tuple]],
    ) -> dict[str, AgentObservation]:
        """The belt-and-suspenders `is_resolved` refusal (module docstring,
        point 2) still runs per subject BEFORE anything else -- an
        already-PlanFacts-resolved subject never renders a crop, never
        enters a batch, never consumes a model call."""
        results: dict[str, AgentObservation] = {}
        batch_subjects: list[BatchSubjectInput] = []
        crops: dict[str, bytes] = {}
        refused = 0
        for subject_id, page_number, bbox in subjects:
            if context.is_resolved(subject_id, "schlaufung_topology"):
                results[subject_id] = self._observation(
                    "schlaufung_topology", subject_id, None, confidence=None, available=False,
                    error=(
                        "Refused: plan_facts already resolved this subject. This agent must never be "
                        "asked about, and never overrides, an already-resolved dead-end/cycle/terminal fact."
                    ),
                    detail={"refused": True},
                )
                refused += 1
                continue

            crop = context.render_crop_png(page_number, bbox, margin_fraction=1.2)
            prompt_hash = hash_text(SYSTEM_PROMPT)
            if crop is None:
                results[subject_id] = self._observation(
                    "schlaufung_topology", subject_id, "UNRESOLVED", confidence="uncertain",
                    evidence=["could not render a crop"], prompt_hash=prompt_hash,
                )
                continue
            crops[subject_id] = crop
            batch_subjects.append(BatchSubjectInput(
                subject_id=subject_id, images=[crop], text="Classify this otherwise-unresolved pipe end/junction.",
            ))

        if refused:
            timing_diagnostics.record_subjects_deterministic_skip(self.agent_id, refused)
        if batch_subjects:
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, crop in crops.items():
                outcome = outcomes[subject_id]
                prompt_hash = hash_text(SYSTEM_PROMPT)
                input_hash = hash_bytes(crop)
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "schlaufung_topology", subject_id, "UNRESOLVED", confidence=None, available=False,
                        error=outcome.error, model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    )
                    continue
                parsed = outcome.tool_input or {}
                results[subject_id] = self._observation(
                    "schlaufung_topology", subject_id, parsed.get("classification", "UNRESOLVED"),
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    raw_response_id=outcome.raw_response_id, latency_ms=outcome.latency_ms,
                    cached=outcome.cached,
                )
        return results
