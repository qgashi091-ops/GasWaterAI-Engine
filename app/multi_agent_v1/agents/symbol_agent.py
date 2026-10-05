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

from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Small crops, single-label classification -> the full Base44 8-image cap.
DEFAULT_BATCH_SIZE = 8

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
        """Kept for single-subject callers (existing tests, ad-hoc use) --
        delegates to the batched path with a batch of exactly one, so
        behavior is identical to before this epic."""
        return self.identify_batch(context, [(subject_id, page_number, bbox, candidate_labels)])[subject_id]

    def identify_batch(
        self, context: PlanAgentContext, subjects: list[tuple[str, int, tuple, list[str]]],
    ) -> dict[str, AgentObservation]:
        """Batches multiple symbol-identification subjects into few model
        calls instead of one-per-subject. Router.py assigns every subject
        in one run the SAME `candidate_labels` list today, but this stays
        correct even if that ever changes: subjects are grouped by their
        own (sorted) candidate-label set, and each group gets its own
        batch tool_schema built from exactly that group's labels -- never
        a schema mismatched to what a subject was actually asked about."""
        results: dict[str, AgentObservation] = {}
        groups: dict[tuple, list] = {}
        for subject_id, page_number, bbox, candidate_labels in subjects:
            crop = context.render_crop_png(page_number, bbox)
            if crop is None:
                results[subject_id] = self._unavailable("component_type", subject_id, "Could not render a crop for this bbox.")
                continue
            labels_key = tuple(sorted(candidate_labels))
            groups.setdefault(labels_key, []).append((subject_id, page_number, bbox, crop))

        batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
        for labels_key, items in groups.items():
            candidate_labels = list(labels_key)
            prompt_hash = hash_text(SYSTEM_PROMPT + "|" + ",".join(candidate_labels))
            batch_subjects = [
                BatchSubjectInput(
                    subject_id=subject_id, images=[crop],
                    text=f"Candidate labels: {', '.join(candidate_labels)}, or UNKNOWN.",
                    cache_facts=labels_key,
                )
                for subject_id, _page_number, _bbox, crop in items
            ]
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _tool_schema(candidate_labels), SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, page_number, bbox, crop in items:
                outcome = outcomes[subject_id]
                input_hash = hash_bytes(crop)
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "component_type", subject_id, None, confidence=None, available=False,
                        error=outcome.error, model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    )
                    continue
                parsed = outcome.tool_input or {}
                component_type = parsed.get("component_type")
                results[subject_id] = self._observation(
                    "component_type", subject_id,
                    None if component_type in (None, "UNKNOWN") else component_type,
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    ambiguity=parsed.get("ambiguity"), model=outcome.model, prompt_hash=prompt_hash,
                    input_hash=input_hash, raw_response_id=outcome.raw_response_id, latency_ms=outcome.latency_ms,
                    cached=outcome.cached, detail={"page": page_number, "bbox": list(bbox)},
                )
        return results
