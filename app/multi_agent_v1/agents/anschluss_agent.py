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

from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Live baseline: 11 AnschlussAgent calls. Moderate crop complexity.
DEFAULT_BATCH_SIZE = 6

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
        return self.classify_connections_batch(context, [(subject_id, page_number, bbox)])[subject_id]

    def classify_connections_batch(
        self, context: PlanAgentContext, subjects: list[tuple[str, int, tuple]],
    ) -> dict[str, AgentObservation]:
        results: dict[str, AgentObservation] = {}
        prompt_hash = hash_text(SYSTEM_PROMPT)
        batch_subjects: list[BatchSubjectInput] = []
        crops: dict[str, bytes] = {}
        for subject_id, page_number, bbox in subjects:
            crop = context.render_crop_png(page_number, bbox, margin_fraction=1.0)
            if crop is None:
                results[subject_id] = self._observation(
                    "anschluss_topology", subject_id, "UNKNOWN", confidence="uncertain",
                    evidence=["could not render a crop"], prompt_hash=prompt_hash,
                )
                continue
            crops[subject_id] = crop
            batch_subjects.append(BatchSubjectInput(
                subject_id=subject_id, images=[crop], text="Classify the highlighted component's connection role.",
            ))

        if batch_subjects:
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for subject_id, crop in crops.items():
                outcome = outcomes[subject_id]
                input_hash = hash_bytes(crop)
                if not outcome.available:
                    results[subject_id] = self._observation(
                        "anschluss_topology", subject_id, "UNKNOWN", confidence=None, available=False,
                        error=outcome.error, model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    )
                    continue
                parsed = outcome.tool_input or {}
                results[subject_id] = self._observation(
                    "anschluss_topology", subject_id, parsed.get("role", "UNKNOWN"),
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    model=outcome.model, prompt_hash=prompt_hash, input_hash=input_hash,
                    raw_response_id=outcome.raw_response_id, latency_ms=outcome.latency_ms,
                    cached=outcome.cached,
                )
        return results
