"""Agent 1/12 -- PlanstrukturAgent.

Sole responsibility: classify WHICH KIND of plan area one page is --
SCHEMA, GRUNDRISS, DETAIL, LEGENDE, or UNKNOWN. Nothing about components,
topology, or compliance is this agent's concern.

Two-tier, cost-minimizing method:
  1. Deterministic-first: reuse `app.plan_analysis.legend_detection`
     UNMODIFIED (same reuse pattern already established for Detector v2 --
     see scripts/detector_v2_candidate_generation.py) to check whether a
     legend/table region covers a large fraction of the page; if so, this
     is answered with NO model call at all. A small, explicit German
     keyword heuristic over the page's own text (schema/grundriss/detail
     headings) covers the next-cheapest case.
  2. Only when neither deterministic signal is decisive does this agent
     call the model with a low-resolution whole-page overview image --
     never full resolution, never any other page's content.

This agent's output is always an AGENT_OBSERVATION, even when derived from
the cheap deterministic heuristics above: there is no existing "plan area
kind" fact anywhere in plan_facts.py for this to wrap, so this agent's
classification (however it was reached) is its own claim, never a
DeterministicFactRef.
"""
from __future__ import annotations

import re

from app.plan_analysis.legend_detection import detect_legend_candidates
from app.microagents.hashing import hash_text

from .. import timing_diagnostics
from ..base_agent import BaseAgent, BatchSubjectInput, get_batch_size
from ..context import PlanAgentContext
from ..schema import AgentObservation

# Whole-page overview images are heavier than a small crop -> smaller batch.
DEFAULT_BATCH_SIZE = 4

PLAN_AREA_KINDS = ("SCHEMA", "GRUNDRISS", "DETAIL", "LEGENDE", "UNKNOWN")

_KEYWORD_PATTERNS = {
    "SCHEMA": re.compile(r"\bschema\b|steigschema|prinzipschema", re.IGNORECASE),
    "GRUNDRISS": re.compile(r"grundriss", re.IGNORECASE),
    "DETAIL": re.compile(r"\bdetail\b|ausschnitt", re.IGNORECASE),
    "LEGENDE": re.compile(r"legende", re.IGNORECASE),
}

SYSTEM_PROMPT = (
    "You classify the TYPE of one page from a Swiss potable-water plumbing "
    "plan set. Choose exactly one: SCHEMA (a schematic riser/pipe diagram, "
    "not drawn to architectural scale), GRUNDRISS (a floor plan showing "
    "rooms/walls with pipe routing overlaid), DETAIL (a zoomed-in detail of "
    "one small installation), LEGENDE (a symbol/legend table), or UNKNOWN if "
    "genuinely unclear. Base your answer only on the visual layout shown. "
    "Respond only via the classify_plan_area tool."
)

TOOL_NAME = "classify_plan_area"
_TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": "Classify the plan-area kind of the page shown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_area_kind": {"type": "string", "enum": list(PLAN_AREA_KINDS)},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
        },
        "required": ["plan_area_kind", "evidence", "confidence"],
    },
}


class PlanstrukturAgent(BaseAgent):
    agent_id = "planstruktur_agent"
    claim_types = ("plan_area_kind",)

    def classify_page(self, context: PlanAgentContext, page_number: int) -> AgentObservation:
        subject_id = f"p{page_number}"
        return self.classify_pages_batch(context, [page_number])[subject_id]

    def classify_pages_batch(self, context: PlanAgentContext, page_numbers: list[int]) -> dict[str, AgentObservation]:
        results: dict[str, AgentObservation] = {}
        batch_subjects: list[BatchSubjectInput] = []
        deterministic_skip = 0
        for page_number in page_numbers:
            subject_id = f"p{page_number}"
            page = context.page(page_number)
            if page is None:
                results[subject_id] = self._unavailable("plan_area_kind", subject_id, f"No such page: {page_number}")
                continue

            # --- Tier 1: legend-region coverage (reused, unmodified detector) ---
            rotation_matrix = None  # v0.1 TextSpan.bbox is already display-space here (see plan_facts docstring on v0.1 frames)
            fake_page = type("P", (), {"text_spans": page.text_spans, "page_number": page_number})()
            try:
                legend_candidates = detect_legend_candidates(fake_page, rotation_matrix=rotation_matrix)
            except Exception:
                legend_candidates = []
            page_area = max(page.width * page.height, 1e-6)
            legend_hit = None
            for cand in legend_candidates:
                if not cand.heading_text:
                    continue
                x0, y0, x1, y1 = cand.bbox
                coverage = abs((x1 - x0) * (y1 - y0)) / page_area
                if coverage >= 0.35:
                    legend_hit = (cand, coverage)
                    break
            if legend_hit is not None:
                cand, coverage = legend_hit
                results[subject_id] = self._observation(
                    "plan_area_kind", subject_id, "LEGENDE", confidence="supported",
                    evidence=[f"legend region {cand.heading_text!r} covers {coverage:.0%} of the page"],
                    detail={"method": "legend_coverage", "legend_id": cand.legend_id},
                )
                deterministic_skip += 1
                continue

            # --- Tier 2: keyword heuristic over the page's own text ---
            joined = "\n".join(s.text for s in page.text_spans)
            keyword_hits = {k: bool(p.search(joined)) for k, p in _KEYWORD_PATTERNS.items()}
            matched = [k for k, hit in keyword_hits.items() if hit]
            if len(matched) == 1:
                results[subject_id] = self._observation(
                    "plan_area_kind", subject_id, matched[0], confidence="supported",
                    evidence=[f"page text contains a {matched[0]!r}-type heading"],
                    detail={"method": "keyword_heuristic"},
                )
                deterministic_skip += 1
                continue

            # --- Tier 3: model fallback on a low-resolution page overview ---
            crop = context.render_full_page_png(page_number)
            if crop is None:
                results[subject_id] = self._observation(
                    "plan_area_kind", subject_id, "UNKNOWN", confidence="uncertain",
                    evidence=["no deterministic signal available and no page image could be rendered"],
                    detail={"method": "fallback_unknown"},
                )
                continue
            batch_subjects.append(BatchSubjectInput(subject_id=subject_id, images=[crop], text="Classify this plan page."))

        if deterministic_skip:
            timing_diagnostics.record_subjects_deterministic_skip(self.agent_id, deterministic_skip)
        if batch_subjects:
            prompt_hash = hash_text(SYSTEM_PROMPT)
            batch_size = get_batch_size(self.agent_id, DEFAULT_BATCH_SIZE)
            outcomes = self.run_batched(batch_subjects, TOOL_NAME, _TOOL_SCHEMA, SYSTEM_PROMPT, batch_size=batch_size)
            for s in batch_subjects:
                outcome = outcomes[s.subject_id]
                if not outcome.available:
                    results[s.subject_id] = self._observation(
                        "plan_area_kind", s.subject_id, "UNKNOWN", confidence="uncertain",
                        evidence=[], available=False, error=outcome.error, model=outcome.model,
                        prompt_hash=prompt_hash, detail={"method": "model_unavailable"},
                    )
                    continue
                parsed = outcome.tool_input or {}
                results[s.subject_id] = self._observation(
                    "plan_area_kind", s.subject_id, parsed.get("plan_area_kind", "UNKNOWN"),
                    confidence=parsed.get("confidence"), evidence=list(parsed.get("evidence") or []),
                    model=outcome.model, prompt_hash=prompt_hash, raw_response_id=outcome.raw_response_id,
                    latency_ms=outcome.latency_ms, cached=outcome.cached, detail={"method": "model"},
                )
        return results
