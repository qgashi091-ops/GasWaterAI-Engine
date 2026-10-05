"""BaseAgent -- the shared plumbing every one of the 12 agents inherits so
each agent module contains only its own narrow domain logic, never repeated
boilerplate for hashing, error handling, or observation construction.

AGENT BOUNDARY ENFORCEMENT: `agent_id` and `claim_types` are declared once
per agent class and checked on every emitted observation
(`_observation()` below raises if a subclass tries to emit a claim_type it
did not declare, or an agent_id other than its own). This is what
tests/multi_agent_v1/test_agent_boundaries.py exercises -- an agent
literally cannot emit outside its declared responsibility, it is not just a
convention."""
from __future__ import annotations

import os
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from . import timing_diagnostics
from .agent_cache import AgentResponseCache, CachedOutcome, cache_key_for
from .provider import AgentModelProvider, AgentModelRequest, AgentModelResponse
from .schema import AgentObservation, Confidence

# Base44 AI Gateway's documented limit (see docs/base44-ai-gateway-contract.md
# and the live-confirmed InvokeLLM file-attachment behaviour) -- a batch
# must NEVER exceed this many images regardless of a per-agent configured
# batch size, so this is enforced centrally here, not duplicated per agent.
MAX_IMAGES_PER_BATCH = 8


def get_batch_size(agent_id: str, default: int) -> int:
    """Per-agent configurable batch size -- GASWATERAI_BATCH_SIZE_<AGENT_ID
    upper-cased, e.g. GASWATERAI_BATCH_SIZE_LEITUNGS_AGENT>. Always clamped
    to [1, MAX_IMAGES_PER_BATCH] -- a configured value above the Base44
    Gateway's hard image limit would just silently get re-chunked smaller
    by `_chunk_subjects` below anyway, so clamping here keeps the
    diagnostics-reported `batch_sizes` honest about what was actually
    requested."""
    env_var = f"GASWATERAI_BATCH_SIZE_{agent_id.upper()}"
    try:
        value = int(os.environ.get(env_var, default))
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, MAX_IMAGES_PER_BATCH))


@dataclass
class BatchSubjectInput:
    """One subject's contribution to a batched model call. `images` is
    almost always exactly one crop today (every current agent renders one
    PNG per subject), but this stays general. `cache_facts` is a hashable
    tuple of anything that affects the answer but is NOT already baked
    into `text` or `images` (e.g. a sorted candidate-label tuple) -- the
    cache key below must cover it or two fachlich-different questions
    could collide."""

    subject_id: str
    images: list
    text: str
    cache_facts: tuple = ()


@dataclass
class BatchCallOutcome:
    tool_input: Optional[dict]
    available: bool
    error: Optional[str]
    model: Optional[str]
    cached: bool
    latency_ms: Optional[float]
    raw_response_id: Optional[str]


def _chunk_subjects(subjects: list, batch_size: int, max_images: int) -> list:
    """Splits `subjects` into chunks respecting BOTH the configured
    per-agent batch_size (subject count) and the hard max_images cap
    (summed image count) -- whichever limit a subject would exceed closes
    the current chunk first. Never reorders subjects."""
    chunks: list[list] = []
    current: list = []
    current_images = 0
    for s in subjects:
        n_images = len(s.images)
        if current and (len(current) >= batch_size or current_images + n_images > max_images):
            chunks.append(current)
            current = []
            current_images = 0
        current.append(s)
        current_images += n_images
    if current:
        chunks.append(current)
    return chunks


def _build_batch_tool_schema(tool_name: str, base_wrapper: dict) -> tuple[str, dict]:
    """Wraps an agent's EXISTING, unmodified per-subject Anthropic tool
    definition into a batched one: the model must return `results`, an
    array with one entry per subject, each entry carrying `subject_id`
    plus exactly the same fields the per-subject schema already declares.
    Still a proper Anthropic-shaped wrapper ({"name", "description",
    "input_schema"}) -- AnthropicAgentModelProvider needs that unchanged,
    and Base44AgentModelProvider's existing `_to_base44_json_schema()`
    (previous epic) unwraps `input_schema` exactly as before. No Base44
    Gateway contract change needed: `tool_schema` is still just a JSON
    Schema object, `images` is still a flat base64 list, `text` is still
    one string -- only their CONTENTS now cover several subjects."""
    base_input_schema = base_wrapper["input_schema"]
    per_subject_schema = {
        "type": "object",
        "properties": {"subject_id": {"type": "string"}, **base_input_schema.get("properties", {})},
        "required": ["subject_id", *base_input_schema.get("required", [])],
    }
    batch_tool_name = f"{tool_name}_batch"
    batch_wrapper = {
        "name": batch_tool_name,
        "description": f"{base_wrapper.get('description', '')} Batched: exactly one result per subject_id.",
        "input_schema": {
            "type": "object",
            "properties": {"results": {"type": "array", "items": per_subject_schema}},
            "required": ["results"],
        },
    }
    return batch_tool_name, batch_wrapper


class BaseAgent(ABC):
    agent_id: str = ""
    claim_types: tuple[str, ...] = ()

    def __init__(self, provider: AgentModelProvider, cache: Optional[AgentResponseCache] = None):
        if not self.agent_id:
            raise ValueError(f"{type(self).__name__} must set a non-empty agent_id class attribute.")
        if not self.claim_types:
            raise ValueError(f"{type(self).__name__} must declare at least one claim_type.")
        self.provider = provider
        self.cache = cache

    def call_model(self, request: AgentModelRequest) -> AgentModelResponse:
        """Every agent's only path to a model call -- the one central
        point that can time it (GASWATERAI_TIMING_DIAGNOSTICS=1 only) and
        attribute it to `self.agent_id`, without touching any agent's own
        logic, retry behavior, or the provider itself. When diagnostics
        are disabled, `timing_diagnostics.current()` is None and this is
        exactly `return self.provider.call(request)` -- same call, same
        count, same timeout, nothing added."""
        if timing_diagnostics.current() is None:
            return self.provider.call(request)
        t0 = time.perf_counter()
        response = self.provider.call(request)
        duration_ms = (time.perf_counter() - t0) * 1000
        timing_diagnostics.record_model_call(
            agent_id=self.agent_id, duration_ms=duration_ms,
            images=len(request.images), status=timing_diagnostics.classify_status(response),
        )
        return response

    def run_batched(
        self, subjects: list, tool_name: str, tool_schema: dict, system_prompt: str,
        batch_size: int, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 900,
    ) -> dict:
        """Shared batched-call runner every model-dependent agent's own
        `*_batch()` method delegates to. Agent-specific fachliche logic
        (what `text`/`images` to build per subject, how to turn a parsed
        result dict into an AgentObservation) stays entirely in the
        calling agent's own file -- this function only owns the TRANSPORT
        concerns: per-subject cache lookup, chunking subjects into
        batches (batch_size subjects AND <= MAX_IMAGES_PER_BATCH images,
        whichever binds first), building ONE wrapped batch tool_schema/
        request per chunk, mapping the "results" array back to subject_id
        (never by position), and isolating a whole chunk's failure to
        exactly that chunk's subjects (never raises, never retries).

        Returns `{subject_id: BatchCallOutcome}`, one entry per input
        subject, always -- a subject missing from the model's own
        response becomes `available=False` here (the calling agent turns
        that into its own UNRESOLVED/UNKNOWN value, exactly as it already
        does for any other `available=False` response)."""
        timing_diagnostics.record_subjects_model(self.agent_id, len(subjects))
        results: dict[str, BatchCallOutcome] = {}
        pending: list[BatchSubjectInput] = []
        for s in subjects:
            if self.cache is not None:
                key = cache_key_for(self.agent_id, model, system_prompt, tool_schema, s.text, s.cache_facts, s.images)
                hit = self.cache.get(key)
                if hit is not None:
                    results[s.subject_id] = BatchCallOutcome(
                        tool_input=hit.tool_input, available=hit.available, error=hit.error,
                        model=hit.model, cached=True, latency_ms=0.0, raw_response_id=hit.raw_response_id,
                    )
                    timing_diagnostics.record_cache_hit(self.agent_id)
                    continue
            pending.append(s)

        if not pending:
            return results

        batch_tool_name, batch_wrapper = _build_batch_tool_schema(tool_name, tool_schema)
        chunks = _chunk_subjects(pending, batch_size, MAX_IMAGES_PER_BATCH)
        timing_diagnostics.record_batch_plan(self.agent_id, batch_count=len(chunks), batch_sizes=[len(c) for c in chunks])

        for chunk in chunks:
            by_subject = {s.subject_id: s for s in chunk}
            combined_text = "\n\n".join(f"=== subject_id: {s.subject_id} ===\n{s.text}" for s in chunk)
            combined_images = [img for s in chunk for img in s.images]
            request = AgentModelRequest(
                system_prompt=system_prompt, tool_name=batch_tool_name, tool_schema=batch_wrapper,
                text=combined_text, images=combined_images, model=model, temperature=temperature, max_tokens=max_tokens,
            )
            response = self.call_model(request)

            if not response.available:
                for s in chunk:
                    results[s.subject_id] = BatchCallOutcome(
                        tool_input=None, available=False, error=response.error, model=response.model,
                        cached=False, latency_ms=response.latency_ms, raw_response_id=None,
                    )
                continue

            raw_results = (response.tool_input or {}).get("results")
            if not isinstance(raw_results, list):
                for s in chunk:
                    results[s.subject_id] = BatchCallOutcome(
                        tool_input=None, available=False,
                        error="Malformed batch response: 'results' missing or not a list.",
                        model=response.model, cached=False, latency_ms=response.latency_ms,
                        raw_response_id=response.raw_response_id,
                    )
                continue

            rows_by_subject: dict[str, dict] = {}
            extra_count = 0
            for row in raw_results:
                if not isinstance(row, dict):
                    extra_count += 1
                    continue
                sid = row.get("subject_id")
                if sid not in by_subject:
                    extra_count += 1
                    continue
                rows_by_subject[sid] = row
            if extra_count:
                timing_diagnostics.record_dropped_extra_results(self.agent_id, extra_count)

            for s in chunk:
                row = rows_by_subject.get(s.subject_id)
                if row is None:
                    results[s.subject_id] = BatchCallOutcome(
                        tool_input=None, available=False,
                        error="Subject missing from batch response -- treated as UNRESOLVED.",
                        model=response.model, cached=False, latency_ms=response.latency_ms,
                        raw_response_id=response.raw_response_id,
                    )
                    continue
                tool_input = {k: v for k, v in row.items() if k != "subject_id"}
                results[s.subject_id] = BatchCallOutcome(
                    tool_input=tool_input, available=True, error=None, model=response.model,
                    cached=False, latency_ms=response.latency_ms, raw_response_id=response.raw_response_id,
                )
                if self.cache is not None:
                    key = cache_key_for(self.agent_id, model, system_prompt, tool_schema, s.text, s.cache_facts, s.images)
                    self.cache.set(key, CachedOutcome(
                        tool_input=tool_input, available=True, error=None,
                        model=response.model, raw_response_id=response.raw_response_id,
                    ))
        return results

    def _observation(
        self, claim_type: str, subject_id: str, value: object,
        confidence: Optional[Confidence], evidence: Optional[list[str]] = None,
        ambiguity: Optional[str] = None, available: bool = True, error: Optional[str] = None,
        model: Optional[str] = None, prompt_hash: Optional[str] = None, input_hash: Optional[str] = None,
        cached: bool = False, latency_ms: Optional[float] = None, raw_response_id: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> AgentObservation:
        if claim_type not in self.claim_types:
            raise ValueError(
                f"{type(self).__name__} (agent_id={self.agent_id!r}) tried to emit claim_type "
                f"{claim_type!r}, which is outside its declared claim_types {self.claim_types!r}."
            )
        return AgentObservation(
            kind="AGENT_OBSERVATION", agent_id=self.agent_id, claim_type=claim_type, subject_id=subject_id,
            value=value, confidence=confidence, evidence=evidence or [], ambiguity=ambiguity,
            available=available, error=error, model=model, prompt_hash=prompt_hash, input_hash=input_hash,
            cached=cached, latency_ms=latency_ms, raw_response_id=raw_response_id, detail=detail or {},
        )

    def _unavailable(self, claim_type: str, subject_id: str, error: str) -> AgentObservation:
        return self._observation(claim_type, subject_id, value=None, confidence=None, available=False, error=error)
