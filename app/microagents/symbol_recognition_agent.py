"""SymbolRecognitionAgent -- MICRO-AGENT ARCHITECTURE POC, Phase 2/3.

Narrow responsibility, minimal context, per this POC's core principle: this
agent answers exactly one question -- "what component is visibly supported
by this crop?" -- and nothing else. It NEVER receives the full PDF, the 150
reference cases, SVGW rule text, Golden findings, topology conclusions, or
another agent's reasoning. Its system prompt is a few short sentences, not
a professional-rules document, and it never asks for chain-of-thought.

Calls the direct external multimodal API interface already built for
Engine v1 (`app/vision_fallback/anthropic_provider.py`'s config/constants --
same credential, same "direct API, not Base44 InvokeLLM" discipline), but
with its OWN tool schema and prompt: this agent's output contract
(`evidence` list, `ambiguity`, a categorical `confidence`) is intentionally
different from `VisionFallbackResult`'s (a numeric `confidence`), because
this is a different, more narrowly-scoped role than the general vision
fallback used inside `component_evidence.py`.

OUTPUT IS ALWAYS AN OBSERVATION, NEVER A PLAN_FACT. Nothing in this module
writes to PlanFacts, the canonical inventory, or any rule result. Promotion
to a COMPONENT_FACT requires independent corroboration elsewhere (see this
package's own __init__.py).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from app.vision_fallback.anthropic_provider import (
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR,
    API_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_SECONDS,
)

from .cache import ResponseCache
from .hashing import cache_key, combine_hashes, hash_bytes, hash_text

TEMPERATURE = 0  # determinism control -- see Phase 4's own stability question this is meant to help answer

# Kept deliberately short -- Phase 3's own discipline: no long professional
# rule text, no chain-of-thought request, no broad plan interpretation.
SYSTEM_PROMPT = (
    "You identify a single plumbing/sanitary component shown in an image "
    "crop from a technical plan. Choose exactly one label from the given "
    "candidate list, or UNKNOWN if the crop does not clearly show one of "
    "them. Base your answer only on what is visually present. A cautious "
    "UNKNOWN is better than a confident wrong guess -- never force a "
    "classification you are not sure of. Respond only via the "
    "classify_symbol tool."
)

TOOL_NAME = "classify_symbol"


def _tool_schema(candidate_labels: list[str]) -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Classify the plumbing/sanitary component in the image crop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_type": {
                    "type": "string",
                    "enum": [*candidate_labels, "UNKNOWN"],
                    "description": "Exactly one candidate label, or UNKNOWN.",
                },
                "evidence": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Short phrases naming the specific visual features that support the answer.",
                },
                "ambiguity": {
                    "type": ["string", "null"],
                    "description": "If genuinely ambiguous with another candidate, name it; otherwise null.",
                },
                "confidence": {"type": "string", "enum": ["supported", "uncertain"]},
            },
            "required": ["component_type", "evidence", "ambiguity", "confidence"],
        },
    }


@dataclass
class AgentObservation:
    kind: str  # always "OBSERVATION" -- see module docstring
    benchmark_id: str
    component_type: str | None  # None when UNKNOWN
    evidence: list[str]
    ambiguity: str | None
    confidence: str | None  # "supported" | "uncertain" | None on failure
    model: str
    prompt_hash: str
    crop_hash: str
    text_hash: str
    cached: bool
    condition: str  # "image_only" | "image_plus_text"
    available: bool
    error: str | None = None
    raw_response_id: str | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "benchmark_id": self.benchmark_id,
            "component_type": self.component_type, "evidence": self.evidence,
            "ambiguity": self.ambiguity, "confidence": self.confidence,
            "model": self.model, "prompt_hash": self.prompt_hash,
            "crop_hash": self.crop_hash, "text_hash": self.text_hash,
            "cached": self.cached, "condition": self.condition,
            "available": self.available, "error": self.error,
            "raw_response_id": self.raw_response_id, "latency_ms": self.latency_ms,
        }


class SymbolRecognitionAgent:
    def __init__(
        self, candidate_labels: list[str], cache_dir: Path,
        api_key: str | None = None, model: str | None = None,
    ):
        import os
        self.candidate_labels = sorted(candidate_labels)
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        self.model = model or os.environ.get("GASWATERAI_VISION_MODEL", DEFAULT_MODEL)
        self.cache = ResponseCache(cache_dir)
        self.prompt_hash = hash_text(SYSTEM_PROMPT + "|" + ",".join(self.candidate_labels))

    def recognize(
        self, benchmark_id: str, tight_crop_bytes: bytes,
        context_crop_bytes: bytes | None = None,
        nearby_text: list[str] | None = None,
        use_cache: bool = True,
    ) -> AgentObservation:
        """`nearby_text=None` (or an empty list) is CONDITION A (image
        only, per Phase 6). Passing a non-empty list is CONDITION B (image
        + local text) -- the only difference between the two conditions,
        nothing else about the prompt or schema changes."""
        tight_hash = hash_bytes(tight_crop_bytes)
        context_hash = hash_bytes(context_crop_bytes) if context_crop_bytes else "none"
        crop_hash = combine_hashes(tight_hash, context_hash)
        text_joined = "\n".join(nearby_text) if nearby_text else ""
        text_hash = hash_text(text_joined)
        condition = "image_plus_text" if nearby_text else "image_only"

        key = cache_key(self.model, self.prompt_hash, crop_hash, text_hash)
        if use_cache:
            cached_value = self.cache.get(key)
            if cached_value is not None:
                return AgentObservation(**{**cached_value, "cached": True})

        if not self.api_key:
            obs = AgentObservation(
                kind="OBSERVATION", benchmark_id=benchmark_id, component_type=None,
                evidence=[], ambiguity=None, confidence=None, model=self.model,
                prompt_hash=self.prompt_hash, crop_hash=crop_hash, text_hash=text_hash,
                cached=False, condition=condition, available=False,
                error=f"No API key configured ({API_KEY_ENV_VAR} not set).",
            )
            return obs

        content = [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(tight_crop_bytes).decode("ascii"),
        }}]
        if context_crop_bytes:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(context_crop_bytes).decode("ascii"),
            }})
        text_block = f"Candidate labels: {', '.join(self.candidate_labels)}, or UNKNOWN."
        if nearby_text:
            text_block += f"\nText extracted deterministically near this crop on the plan: {json.dumps(nearby_text, ensure_ascii=False)}"
        content.append({"type": "text", "text": text_block})

        body = {
            "model": self.model, "max_tokens": 300, "temperature": TEMPERATURE,
            "system": SYSTEM_PROMPT,
            "tools": [_tool_schema(self.candidate_labels)],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "messages": [{"role": "user", "content": content}],
        }
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            return AgentObservation(
                kind="OBSERVATION", benchmark_id=benchmark_id, component_type=None,
                evidence=[], ambiguity=None, confidence=None, model=self.model,
                prompt_hash=self.prompt_hash, crop_hash=crop_hash, text_hash=text_hash,
                cached=False, condition=condition, available=False, error=f"Request failed: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        obs = self._parse(payload, benchmark_id, crop_hash, text_hash, condition, latency_ms)
        if use_cache and obs.available:
            self.cache.set(key, {k: v for k, v in obs.to_dict().items() if k != "cached"})
        return obs

    def _parse(self, payload: dict, benchmark_id: str, crop_hash: str, text_hash: str, condition: str, latency_ms: float) -> AgentObservation:
        try:
            tool_use = next(b for b in payload["content"] if b.get("type") == "tool_use")
            parsed = tool_use["input"]
            component_type = parsed["component_type"]
            return AgentObservation(
                kind="OBSERVATION", benchmark_id=benchmark_id,
                component_type=None if component_type == "UNKNOWN" else component_type,
                evidence=list(parsed.get("evidence") or []), ambiguity=parsed.get("ambiguity"),
                confidence=parsed.get("confidence"), model=self.model, prompt_hash=self.prompt_hash,
                crop_hash=crop_hash, text_hash=text_hash, cached=False, condition=condition,
                available=True, raw_response_id=payload.get("id"), latency_ms=latency_ms,
            )
        except (KeyError, StopIteration, TypeError) as exc:
            return AgentObservation(
                kind="OBSERVATION", benchmark_id=benchmark_id, component_type=None,
                evidence=[], ambiguity=None, confidence=None, model=self.model,
                prompt_hash=self.prompt_hash, crop_hash=crop_hash, text_hash=text_hash,
                cached=False, condition=condition, available=False, error=f"Could not parse response: {exc}",
            )
