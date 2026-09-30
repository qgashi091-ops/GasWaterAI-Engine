"""AgentModelProvider -- the one abstraction every one of the 12 agents
calls through, so the underlying model/vendor is swappable without touching
any agent. Deliberately more general than `app/vision_fallback/interface.py`
(which is narrowly "classify this crop against a label list"): agents here
need arbitrary tool schemas (routing decisions, topology classifications,
category extraction, ...), so the provider's contract is "run this one
forced-tool-use call and hand back the parsed tool input", nothing more
provider-specific than that leaks into an agent.

Two concrete providers:
  - AnthropicAgentModelProvider: direct Anthropic Messages API call, reusing
    the SAME credential/config discipline already established for Engine v1
    and the Micro-Agent POC (GASWATERAI_VISION_API_KEY, never a harness
    credential -- see app/vision_fallback/anthropic_provider.py's docstring).
    No such key exists in this build/development session (confirmed absent
    there and re-confirmed here), so this provider is implemented and unit
    tested against a mocked HTTP layer only -- exactly per this epic's own
    instruction ("Live-Modellaufrufe sind nicht Voraussetzung... mit
    Fixtures/Mocks testen und weiterbauen").
  - FixtureAgentModelProvider: a deterministic, file/dict-backed provider
    for tests and for running the whole pipeline end-to-end with no
    credential at all -- every agent's tests and the integration pipeline
    test use this, never a live network call.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from app.vision_fallback.anthropic_provider import (
    ANTHROPIC_VERSION,
    API_KEY_ENV_VAR,
    API_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT_SECONDS,
)


@dataclass
class AgentModelRequest:
    """Everything one agent call needs, provider-agnostic. `images` are raw
    PNG bytes (already rendered crops -- this package never renders a PDF
    page itself; that stays in app/plan_analysis). `text` is the small,
    already-deterministic context block the caller composed -- never a full
    plan dump (see each agent module's own docstring for exactly what it
    includes)."""

    system_prompt: str
    tool_name: str
    tool_schema: dict
    text: str
    images: list[bytes] = field(default_factory=list)
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 400


@dataclass
class AgentModelResponse:
    available: bool
    tool_input: Optional[dict]
    model: str
    raw_response_id: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "available": self.available, "tool_input": self.tool_input, "model": self.model,
            "raw_response_id": self.raw_response_id, "latency_ms": self.latency_ms, "error": self.error,
        }


class AgentModelProvider(ABC):
    @abstractmethod
    def call(self, request: AgentModelRequest) -> AgentModelResponse:
        raise NotImplementedError


class AnthropicAgentModelProvider(AgentModelProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        self.model = model or os.environ.get("GASWATERAI_VISION_MODEL", DEFAULT_MODEL)

    def call(self, request: AgentModelRequest) -> AgentModelResponse:
        model = request.model or self.model
        if not self.api_key:
            return AgentModelResponse(
                available=False, tool_input=None, model=model,
                error=f"No API key configured ({API_KEY_ENV_VAR} not set).",
            )

        import base64

        content: list[dict] = []
        for img in request.images:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": base64.b64encode(img).decode("ascii"),
            }})
        content.append({"type": "text", "text": request.text})

        body = {
            "model": model, "max_tokens": request.max_tokens, "temperature": request.temperature,
            "system": request.system_prompt,
            "tools": [request.tool_schema],
            "tool_choice": {"type": "tool", "name": request.tool_name},
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
            return AgentModelResponse(available=False, tool_input=None, model=model, error=f"Request failed: {exc}")
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            tool_use = next(b for b in payload["content"] if b.get("type") == "tool_use")
            return AgentModelResponse(
                available=True, tool_input=tool_use["input"], model=model,
                raw_response_id=payload.get("id"), latency_ms=latency_ms,
            )
        except (KeyError, StopIteration, TypeError) as exc:
            return AgentModelResponse(available=False, tool_input=None, model=model, error=f"Could not parse response: {exc}")


class FixtureAgentModelProvider(AgentModelProvider):
    """Deterministic, no-network provider for tests and credential-free
    pipeline runs. `responses` maps `tool_name` -> either a fixed dict
    (always returned) or a callable `(request) -> dict | None` (None means
    "simulate unavailable", matching how the real provider behaves with no
    key). Every call is logged in `.calls` for test assertions (call count
    == cost diagnostics correctness)."""

    def __init__(self, responses: Optional[dict] = None, model: str = "fixture-model"):
        self.responses = responses or {}
        self.model = model
        self.calls: list[AgentModelRequest] = []

    def call(self, request: AgentModelRequest) -> AgentModelResponse:
        self.calls.append(request)
        handler = self.responses.get(request.tool_name)
        if handler is None:
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or self.model,
                error=f"FixtureAgentModelProvider has no response registered for tool {request.tool_name!r}.",
            )
        tool_input = handler(request) if callable(handler) else handler
        if tool_input is None:
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or self.model,
                error="Fixture simulated an unavailable model call.",
            )
        return AgentModelResponse(
            available=True, tool_input=tool_input, model=request.model or self.model,
            raw_response_id=f"fixture-{len(self.calls)}", latency_ms=0.0,
        )
