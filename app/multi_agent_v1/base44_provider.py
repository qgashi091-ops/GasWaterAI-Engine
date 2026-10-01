"""Base44AgentModelProvider -- a second, alternate implementation of the
EXISTING `AgentModelProvider` abstraction (provider.py), so the 12 agents
can run their model-dependent calls through a Base44-operated AI Gateway /
adapter instead of requiring a separately-paid Anthropic API key for product
operation. No agent, prompt, or tool schema changes at all: every agent
already only depends on `AgentModelProvider.call(AgentModelRequest) ->
AgentModelResponse` (see base_agent.py), which this class implements exactly
like `AnthropicAgentModelProvider` does -- it is purely an alternate
TRANSPORT for the same request/response contract, selected by configuration
(see app/main.py's `_select_agent_model_provider`), never by agent code.

REQUIRED CONFIGURATION (read from the environment, never hardcoded, never
committed -- same discipline as GASWATERAI_VISION_API_KEY):
  BASE44_AI_GATEWAY_URL      -- the full HTTPS endpoint Base44 exposes for
                                 this purpose (this engine's own URL/path
                                 choice is Base44's to decide; nothing here
                                 assumes a specific path shape).
  BASE44_AI_GATEWAY_API_KEY  -- a bearer credential this engine presents TO
                                 Base44 (sent as `Authorization: Bearer
                                 <key>`) so Base44's gateway can authenticate
                                 the caller. This is the OPPOSITE direction
                                 from GASWATERAI_MULTI_AGENT_API_KEY (app/
                                 main.py), which protects calls INTO this
                                 engine from Base44.

See docs/base44-ai-gateway-contract.md for the exact JSON request/response
shape Base44's gateway must implement -- it mirrors AgentModelRequest/
AgentModelResponse (provider.py) field-for-field by design, so there is
nothing for Base44 to interpret beyond serializing/deserializing those same
fields.

Not exercised against a live endpoint in this session (none exists yet --
same "implement and test against fixtures, continue" discipline already
established for AnthropicAgentModelProvider when no vision credential was
available; see that module's own docstring). Fully covered by
tests/multi_agent_v1/test_base44_provider.py against a mocked HTTP layer.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

from .provider import AgentModelProvider, AgentModelRequest, AgentModelResponse

GATEWAY_URL_ENV_VAR = "BASE44_AI_GATEWAY_URL"
GATEWAY_API_KEY_ENV_VAR = "BASE44_AI_GATEWAY_API_KEY"
REQUEST_TIMEOUT_SECONDS = 30


class Base44AgentModelProvider(AgentModelProvider):
    def __init__(self, gateway_url: str | None = None, api_key: str | None = None):
        self.gateway_url = gateway_url or os.environ.get(GATEWAY_URL_ENV_VAR)
        self.api_key = api_key or os.environ.get(GATEWAY_API_KEY_ENV_VAR)

    def call(self, request: AgentModelRequest) -> AgentModelResponse:
        if not self.gateway_url or not self.api_key:
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or "base44-ai-gateway",
                error=(
                    f"Base44 AI Gateway not configured "
                    f"({GATEWAY_URL_ENV_VAR} / {GATEWAY_API_KEY_ENV_VAR} not set)."
                ),
            )

        body = {
            "system_prompt": request.system_prompt,
            "tool_name": request.tool_name,
            "tool_schema": request.tool_schema,
            "text": request.text,
            "images": [base64.b64encode(img).decode("ascii") for img in request.images],
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        req = urllib.request.Request(
            self.gateway_url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw_body = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or "base44-ai-gateway",
                error=f"Base44 AI Gateway request failed: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            payload = json.loads(raw_body.decode("utf-8"))
            return AgentModelResponse(
                available=bool(payload["available"]),
                tool_input=payload.get("tool_input"),
                model=payload.get("model") or request.model or "base44-ai-gateway",
                raw_response_id=payload.get("raw_response_id"),
                latency_ms=latency_ms,
                error=payload.get("error"),
            )
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or "base44-ai-gateway",
                error=f"Could not parse Base44 AI Gateway response: {exc}",
            )
