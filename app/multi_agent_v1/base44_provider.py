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
                                 this purpose (confirmed live:
                                 https://gaswaterai.base44.app/functions/aiGateway).
  BASE44_AI_GATEWAY_API_KEY  -- the shared secret this engine presents TO
                                 Base44, sent as the `x-gateway-secret`
                                 header (confirmed against Base44's live,
                                 tested endpoint -- NOT an Authorization
                                 bearer token) so Base44's gateway can
                                 authenticate the caller. This is the
                                 OPPOSITE direction from
                                 GASWATERAI_MULTI_AGENT_API_KEY (app/
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
import re
import time
import urllib.error
import urllib.request

from .provider import AgentModelProvider, AgentModelRequest, AgentModelResponse

GATEWAY_URL_ENV_VAR = "BASE44_AI_GATEWAY_URL"
GATEWAY_API_KEY_ENV_VAR = "BASE44_AI_GATEWAY_API_KEY"
REQUEST_TIMEOUT_SECONDS = 30

# Live-confirmed root cause (Render -> Base44 diagnostic probe, HTML_EDGE_WAF
# classification): Cloudflare's edge in front of Base44 blocks the Python
# urllib default User-Agent ("Python-urllib/3.x") with HTTP 403 / error 1010
# before the request ever reaches aiGateway. A normal, static User-Agent
# reaches the function (confirmed: HTTP 401 "invalid shared secret" from
# Base44 itself, not Cloudflare). This is transport-only -- it changes
# nothing about the request body, auth header, or retry behavior.
REQUEST_USER_AGENT = "Mozilla/5.0 (compatible; GasWaterAI-Engine/1.0; +https://gaswaterai.ch)"

# Error-diagnostics bounds, for the next live incident (currently: HTTP 400
# from Base44 itself, now that the Cloudflare/User-Agent block is fixed).
# Keeps the captured response body short and scrubbed -- long enough to see
# a real validation message, never long enough to carry back a whole
# echoed request payload.
MAX_HTTP_ERROR_BODY_CHARS = 500
_LONG_BASE64_RUN = re.compile(r"[A-Za-z0-9+/]{100,}={0,2}")


def _redact_secret(text: str, secret: str | None) -> str:
    if secret and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text


def _redact_long_base64_runs(text: str) -> str:
    """Defensive scrub for a Base44 error response that echoes back part of
    the request it rejected -- this engine's own images are base64-encoded
    PNGs (see `body["images"]` below), so any sufficiently long base64-like
    run is treated as image/plan data and never logged."""
    return _LONG_BASE64_RUN.sub("[IMAGE_DATA_REDACTED]", text)


def _safe_http_error_detail(exc: urllib.error.HTTPError, secret: str | None) -> str:
    """Safely describes an HTTPError's response for diagnostics: status is
    read by the caller from `exc.code`; this returns a bounded, secret- and
    image-data-redacted summary of Content-Type + body. Reads only Base44's
    own RESPONSE (never the request this engine sent), and never raises --
    a body that can't be read or decoded degrades to a short placeholder
    rather than losing the HTTP status this error is reporting."""
    try:
        raw = exc.read() if hasattr(exc, "read") else b""
    except Exception:  # noqa: BLE001
        raw = b""

    content_type = ""
    if exc.headers:
        content_type = exc.headers.get("Content-Type") or exc.headers.get("content-type") or ""

    body_text = raw.decode("utf-8", errors="replace")
    body_text = _redact_secret(body_text, secret)
    body_text = _redact_long_base64_runs(body_text)
    body_text = body_text.strip()[:MAX_HTTP_ERROR_BODY_CHARS]

    if body_text and content_type:
        return f"{content_type}: {body_text}"
    if body_text:
        return body_text
    if content_type:
        return f"{content_type} (empty body)"
    return "no response body"


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
            headers={
                "content-type": "application/json",
                "x-gateway-secret": self.api_key,
                "user-agent": REQUEST_USER_AGENT,
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw_body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = _safe_http_error_detail(exc, self.api_key)
            return AgentModelResponse(
                available=False, tool_input=None, model=request.model or "base44-ai-gateway",
                error=f"Base44 AI Gateway request failed: HTTP {exc.code}: {detail}",
            )
        except (urllib.error.URLError, TimeoutError) as exc:
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
