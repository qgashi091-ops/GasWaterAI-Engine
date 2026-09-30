"""Direct Anthropic Messages API call -- NOT Base44's InvokeLLM, no
intermediary. Region/crop-based, structured JSON output via forced tool use,
temperature=0 for determinism (the API does not guarantee bit-identical
output even at temperature=0, which is exactly why this provider's result is
never allowed to touch a deterministic fact -- see interface.py).

REQUIRED SECRET: the environment variable named by `API_KEY_ENV_VAR` below
(`GASWATERAI_VISION_API_KEY`) must hold a valid Anthropic API key. It is
deliberately NOT the same variable name as any Claude Code harness
credential -- this is the running PRODUCT's own credential, provisioned
separately for its deployment environment, never reused from a development
session. No such credential is present in this development/build session
(confirmed: no vision-capable API key in this environment; see the
engine's final report), so this provider could not be exercised against a
live endpoint here -- it is implemented and tested against fixtures only
(tests/test_vision_fallback.py mocks the HTTP layer). Wire in
GASWATERAI_VISION_API_KEY at deployment to activate it.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from .interface import VisionFallbackProvider, VisionFallbackResult

API_KEY_ENV_VAR = "GASWATERAI_VISION_API_KEY"
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# Pinned explicitly (never "latest") for reproducibility; overridable per
# deployment via GASWATERAI_VISION_MODEL without a code change.
DEFAULT_MODEL = "claude-sonnet-5-5"
REQUEST_TIMEOUT_SECONDS = 30

_TOOL_SCHEMA = {
    "name": "classify_component",
    "description": "Classify the plumbing component shown in the image crop.",
    "input_schema": {
        "type": "object",
        "properties": {
            "component_type": {
                "type": "string",
                "description": "One label from the provided candidate list, or 'unknown' if none applies.",
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string", "description": "One or two sentences, visual evidence only."},
        },
        "required": ["component_type", "confidence", "reasoning"],
    },
}


class AnthropicVisionProvider(VisionFallbackProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        self.model = model or os.environ.get("GASWATERAI_VISION_MODEL", DEFAULT_MODEL)

    def classify_region(
        self, image_bytes: bytes, image_media_type: str, candidate_labels: list[str], context: dict,
    ) -> VisionFallbackResult:
        if not self.api_key:
            return VisionFallbackResult(
                available=False, component_type=None, confidence=None, reasoning=None,
                provider="anthropic", model=self.model,
                error=f"No API key configured ({API_KEY_ENV_VAR} not set).",
            )

        prompt = (
            "This is a cropped region from a Swiss potable-water plumbing plan. "
            f"Identify which of these component types it shows: {', '.join(candidate_labels)}, or 'unknown'. "
            "Base your answer only on what is visually present in the crop. "
            f"Context (already extracted deterministically by the plan-analysis engine, provided for reference only): {json.dumps(context, ensure_ascii=False)}"
        )
        body = {
            "model": self.model,
            "max_tokens": 300,
            "temperature": 0,
            "tools": [_TOOL_SCHEMA],
            "tool_choice": {"type": "tool", "name": "classify_component"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": image_media_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            return VisionFallbackResult(
                available=False, component_type=None, confidence=None, reasoning=None,
                provider="anthropic", model=self.model, error=f"Request failed: {exc}",
            )

        return self._parse_response(payload, candidate_labels)

    def _parse_response(self, payload: dict, candidate_labels: list[str]) -> VisionFallbackResult:
        try:
            tool_use = next(b for b in payload["content"] if b.get("type") == "tool_use")
            parsed = tool_use["input"]
            component_type = parsed["component_type"]
            if component_type not in candidate_labels and component_type != "unknown":
                # The provider must choose from the closed set given -- an
                # out-of-taxonomy answer is treated as unresolved, never
                # silently accepted as a new class.
                return VisionFallbackResult(
                    available=True, component_type=None, confidence=None,
                    reasoning=f"Model returned out-of-taxonomy label {component_type!r}; discarded.",
                    provider="anthropic", model=self.model, raw_response_id=payload.get("id"),
                )
            return VisionFallbackResult(
                available=True,
                component_type=None if component_type == "unknown" else component_type,
                confidence=float(parsed["confidence"]), reasoning=parsed["reasoning"],
                provider="anthropic", model=self.model, raw_response_id=payload.get("id"),
            )
        except (KeyError, StopIteration, ValueError, TypeError) as exc:
            return VisionFallbackResult(
                available=False, component_type=None, confidence=None, reasoning=None,
                provider="anthropic", model=self.model, error=f"Could not parse response: {exc}",
            )
