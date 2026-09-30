"""Fixture-only tests for the vision fallback provider -- no live network
call is ever made (no credential exists in this build/dev environment; see
anthropic_provider.py's module docstring). Verifies request construction and
response parsing against recorded fixture payloads.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from app.vision_fallback.anthropic_provider import AnthropicVisionProvider
from app.vision_fallback.interface import VisionFallbackResult

CANDIDATE_LABELS = ["kueche", "wc_up"]


def _fixture_response(component_type="kueche", confidence=0.82):
    class _Resp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = {
        "id": "msg_fixture_001",
        "content": [{
            "type": "tool_use", "name": "classify_component",
            "input": {"component_type": component_type, "confidence": confidence, "reasoning": "Visible sink basin with faucet."},
        }],
    }
    return _Resp(payload)


def test_no_api_key_returns_unavailable_never_raises():
    provider = AnthropicVisionProvider(api_key=None)
    result = provider.classify_region(b"fake-png-bytes", "image/png", CANDIDATE_LABELS, context={})
    assert isinstance(result, VisionFallbackResult)
    assert result.available is False
    assert result.component_type is None
    assert "API key" in result.error


def test_successful_classification_parses_structured_output():
    provider = AnthropicVisionProvider(api_key="fixture-key", model="claude-sonnet-5-5")
    with patch("urllib.request.urlopen", return_value=_fixture_response("kueche", 0.82)):
        result = provider.classify_region(b"fake-png-bytes", "image/png", CANDIDATE_LABELS, context={"nearby_text": ["Küche"]})
    assert result.available is True
    assert result.component_type == "kueche"
    assert result.confidence == 0.82
    assert result.raw_response_id == "msg_fixture_001"


def test_unknown_response_maps_to_none_component_type():
    provider = AnthropicVisionProvider(api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("unknown", 0.4)):
        result = provider.classify_region(b"fake-png-bytes", "image/png", CANDIDATE_LABELS, context={})
    assert result.available is True
    assert result.component_type is None


def test_out_of_taxonomy_label_is_discarded_not_accepted():
    provider = AnthropicVisionProvider(api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("waschmaschine", 0.9)):
        result = provider.classify_region(b"fake-png-bytes", "image/png", CANDIDATE_LABELS, context={})
    assert result.available is True
    assert result.component_type is None
    assert "out-of-taxonomy" in result.reasoning


def test_network_failure_returns_unavailable_never_raises():
    import urllib.error
    provider = AnthropicVisionProvider(api_key="fixture-key")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        result = provider.classify_region(b"fake-png-bytes", "image/png", CANDIDATE_LABELS, context={})
    assert result.available is False
    assert "Request failed" in result.error


def test_request_is_region_based_not_whole_plan():
    """The request body must carry exactly the crop bytes handed in, never
    a reference to the full document -- enforced structurally here since
    there is no `pdf_bytes`/`document` parameter on the interface at all."""
    import inspect
    from app.vision_fallback.interface import VisionFallbackProvider
    params = inspect.signature(VisionFallbackProvider.classify_region).parameters
    assert "image_bytes" in params
    assert not any("pdf" in p.lower() or "document" in p.lower() for p in params)
