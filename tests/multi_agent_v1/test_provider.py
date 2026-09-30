"""AgentModelProvider abstraction: both concrete providers honor the same
contract, and the Anthropic provider is fully testable via a mocked HTTP
layer -- no live credential needed or used (none exists in this
environment; see app/vision_fallback/anthropic_provider.py's own docstring
for why that is expected, not a bug)."""
from __future__ import annotations

import json
from unittest import mock

from app.multi_agent_v1.provider import (
    AgentModelRequest,
    AnthropicAgentModelProvider,
    FixtureAgentModelProvider,
)

_REQUEST = AgentModelRequest(
    system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
    text="hello", images=[b"fake-png-bytes"],
)


def test_fixture_provider_returns_fixed_response_and_logs_call():
    provider = FixtureAgentModelProvider(responses={"do_thing": {"value": 42}})
    response = provider.call(_REQUEST)
    assert response.available is True
    assert response.tool_input == {"value": 42}
    assert len(provider.calls) == 1


def test_fixture_provider_unavailable_when_no_handler_registered():
    provider = FixtureAgentModelProvider(responses={})
    response = provider.call(_REQUEST)
    assert response.available is False
    assert response.error is not None


def test_fixture_provider_supports_callable_handler_for_per_call_logic():
    def handler(request: AgentModelRequest):
        return {"echo": request.text}

    provider = FixtureAgentModelProvider(responses={"do_thing": handler})
    response = provider.call(_REQUEST)
    assert response.tool_input == {"echo": "hello"}


def test_fixture_provider_simulates_unavailable_when_handler_returns_none():
    provider = FixtureAgentModelProvider(responses={"do_thing": lambda r: None})
    response = provider.call(_REQUEST)
    assert response.available is False


def test_anthropic_provider_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("GASWATERAI_VISION_API_KEY", raising=False)
    provider = AnthropicAgentModelProvider(api_key=None)
    response = provider.call(_REQUEST)
    assert response.available is False
    assert "No API key" in response.error


def test_anthropic_provider_parses_mocked_http_response():
    provider = AnthropicAgentModelProvider(api_key="fake-test-key", model="claude-test")
    fake_payload = {"id": "resp_1", "content": [{"type": "tool_use", "input": {"value": 7}}]}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps(fake_payload).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
        response = provider.call(_REQUEST)
    assert response.available is True
    assert response.tool_input == {"value": 7}
    assert response.raw_response_id == "resp_1"


def test_anthropic_provider_handles_request_failure():
    import urllib.error

    provider = AnthropicAgentModelProvider(api_key="fake-test-key")
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        response = provider.call(_REQUEST)
    assert response.available is False
    assert "Request failed" in response.error
