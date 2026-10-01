"""Base44AgentModelProvider: a second transport for the EXISTING
AgentModelProvider contract -- no agent/prompt changes, just an alternate
way to reach a model so product operation does not require a separately-
paid Anthropic API key. Tested against a mocked HTTP layer, same discipline
as test_provider.py's AnthropicAgentModelProvider tests (no live Base44
endpoint exists yet)."""
from __future__ import annotations

import json
from unittest import mock

from app.multi_agent_v1.base44_provider import (
    GATEWAY_API_KEY_ENV_VAR,
    GATEWAY_URL_ENV_VAR,
    Base44AgentModelProvider,
)
from app.multi_agent_v1.provider import AgentModelRequest

_REQUEST = AgentModelRequest(
    system_prompt="sys", tool_name="do_thing", tool_schema={"name": "do_thing", "input_schema": {}},
    text="hello", images=[b"fake-png-bytes"], model="claude-test", temperature=0.0, max_tokens=300,
)


def test_unavailable_without_configuration(monkeypatch):
    monkeypatch.delenv(GATEWAY_URL_ENV_VAR, raising=False)
    monkeypatch.delenv(GATEWAY_API_KEY_ENV_VAR, raising=False)
    provider = Base44AgentModelProvider()
    response = provider.call(_REQUEST)
    assert response.available is False
    assert "not configured" in response.error


def test_unavailable_with_only_url_set(monkeypatch):
    monkeypatch.setenv(GATEWAY_URL_ENV_VAR, "https://base44.example/ai-gateway")
    monkeypatch.delenv(GATEWAY_API_KEY_ENV_VAR, raising=False)
    provider = Base44AgentModelProvider()
    assert provider.call(_REQUEST).available is False


def test_sends_the_documented_request_shape_and_bearer_header():
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    fake_payload = {"available": True, "tool_input": {"value": 1}, "model": "claude-test", "raw_response_id": "r1", "error": None}
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps(fake_payload).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp()

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        response = provider.call(_REQUEST)

    assert response.available is True
    assert response.tool_input == {"value": 1}
    assert captured["url"] == "https://base44.example/ai-gateway"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["body"] == {
        "system_prompt": "sys", "tool_name": "do_thing", "tool_schema": {"name": "do_thing", "input_schema": {}},
        "text": "hello", "images": ["ZmFrZS1wbmctYnl0ZXM="], "model": "claude-test",
        "temperature": 0.0, "max_tokens": 300,
    }


def test_parses_unavailable_response_from_gateway():
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    fake_payload = {"available": False, "tool_input": None, "model": "claude-test", "error": "upstream model refused"}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps(fake_payload).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
        response = provider.call(_REQUEST)
    assert response.available is False
    assert response.error == "upstream model refused"


def test_handles_request_failure():
    import urllib.error

    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        response = provider.call(_REQUEST)
    assert response.available is False
    assert "Base44 AI Gateway request failed" in response.error


def test_handles_malformed_response_payload():
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return b"not json"

    with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
        response = provider.call(_REQUEST)
    assert response.available is False
