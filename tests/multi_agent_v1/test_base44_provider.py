"""Base44AgentModelProvider: a second transport for the EXISTING
AgentModelProvider contract -- no agent/prompt changes, just an alternate
way to reach a model so product operation does not require a separately-
paid Anthropic API key. Tested against a mocked HTTP layer, same discipline
as test_provider.py's AnthropicAgentModelProvider tests (no live Base44
endpoint exists yet)."""
from __future__ import annotations

import base64
import io
import json
from unittest import mock

from app.multi_agent_v1.base44_provider import (
    GATEWAY_API_KEY_ENV_VAR,
    GATEWAY_URL_ENV_VAR,
    REQUEST_USER_AGENT,
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


def test_sends_the_documented_request_shape_and_gateway_secret_header():
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
    assert captured["headers"]["X-gateway-secret"] == "secret-key"
    assert "Authorization" not in captured["headers"]
    assert captured["body"] == {
        "system_prompt": "sys", "tool_name": "do_thing", "tool_schema": {"name": "do_thing", "input_schema": {}},
        "text": "hello", "images": ["ZmFrZS1wbmctYnl0ZXM="], "model": "claude-test",
        "temperature": 0.0, "max_tokens": 300,
    }


def test_base44_gateway_api_key_is_sent_exactly_as_x_gateway_secret_header(monkeypatch):
    """Exact, narrow regression test for the live-confirmed Base44 contract:
    BASE44_AI_GATEWAY_API_KEY -> the `x-gateway-secret` header, verbatim,
    with no Bearer/Authorization wrapping. This is the root cause the live
    403 ("Base44 AI Gateway request failed: HTTP Error 403: Forbidden")
    traced back to -- the provider previously sent
    `Authorization: Bearer <key>` instead."""
    monkeypatch.setenv(GATEWAY_URL_ENV_VAR, "https://gaswaterai.base44.app/functions/aiGateway")
    monkeypatch.setenv(GATEWAY_API_KEY_ENV_VAR, "live-secret-value")
    provider = Base44AgentModelProvider()
    captured_headers = {}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps({"available": True, "tool_input": {}, "model": "m"}).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        captured_headers.update(req.header_items())
        return _FakeResp()

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(_REQUEST)

    assert captured_headers.get("X-gateway-secret") == "live-secret-value"


def test_sends_a_fixed_non_default_user_agent_not_python_urllib():
    """Live-confirmed root cause of the Render -> Base44 403: Cloudflare's
    edge blocks the Python urllib default User-Agent ("Python-urllib/3.x")
    with HTTP 403 / error 1010 before the request reaches aiGateway at all.
    A plain, static User-Agent gets through (confirmed: a 401 "invalid
    shared secret" response FROM Base44 itself). x-gateway-secret must still
    be sent exactly as before -- this changes only the User-Agent header."""
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    captured_headers = {}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self):
            return json.dumps({"available": True, "tool_input": {}, "model": "m"}).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        captured_headers.update(req.header_items())
        return _FakeResp()

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        provider.call(_REQUEST)

    sent_user_agent = captured_headers.get("User-agent")
    assert sent_user_agent == REQUEST_USER_AGENT
    assert sent_user_agent is not None
    assert not sent_user_agent.lower().startswith("python-urllib")
    assert captured_headers.get("X-gateway-secret") == "secret-key"


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


def test_http_error_captures_status_content_type_and_safe_body():
    """Live incident, next step: after the User-Agent fix, Render's real
    /multi_agent_v1/analyze run got HTTP 400 from Base44 instead of 403.
    The provider must surface Base44's actual error body so the 400 can be
    diagnosed, not just the bare status code."""
    import urllib.error

    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    body = json.dumps({"available": False, "error": "invalid tool_schema: missing 'name'"}).encode("utf-8")

    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://base44.example/ai-gateway", 400, "Bad Request",
            {"Content-Type": "application/json"}, io.BytesIO(body),
        )

    with mock.patch("urllib.request.urlopen", side_effect=_raise):
        response = provider.call(_REQUEST)

    assert response.available is False
    assert "HTTP 400" in response.error
    assert "application/json" in response.error
    assert "invalid tool_schema" in response.error


def test_http_error_body_never_leaks_gateway_secret():
    import urllib.error

    secret = "super-secret-live-value"
    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key=secret)
    body = f'{{"error": "rejected request signed with {secret}"}}'.encode("utf-8")

    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://base44.example/ai-gateway", 400, "Bad Request",
            {"Content-Type": "application/json"}, io.BytesIO(body),
        )

    with mock.patch("urllib.request.urlopen", side_effect=_raise):
        response = provider.call(_REQUEST)

    assert secret not in response.error
    assert "[REDACTED]" in response.error


def test_http_error_body_strips_base64_image_payloads():
    import urllib.error

    provider = Base44AgentModelProvider(gateway_url="https://base44.example/ai-gateway", api_key="secret-key")
    fake_image_b64 = base64.b64encode(b"fake-png-bytes" * 20).decode("ascii")
    assert len(fake_image_b64) > 100
    body = json.dumps({"error": f"could not decode images[0]: {fake_image_b64}"}).encode("utf-8")

    def _raise(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://base44.example/ai-gateway", 400, "Bad Request",
            {"Content-Type": "application/json"}, io.BytesIO(body),
        )

    with mock.patch("urllib.request.urlopen", side_effect=_raise):
        response = provider.call(_REQUEST)

    assert fake_image_b64 not in response.error
    assert "[IMAGE_DATA_REDACTED]" in response.error


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
