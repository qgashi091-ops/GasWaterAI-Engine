"""POST /diagnostics/base44-gateway: TEMPORARY probe for the live Render ->
Base44 403. Covers exactly what this epic requires: no diagnosis runs
without a valid X-API-Key, secret values never leak into the response, a
second request only ever happens when the first came back 403, and never
more than two requests reach Base44 in total."""
from __future__ import annotations

import json
from unittest import mock

from fastapi.testclient import TestClient

from app.main import MULTI_AGENT_API_KEY_ENV_VAR, app
from app.multi_agent_v1.base44_provider import GATEWAY_API_KEY_ENV_VAR, GATEWAY_URL_ENV_VAR

client = TestClient(app)

GATEWAY_URL = "https://gaswaterai.base44.app/functions/aiGateway"
GATEWAY_SECRET = "live-secret-value"
MULTI_AGENT_KEY = "the-real-secret"


class _FakeResp:
    def __init__(self, status: int, body: bytes, headers: dict | None = None):
        self.status = status
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def geturl(self):
        return GATEWAY_URL

    def read(self):
        return self._body


def _configure(monkeypatch, multi_agent_key: str | None = MULTI_AGENT_KEY):
    if multi_agent_key is None:
        monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, multi_agent_key)
    monkeypatch.setenv(GATEWAY_URL_ENV_VAR, GATEWAY_URL)
    monkeypatch.setenv(GATEWAY_API_KEY_ENV_VAR, GATEWAY_SECRET)


def _post(headers: dict | None = None):
    return client.post("/diagnostics/base44-gateway", headers=headers or {})


def test_endpoint_rejects_missing_key_once_configured(monkeypatch):
    _configure(monkeypatch)
    r = _post()
    assert r.status_code == 401


def test_endpoint_refuses_without_any_key_configured(monkeypatch):
    """Unlike /multi_agent_v1/analyze, this diagnostic endpoint must never
    fail open: no GASWATERAI_MULTI_AGENT_API_KEY configured at all still
    means no diagnosis runs."""
    _configure(monkeypatch, multi_agent_key=None)
    r = _post()
    assert r.status_code != 200


def test_endpoint_rejects_wrong_key_once_configured(monkeypatch):
    _configure(monkeypatch)
    r = _post(headers={"X-API-Key": "wrong-secret"})
    assert r.status_code == 401


def test_secret_values_never_appear_in_response(monkeypatch):
    _configure(monkeypatch)
    body = json.dumps({"available": True, "tool_input": {}, "model": "m"}).encode("utf-8")

    def _fake_urlopen(req, timeout=None):
        return _FakeResp(200, body)

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = _post(headers={"X-API-Key": MULTI_AGENT_KEY})

    assert r.status_code == 200
    raw = r.text
    assert GATEWAY_SECRET not in raw
    assert MULTI_AGENT_KEY not in raw


def test_request_2_only_sent_when_request_1_is_403(monkeypatch):
    _configure(monkeypatch)
    calls = []

    def _fake_urlopen(req, timeout=None):
        calls.append(dict(req.header_items()))
        return _FakeResp(200, b'{"available": true, "tool_input": {}, "model": "m"}')

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = _post(headers={"X-API-Key": MULTI_AGENT_KEY})

    assert r.status_code == 200
    assert len(calls) == 1
    data = r.json()
    assert data["request_2"] is None
    assert data["request_2_skipped_reason"] is not None


def test_request_2_sent_and_capped_at_two_when_request_1_is_403(monkeypatch):
    _configure(monkeypatch)
    calls = []

    def _fake_urlopen(req, timeout=None):
        calls.append(dict(req.header_items()))
        import urllib.error
        raise urllib.error.HTTPError(
            GATEWAY_URL, 403, "Forbidden", {"Content-Type": "text/html"}, None,
        )

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = _post(headers={"X-API-Key": MULTI_AGENT_KEY})

    assert r.status_code == 200
    assert len(calls) == 2
    data = r.json()
    assert data["request_1"]["status"] == 403
    assert data["request_2"]["status"] == 403
    assert data["request_2"]["user_agent"] != data["request_1"]["user_agent"]


def test_request_1_status_200_never_triggers_request_2(monkeypatch):
    _configure(monkeypatch)
    call_count = {"n": 0}

    def _fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        return _FakeResp(200, b'{"available": false, "tool_input": null, "model": "m", "error": "none"}')

    with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        r = _post(headers={"X-API-Key": MULTI_AGENT_KEY})

    assert r.status_code == 200
    assert call_count["n"] == 1
