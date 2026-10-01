"""Shared-secret protection for POST /multi_agent_v1/analyze: fails open
when GASWATERAI_MULTI_AGENT_API_KEY is unset (today's default, and every
other test's environment), enforces X-API-Key once it is set, and never
touches /analyze or /check."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import MULTI_AGENT_API_KEY_ENV_VAR, app

FIXTURE = Path(__file__).parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
client = TestClient(app)


def _post_multi_agent():
    with FIXTURE.open("rb") as f:
        return client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})


def test_endpoint_stays_open_when_key_not_configured(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    r = _post_multi_agent()
    assert r.status_code == 200


def test_endpoint_rejects_missing_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    r = _post_multi_agent()
    assert r.status_code == 401


def test_endpoint_rejects_wrong_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    with FIXTURE.open("rb") as f:
        r = client.post(
            "/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")},
            headers={"X-API-Key": "wrong-secret"},
        )
    assert r.status_code == 401


def test_endpoint_accepts_correct_key_once_configured(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    with FIXTURE.open("rb") as f:
        r = client.post(
            "/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")},
            headers={"X-API-Key": "the-real-secret"},
        )
    assert r.status_code == 200


def test_analyze_and_check_are_never_protected(monkeypatch):
    monkeypatch.setenv(MULTI_AGENT_API_KEY_ENV_VAR, "the-real-secret")
    with FIXTURE.open("rb") as f:
        r1 = client.post("/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})
    assert r1.status_code == 200
    with FIXTURE.open("rb") as f:
        r2 = client.post("/check", files={"file": (FIXTURE.name, f, "application/pdf")})
    assert r2.status_code == 200
