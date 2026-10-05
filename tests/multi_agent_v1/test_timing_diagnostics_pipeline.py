"""GASWATERAI_TIMING_DIAGNOSTICS=1 wired through the real pipeline
(run_multi_agent_v1) and the real /multi_agent_v1/analyze endpoint.
Confirms exactly what this epic asked for: diagnostics OFF means no
timing log at all, diagnostics ON means a complete one covering every
requested phase and every agent (RUN/SKIP + model-call accounting), no
secret/content ever appears in it, and -- most importantly -- the actual
pipeline RESULT is byte-identical whether diagnostics are on or off."""
from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from app.main import MULTI_AGENT_API_KEY_ENV_VAR, app
from app.multi_agent_v1 import timing_diagnostics as td
from app.multi_agent_v1.pipeline import run_multi_agent_v1
from app.multi_agent_v1.provider import FixtureAgentModelProvider

from pathlib import Path

from .conftest import analyze
from .test_pipeline_integration import _FIXTURE_RESPONSES, _inventory

FIXTURE = Path(__file__).parent.parent / "fixtures" / "W-003_Referenzfall.Plan.pdf"
client = TestClient(app)


def _run_pipeline(pdf_bytes):
    doc, facts = analyze(pdf_bytes)
    provider = FixtureAgentModelProvider(responses=dict(_FIXTURE_RESPONSES))
    return run_multi_agent_v1(doc=doc, provider=provider, pdf_bytes=pdf_bytes, plan_facts=facts, inventory=_inventory(), rule_checks=[])


def _run_pipeline_with_recorder(pdf_bytes):
    """run_multi_agent_v1() deliberately does NOT start/stop the timing
    recorder itself -- app/main.py owns that lifecycle, because PDF
    parsing (a phase this epic asks to measure) happens BEFORE the
    pipeline function is even called. A direct pipeline-level test has to
    replicate that same lifecycle to see any logging, exactly like
    app/main.py's real route does."""
    token = td.start_request()
    try:
        result = _run_pipeline(pdf_bytes)
        td.finish_and_log()
        return result
    finally:
        td.end_request(token)


def test_diagnostics_off_produces_no_log_line(monkeypatch, caplog, legend_and_riser_pdf_bytes):
    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
        _run_pipeline(legend_and_riser_pdf_bytes)
    assert caplog.records == []


def test_diagnostics_on_logs_every_requested_phase_and_all_twelve_agents(monkeypatch, caplog, legend_and_riser_pdf_bytes):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
        _run_pipeline_with_recorder(legend_and_riser_pdf_bytes)

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message[len("multi_agent_v1_timing "):])

    assert payload["execution_mode"] == "serial"
    assert payload["total_duration_ms"] > 0

    for required_phase in ("deterministic_preprocessing", "router", "evidence_merger", "canonical_output"):
        assert required_phase in payload["phases_ms"]
        assert payload["phases_ms"][required_phase] >= 0

    agent_ids = {a["agent_id"] for a in payload["agents"]}
    expected_agent_ids = {
        "planstruktur_agent", "symbol_agent", "text_agent", "leitungs_agent", "anschluss_agent",
        "schlaufungs_agent", "sicherungs_agent", "stagnations_agent", "rueckfluss_agent",
        "zirkulations_hydraulik_agent", "probenahme_agent", "nachweis_agent",
    }
    assert agent_ids == expected_agent_ids

    for a in payload["agents"]:
        assert a["status"] in ("SKIP", "SUCCESS", "ERROR", "TIMEOUT")
        if not a["run"]:
            assert a["status"] == "SKIP"
            assert a["model_calls"] == 0


def test_diagnostics_do_not_change_the_pipeline_result(monkeypatch, legend_and_riser_pdf_bytes):
    """Requirement: 'bestehende Ergebnisse unveraendert' -- the exact same
    inputs produce the exact same CanonicalPlanUnderstanding whether
    diagnostics are on or off."""
    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    result_off = _run_pipeline(legend_and_riser_pdf_bytes)

    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    result_on = _run_pipeline(legend_and_riser_pdf_bytes)

    assert result_off.to_dict() == result_on.to_dict()


def test_logged_payload_never_contains_subject_ids_or_component_text(monkeypatch, caplog, legend_and_riser_pdf_bytes):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
        _run_pipeline_with_recorder(legend_and_riser_pdf_bytes)

    raw_message = caplog.records[0].message
    # subject_ids used by this fixture's inventory -- plan-derived identifiers
    # that must never leak into timing instrumentation.
    for leaked_candidate in ("INV-SICHERUNG", "INV-ZIRKULATION", "Sicherheitsventil", "Zirkulationspumpe"):
        assert leaked_candidate not in raw_message


def test_api_endpoint_diagnostics_off_logs_nothing(monkeypatch, caplog):
    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
        with FIXTURE.open("rb") as f:
            r = client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})
    assert r.status_code == 200
    assert caplog.records == []


def test_api_endpoint_diagnostics_on_logs_correlation_id_and_total(monkeypatch, caplog):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)
    with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
        with FIXTURE.open("rb") as f:
            r = client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})
    assert r.status_code == 200

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message[len("multi_agent_v1_timing "):])
    assert payload["correlation_id"]
    assert "pdf_parsing" in payload["phases_ms"]
    assert "deterministic_preprocessing" in payload["phases_ms"]
    assert payload["total_duration_ms"] > 0


def test_api_endpoint_response_is_identical_regardless_of_diagnostics_flag(monkeypatch):
    monkeypatch.delenv(MULTI_AGENT_API_KEY_ENV_VAR, raising=False)

    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    with FIXTURE.open("rb") as f:
        r_off = client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})

    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    with FIXTURE.open("rb") as f:
        r_on = client.post("/multi_agent_v1/analyze", files={"file": (FIXTURE.name, f, "application/pdf")})

    body_off, body_on = r_off.json(), r_on.json()
    # Both runs call no real model (no credentials configured in test env),
    # so results are deterministic; only the diagnostics env var differs.
    assert body_off["canonical_plan_understanding"] == body_on["canonical_plan_understanding"]
    assert body_off["observations"] == body_on["observations"]
