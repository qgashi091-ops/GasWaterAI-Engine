"""Unit tests for the GASWATERAI_TIMING_DIAGNOSTICS=1 instrumentation
module itself -- enable/disable gating, additive phase timing, RUN/SKIP
agent scopes, model-call status classification, and the no-content
guarantee of the logged payload shape. See
test_timing_diagnostics_pipeline.py for the full pipeline/API-level
behavior this module is wired into."""
from __future__ import annotations

import json
import logging
import time

from app.multi_agent_v1 import timing_diagnostics as td
from app.multi_agent_v1.provider import AgentModelResponse


def test_disabled_by_default_and_every_helper_is_a_safe_no_op(monkeypatch):
    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    assert td.enabled() is False

    token = td.start_request()
    try:
        assert td.current() is None
        with td.phase("router"):
            pass
        with td.agent_scope("symbol_agent", True) as timing:
            assert timing is None
        td.record_model_call("symbol_agent", duration_ms=1.0, images=1, status="SUCCESS")
        td.finish_and_log()  # must not raise, must not log (checked below)
    finally:
        td.end_request(token)


def test_disabled_logs_nothing(monkeypatch, caplog):
    monkeypatch.delenv(td.TIMING_DIAGNOSTICS_ENV_VAR, raising=False)
    token = td.start_request()
    try:
        with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
            with td.phase("router"):
                pass
            td.finish_and_log()
    finally:
        td.end_request(token)
    assert caplog.records == []


def test_enabled_logs_exactly_one_structured_line(monkeypatch, caplog):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    token = td.start_request()
    try:
        with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
            with td.phase("router"):
                time.sleep(0.001)
            with td.agent_scope("symbol_agent", True):
                td.record_model_call("symbol_agent", duration_ms=12.5, images=2, status="SUCCESS")
            with td.agent_scope("stagnations_agent", False):
                pass
            td.finish_and_log()
    finally:
        td.end_request(token)

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("multi_agent_v1_timing ")
    payload = json.loads(message[len("multi_agent_v1_timing "):])

    assert payload["execution_mode"] == "serial"
    assert isinstance(payload["correlation_id"], str) and payload["correlation_id"]
    assert payload["phases_ms"]["router"] >= 0

    by_id = {a["agent_id"]: a for a in payload["agents"]}
    assert by_id["symbol_agent"]["run"] is True
    assert by_id["symbol_agent"]["model_calls"] == 1
    assert by_id["symbol_agent"]["images"] == 2
    assert by_id["symbol_agent"]["provider_duration_ms_per_call"] == [12.5]
    assert by_id["symbol_agent"]["status"] == "SUCCESS"

    assert by_id["stagnations_agent"]["run"] is False
    assert by_id["stagnations_agent"]["status"] == "SKIP"
    assert by_id["stagnations_agent"]["model_calls"] == 0


def test_phase_timing_is_additive_across_multiple_with_blocks(monkeypatch):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    token = td.start_request()
    try:
        recorder = td.current()
        with td.phase("deterministic_preprocessing"):
            time.sleep(0.001)
        with td.phase("deterministic_preprocessing"):
            time.sleep(0.001)
        assert recorder.phases_ms["deterministic_preprocessing"] > 0
        # two sleeps accumulated into the SAME key, not overwritten
        first_total = recorder.phases_ms["deterministic_preprocessing"]
        with td.phase("deterministic_preprocessing"):
            time.sleep(0.001)
        assert recorder.phases_ms["deterministic_preprocessing"] > first_total
    finally:
        td.end_request(token)


def test_agent_status_is_error_then_timeout_takes_precedence_over_success(monkeypatch):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    token = td.start_request()
    try:
        with td.agent_scope("leitungs_agent", True):
            td.record_model_call("leitungs_agent", duration_ms=1.0, images=0, status="SUCCESS")
            td.record_model_call("leitungs_agent", duration_ms=1.0, images=0, status="ERROR")
        recorder = td.current()
        assert recorder.agents["leitungs_agent"].to_dict()["status"] == "ERROR"

        with td.agent_scope("text_agent", True):
            td.record_model_call("text_agent", duration_ms=1.0, images=0, status="ERROR")
            td.record_model_call("text_agent", duration_ms=1.0, images=0, status="TIMEOUT")
        assert recorder.agents["text_agent"].to_dict()["status"] == "TIMEOUT"
    finally:
        td.end_request(token)


def test_classify_status_success():
    response = AgentModelResponse(available=True, tool_input={}, model="m")
    assert td.classify_status(response) == "SUCCESS"


def test_classify_status_timeout_from_error_text():
    response = AgentModelResponse(available=False, tool_input=None, model="m", error="Base44 AI Gateway request failed: <urlopen error timed out>")
    assert td.classify_status(response) == "TIMEOUT"


def test_classify_status_error_for_anything_else():
    response = AgentModelResponse(available=False, tool_input=None, model="m", error="Invalid file attachment")
    assert td.classify_status(response) == "ERROR"


def test_logged_payload_never_carries_subject_or_content_fields(monkeypatch, caplog):
    """The instrumentation only ever receives agent_id, a byte COUNT, and a
    duration -- subject_id, prompts, and image bytes never reach this
    module at all (see base_agent.py's call_model -- it passes
    `len(request.images)`, never `request.images` itself, and never
    `request.text`/`request.system_prompt`). This locks in that no logged
    payload key could ever carry plan-derived content."""
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    token = td.start_request()
    try:
        with caplog.at_level(logging.INFO, logger="gaswaterai.timing_diagnostics"):
            with td.agent_scope("symbol_agent", True):
                td.record_model_call("symbol_agent", duration_ms=1.0, images=1, status="SUCCESS")
            td.finish_and_log()
    finally:
        td.end_request(token)

    payload = json.loads(caplog.records[0].message[len("multi_agent_v1_timing "):])
    allowed_top_level = {"correlation_id", "execution_mode", "total_duration_ms", "phases_ms", "agents"}
    assert set(payload.keys()) == allowed_top_level
    allowed_agent_keys = {
        "agent_id", "run", "start_ms", "end_ms", "duration_ms",
        "model_calls", "images", "provider_duration_ms_per_call", "status",
    }
    for agent_entry in payload["agents"]:
        assert set(agent_entry.keys()) == allowed_agent_keys
