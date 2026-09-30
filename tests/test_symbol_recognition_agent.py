"""Fixture-only tests for SymbolRecognitionAgent -- no live network call is
ever made (no GASWATERAI_VISION_API_KEY in this build/dev environment).
Verifies request construction, strict OBSERVATION contract, caching by
model+prompt_hash+crop_hash+text_hash, and the no-credential STOP path.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.microagents.symbol_recognition_agent import AgentObservation, SymbolRecognitionAgent

CANDIDATE_LABELS = ["kueche", "wc_up", "dusche"]


def _fixture_response(component_type="kueche", confidence="supported", ambiguity=None, response_id="msg_fixture_001"):
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
        "id": response_id,
        "content": [{
            "type": "tool_use", "name": "classify_symbol",
            "input": {
                "component_type": component_type, "evidence": ["sink basin outline", "faucet symbol"],
                "ambiguity": ambiguity, "confidence": confidence,
            },
        }],
    }
    return _Resp(payload)


def test_no_api_key_returns_unavailable_observation_never_raises(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key=None)
    obs = agent.recognize("bench-1", b"tight-bytes")
    assert isinstance(obs, AgentObservation)
    assert obs.kind == "OBSERVATION"
    assert obs.available is False
    assert obs.component_type is None
    assert "API key" in obs.error


def test_successful_recognition_image_only(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key", model="claude-sonnet-5-5")
    with patch("urllib.request.urlopen", return_value=_fixture_response("kueche", "supported")):
        obs = agent.recognize("bench-1", b"tight-bytes")
    assert obs.available is True
    assert obs.component_type == "kueche"
    assert obs.confidence == "supported"
    assert obs.condition == "image_only"
    assert obs.cached is False
    assert obs.raw_response_id == "msg_fixture_001"


def test_unknown_is_a_first_class_outcome_not_an_error(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("UNKNOWN", "uncertain")):
        obs = agent.recognize("bench-1", b"tight-bytes")
    assert obs.available is True
    assert obs.component_type is None  # UNKNOWN maps to None, not an error
    assert obs.confidence == "uncertain"


def test_image_plus_text_condition_is_recorded_and_changes_only_the_text(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("wc_up")) as mock_open:
        obs = agent.recognize("bench-2", b"tight-bytes", nearby_text=["WC up", "3 LU"])
    assert obs.condition == "image_plus_text"
    sent_body = json.loads(mock_open.call_args[0][0].data)
    text_blocks = [c["text"] for c in sent_body["messages"][0]["content"] if c["type"] == "text"]
    assert any("WC up" in t for t in text_blocks)


def test_caching_reuses_response_for_identical_inputs(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("dusche")) as mock_open:
        first = agent.recognize("bench-3", b"same-bytes")
        assert first.cached is False
        second = agent.recognize("bench-3", b"same-bytes")
    assert second.cached is True
    assert second.component_type == "dusche"
    mock_open.assert_called_once()  # the second call must not hit the network at all


def test_different_crop_bytes_bypass_the_cache(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("kueche")) as mock_open:
        agent.recognize("bench-4a", b"bytes-a")
        agent.recognize("bench-4b", b"bytes-b")
    assert mock_open.call_count == 2


def test_temperature_zero_and_prompt_is_short(tmp_path):
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", return_value=_fixture_response("kueche")) as mock_open:
        agent.recognize("bench-5", b"tight-bytes")
    sent_body = json.loads(mock_open.call_args[0][0].data)
    assert sent_body["temperature"] == 0
    assert len(sent_body["system"]) < 700  # Phase 3 discipline: extremely small system prompt, no rule text
    assert "chain of thought" not in sent_body["system"].lower()
    assert "svgw" not in sent_body["system"].lower()


def test_network_failure_returns_unavailable_never_raises(tmp_path):
    import urllib.error
    agent = SymbolRecognitionAgent(CANDIDATE_LABELS, cache_dir=tmp_path, api_key="fixture-key")
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
        obs = agent.recognize("bench-6", b"tight-bytes")
    assert obs.available is False
    assert "Request failed" in obs.error


def test_never_receives_full_pdf_or_topology_parameters():
    """Structural guard: the public method signature must not offer a way
    to pass a whole document, page, or topology conclusion -- the agent can
    literally only ever see crop bytes and short text."""
    import inspect
    params = inspect.signature(SymbolRecognitionAgent.recognize).parameters
    forbidden = ("pdf", "document", "page", "topology", "plan_facts", "graph")
    assert not any(f in p.lower() for p in params for f in forbidden)
