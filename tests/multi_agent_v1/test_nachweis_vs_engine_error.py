"""Nachweis vs Enginefehler: ENGINE_ERROR, PARSER_ERROR and RECOGNITION_GAP
must never be reported as a customer deficiency (Kundenmangel)."""
from __future__ import annotations

from app.multi_agent_v1.agents.nachweis_agent import NachweisAgent, classify_gap_reason
from app.multi_agent_v1.provider import FixtureAgentModelProvider


def test_missing_credential_is_engine_error():
    assert classify_gap_reason("No API key configured (GASWATERAI_VISION_API_KEY not set).") == "ENGINE_ERROR"


def test_http_failure_is_engine_error():
    assert classify_gap_reason("Request failed: <urlopen error timed out>") == "ENGINE_ERROR"


def test_unparseable_response_is_engine_error():
    assert classify_gap_reason("Could not parse response: 'component_type'") == "ENGINE_ERROR"


def test_reconstructed_bridge_geometry_is_parser_error():
    assert classify_gap_reason("terminal node is adjacent to a reconstructed bridge edge") == "PARSER_ERROR"


def test_missing_rule_corpus_is_recognition_gap_not_customer_fault():
    assert classify_gap_reason("no existing rule in app/rules/water_rules.py defines the required device for it") == "RECOGNITION_GAP"
    assert classify_gap_reason("no_explicit_category_evidence") == "RECOGNITION_GAP"


def test_genuine_content_absence_is_customer_deficiency():
    assert classify_gap_reason("the plan shows no sampling point anywhere near the distributor") == "CUSTOMER_DEFICIENCY"


def test_nachweis_agent_wraps_classification_in_an_observation():
    agent = NachweisAgent(FixtureAgentModelProvider())
    obs = agent.classify_gap(context=None, subject_id="INV1:sicherungseinrichtung", reason="No API key configured")
    assert obs.value["gap_type"] == "ENGINE_ERROR"
    assert obs.agent_id == "nachweis_agent"
    assert obs.claim_type == "nachweis_gap"
