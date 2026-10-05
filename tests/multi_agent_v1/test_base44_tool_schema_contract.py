"""Live Base44 contract fix: aiGateway validates `tool_schema` as a raw
JSON Schema whose ROOT must itself be {"type": "object", "properties": {...}}
(confirmed HTTP 400: "tool_schema" muss vom Typ "object" sein
(JSON-Schema-Root)). Every one of the 9 model-dependent agents builds its
tool_schema in the ANTHROPIC tool-definition shape ({"name", "description",
"input_schema": <JSON Schema>}) -- required unchanged for
AnthropicAgentModelProvider's `tools` parameter. The fix lives entirely in
`Base44AgentModelProvider._to_base44_json_schema` (base44_provider.py),
which unwraps `input_schema` before sending. This file verifies, for every
model-dependent agent, that (1) its Anthropic-shaped tool_schema is
unchanged (no agent touched), and (2) what Base44 actually receives after
the unwrap is a valid JSON-Schema-root object with every expected output
field intact -- no agent output field lost, nothing blindly wrapped."""
from __future__ import annotations

import json

import pytest

from app.multi_agent_v1.base44_provider import _to_base44_json_schema

from app.multi_agent_v1.agents import (
    anschluss_agent,
    leitungs_agent,
    planstruktur_agent,
    probenahme_agent,
    schlaufungs_agent,
    sicherungs_agent,
    symbol_agent,
    text_agent,
    zirkulations_hydraulik_agent,
)

# Every model-dependent agent's (module-private) tool schema, built exactly
# as its own agent module builds it -- nothing re-implemented here.
_FIXED_SCHEMA_AGENTS = [
    (anschluss_agent, "_TOOL_SCHEMA"),
    (leitungs_agent, "_TOOL_SCHEMA"),
    (planstruktur_agent, "_TOOL_SCHEMA"),
    (probenahme_agent, "_TOOL_SCHEMA"),
    (schlaufungs_agent, "_TOOL_SCHEMA"),
    (sicherungs_agent, "_TOOL_SCHEMA"),
    (text_agent, "_TOOL_SCHEMA"),
    (zirkulations_hydraulik_agent, "_TOOL_SCHEMA"),
]


def _all_agent_tool_schemas() -> list[tuple[str, dict]]:
    schemas = [
        (module.__name__, getattr(module, attr_name))
        for module, attr_name in _FIXED_SCHEMA_AGENTS
    ]
    # symbol_agent builds its schema dynamically from candidate labels --
    # exercise it with a representative, non-empty candidate list.
    schemas.append((symbol_agent.__name__, symbol_agent._tool_schema(["Absperrklappe", "Wasserzaehler"])))
    return schemas


ALL_AGENT_SCHEMAS = _all_agent_tool_schemas()


def test_exactly_nine_model_dependent_agents_are_covered():
    """Documents and pins down the full set this epic had to check: 9 of
    the 12 agents call a model at all (the other 3 -- nachweis_agent,
    rueckfluss_agent, stagnations_agent -- are purely deterministic and
    build no tool_schema, confirmed by grep: no AgentModelRequest/
    tool_schema reference in those modules)."""
    assert len(ALL_AGENT_SCHEMAS) == 9


@pytest.mark.parametrize("agent_name,wrapper", ALL_AGENT_SCHEMAS, ids=[n for n, _ in ALL_AGENT_SCHEMAS])
def test_agent_tool_schema_wrapper_is_untouched_anthropic_shape(agent_name, wrapper):
    """Requirement 1: tool_schema (as every agent builds it, unchanged) is
    a JSON object, still in the exact Anthropic tool-definition shape
    AnthropicAgentModelProvider needs -- this epic never touched agent
    code, so this must still hold."""
    assert isinstance(wrapper, dict)
    assert "name" in wrapper and "description" in wrapper
    assert isinstance(wrapper["input_schema"], dict)
    assert wrapper["input_schema"]["type"] == "object"
    assert "properties" in wrapper["input_schema"]


@pytest.mark.parametrize("agent_name,wrapper", ALL_AGENT_SCHEMAS, ids=[n for n, _ in ALL_AGENT_SCHEMAS])
def test_base44_receives_a_valid_json_schema_root(agent_name, wrapper):
    """Requirements 2+3: what Base44 actually receives (after the central
    unwrap) has root "type": "object" and "properties" -- exactly what the
    live 400 demanded."""
    sent = _to_base44_json_schema(wrapper)
    assert isinstance(sent, dict)
    assert sent["type"] == "object"
    assert "properties" in sent and isinstance(sent["properties"], dict)


@pytest.mark.parametrize("agent_name,wrapper", ALL_AGENT_SCHEMAS, ids=[n for n, _ in ALL_AGENT_SCHEMAS])
def test_no_output_fields_are_lost_by_the_unwrap(agent_name, wrapper):
    """Requirements 4+6: every property and every required field the agent
    declared survives the transform byte-for-byte -- the unwrap only
    discards the Anthropic wrapper's own "name"/"description" keys, never
    anything inside "input_schema"."""
    sent = _to_base44_json_schema(wrapper)
    input_schema = wrapper["input_schema"]
    assert sent["properties"] == input_schema["properties"]
    assert sent.get("required", []) == input_schema.get("required", [])
    assert sent == input_schema


@pytest.mark.parametrize("agent_name,wrapper", ALL_AGENT_SCHEMAS, ids=[n for n, _ in ALL_AGENT_SCHEMAS])
def test_schema_sent_to_base44_is_response_json_schema_suitable(agent_name, wrapper):
    """Requirement 5: suitable as a Base44 `response_json_schema` -- a bare
    JSON Schema object, round-trips through JSON (no Python-only types),
    and carries none of the Anthropic wrapper's own keys (which Base44's
    JSON-Schema-root validator would reject)."""
    sent = _to_base44_json_schema(wrapper)
    round_tripped = json.loads(json.dumps(sent))
    assert round_tripped == sent
    assert "name" not in sent
    assert "description" not in sent
    assert "input_schema" not in sent


def test_unwrap_is_a_no_op_for_a_schema_without_an_object_rooted_input_schema():
    """Defensive, not agent-specific: a tool_schema that doesn't carry the
    expected `input_schema` (or whose `input_schema` isn't itself rooted
    `type: object`) is passed through unchanged rather than mangled --
    this is the exact shape of the pre-existing test fixture in
    test_base44_provider.py (`{"name": "do_thing", "input_schema": {}}`),
    so that test's behavior stays exactly as it was before this fix."""
    trivial = {"name": "do_thing", "input_schema": {}}
    assert _to_base44_json_schema(trivial) == trivial
