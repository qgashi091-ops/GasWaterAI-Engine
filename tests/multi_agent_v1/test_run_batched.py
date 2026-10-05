"""Direct, agent-agnostic tests of BaseAgent.run_batched() -- the shared
transport every model-dependent agent's own *_batch() method delegates
to. Exercises batch splitting, subject_id-based result mapping (never
positional), missing/extra-id handling, whole-chunk error isolation, and
the request-scoped cache -- all independent of any one agent's own
fachliche parsing, which is tested separately per agent."""
from __future__ import annotations

from app.multi_agent_v1.agent_cache import AgentResponseCache
from app.multi_agent_v1.base_agent import BaseAgent, BatchSubjectInput, MAX_IMAGES_PER_BATCH, get_batch_size
from app.multi_agent_v1.provider import AgentModelResponse, FixtureAgentModelProvider

_TOOL_SCHEMA = {
    "name": "do_thing",
    "description": "test tool",
    "input_schema": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    },
}


class _FakeAgent(BaseAgent):
    agent_id = "leitungs_agent"  # any real id works -- run_batched() never constructs an AgentObservation
    claim_types = ("leitung_medium",)


def _subject(sid: str, value: bytes = b"\x89PNG-fake") -> BatchSubjectInput:
    return BatchSubjectInput(subject_id=sid, images=[value], text=f"text for {sid}")


def test_batch_splitting_respects_configured_batch_size():
    agent = _FakeAgent(FixtureAgentModelProvider(responses={"do_thing": {"value": "ok"}}))
    subjects = [_subject(f"s{i}") for i in range(5)]
    outcomes = agent.run_batched(subjects, "do_thing", _TOOL_SCHEMA, "sys", batch_size=2)
    assert len(outcomes) == 5
    assert all(o.available for o in outcomes.values())
    # 5 subjects at batch_size=2 -> 3 provider calls (2, 2, 1)
    assert len(agent.provider.calls) == 3


def test_batch_splitting_never_exceeds_the_hard_image_cap_even_if_batch_size_is_larger():
    agent = _FakeAgent(FixtureAgentModelProvider(responses={"do_thing": {"value": "ok"}}))
    subjects = [_subject(f"s{i}") for i in range(MAX_IMAGES_PER_BATCH + 3)]
    outcomes = agent.run_batched(subjects, "do_thing", _TOOL_SCHEMA, "sys", batch_size=50)
    assert len(outcomes) == MAX_IMAGES_PER_BATCH + 3
    for call in agent.provider.calls:
        assert len(call.images) <= MAX_IMAGES_PER_BATCH


def test_get_batch_size_is_clamped_to_the_hard_image_cap(monkeypatch):
    monkeypatch.setenv("GASWATERAI_BATCH_SIZE_LEITUNGS_AGENT", "999")
    assert get_batch_size("leitungs_agent", 8) == MAX_IMAGES_PER_BATCH


def test_get_batch_size_falls_back_to_default_on_invalid_value(monkeypatch):
    monkeypatch.setenv("GASWATERAI_BATCH_SIZE_LEITUNGS_AGENT", "not-a-number")
    assert get_batch_size("leitungs_agent", 5) == 5


def test_results_are_mapped_by_subject_id_never_by_position():
    """The model is free to return results in any order -- run_batched
    must still attribute each one to the correct subject_id."""
    class _ShufflingProvider(FixtureAgentModelProvider):
        def call(self, request):
            import re
            ids = re.findall(r"=== subject_id: (.*?) ===", request.text)
            results = [{"subject_id": sid, "value": f"answer-for-{sid}"} for sid in reversed(ids)]
            return AgentModelResponse(available=True, tool_input={"results": results}, model=self.model)

    agent = _FakeAgent(_ShufflingProvider())
    subjects = [_subject("alpha"), _subject("beta"), _subject("gamma")]
    outcomes = agent.run_batched(subjects, "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert outcomes["alpha"].tool_input == {"value": "answer-for-alpha"}
    assert outcomes["beta"].tool_input == {"value": "answer-for-beta"}
    assert outcomes["gamma"].tool_input == {"value": "answer-for-gamma"}


def test_subject_missing_from_batch_response_becomes_unavailable():
    class _DroppingProvider(FixtureAgentModelProvider):
        def call(self, request):
            # Only ever answer for "alpha", regardless of who was asked.
            return AgentModelResponse(
                available=True, tool_input={"results": [{"subject_id": "alpha", "value": "ok"}]}, model=self.model,
            )

    agent = _FakeAgent(_DroppingProvider())
    outcomes = agent.run_batched([_subject("alpha"), _subject("beta")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert outcomes["alpha"].available is True
    assert outcomes["beta"].available is False
    assert "missing from batch response" in outcomes["beta"].error


def test_extra_unknown_subject_ids_in_response_are_dropped_not_misassigned():
    class _ExtraIdProvider(FixtureAgentModelProvider):
        def call(self, request):
            return AgentModelResponse(
                available=True,
                tool_input={"results": [
                    {"subject_id": "alpha", "value": "ok"},
                    {"subject_id": "not-asked-about", "value": "should be dropped"},
                ]},
                model=self.model,
            )

    agent = _FakeAgent(_ExtraIdProvider())
    outcomes = agent.run_batched([_subject("alpha")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert set(outcomes.keys()) == {"alpha"}
    assert outcomes["alpha"].tool_input == {"value": "ok"}


def test_whole_chunk_fails_cleanly_when_provider_reports_unavailable():
    """Error isolation: a provider-level failure (timeout, gateway error)
    marks every subject in that one chunk unavailable -- never raises,
    never fabricates a result, never retries."""
    class _FailingProvider(FixtureAgentModelProvider):
        def call(self, request):
            return AgentModelResponse(available=False, tool_input=None, model=self.model, error="simulated timeout")

    agent = _FakeAgent(_FailingProvider())
    outcomes = agent.run_batched([_subject("a"), _subject("b")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert outcomes["a"].available is False
    assert outcomes["a"].error == "simulated timeout"
    assert outcomes["b"].available is False
    assert outcomes["b"].error == "simulated timeout"


def test_malformed_results_field_is_treated_as_unavailable_not_a_crash():
    class _MalformedProvider(FixtureAgentModelProvider):
        def call(self, request):
            return AgentModelResponse(available=True, tool_input={"not_results": []}, model=self.model)

    agent = _FakeAgent(_MalformedProvider())
    outcomes = agent.run_batched([_subject("a")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert outcomes["a"].available is False
    assert "Malformed batch response" in outcomes["a"].error


def test_cache_hit_avoids_a_second_provider_call_for_identical_input():
    cache = AgentResponseCache()
    provider = FixtureAgentModelProvider(responses={"do_thing": {"value": "ok"}})
    agent = _FakeAgent(provider, cache=cache)

    subject = _subject("same-subject", value=b"identical-bytes")
    agent.run_batched([subject], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert len(provider.calls) == 1

    # A second agent instance, SAME cache, SAME inputs -- no new provider call.
    agent2 = _FakeAgent(provider, cache=cache)
    outcomes = agent2.run_batched([subject], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert len(provider.calls) == 1
    assert outcomes["same-subject"].cached is True
    assert outcomes["same-subject"].tool_input == {"value": "ok"}


def test_cache_miss_when_image_bytes_differ():
    cache = AgentResponseCache()
    provider = FixtureAgentModelProvider(responses={"do_thing": {"value": "ok"}})
    agent = _FakeAgent(provider, cache=cache)

    agent.run_batched([_subject("s1", value=b"image-A")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    agent.run_batched([_subject("s1", value=b"image-B")], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert len(provider.calls) == 2


def test_no_cache_means_no_caching_behavior_at_all():
    """Backward compatible default: an agent constructed without a cache
    (every existing direct-agent test, e.g. test_sicherung_category.py)
    behaves exactly as before this epic -- every call reaches the
    provider."""
    provider = FixtureAgentModelProvider(responses={"do_thing": {"value": "ok"}})
    agent = _FakeAgent(provider)  # no cache=
    subject = _subject("same-subject")
    agent.run_batched([subject], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    agent.run_batched([subject], "do_thing", _TOOL_SCHEMA, "sys", batch_size=8)
    assert len(provider.calls) == 2
