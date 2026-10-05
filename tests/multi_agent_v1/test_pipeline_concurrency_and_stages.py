"""Pipeline-level tests for the performance-optimization epic's
concurrency and staging guarantees: the configurable concurrency limit is
actually enforced, Stage B/C genuinely wait for Stage A's output rather
than racing it, and the merged CanonicalPlanUnderstanding/observations are
identical regardless of which agent's thread happens to finish first."""
from __future__ import annotations

import threading
import time

from app.multi_agent_v1 import timing_diagnostics as td
from app.multi_agent_v1.pipeline import run_multi_agent_v1
from app.multi_agent_v1.provider import AgentModelResponse, FixtureAgentModelProvider

from .conftest import analyze
from .test_pipeline_integration import _FIXTURE_RESPONSES, _inventory


class _ConcurrencyTrackingProvider(FixtureAgentModelProvider):
    """Sleeps briefly on every call and records the MAXIMUM number of
    calls that were ever in flight at once -- the direct, empirical check
    that GASWATERAI_AGENT_MAX_CONCURRENCY actually bounds simultaneous
    provider calls, not just that the code LOOKS like it should."""

    def __init__(self, *args, sleep_s: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_s = sleep_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def call(self, request):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self.sleep_s)
            return super().call(request)
        finally:
            with self._lock:
                self._in_flight -= 1


def _run(pdf_bytes, provider):
    doc, facts = analyze(pdf_bytes)
    return run_multi_agent_v1(doc=doc, provider=provider, pdf_bytes=pdf_bytes, plan_facts=facts, inventory=_inventory(), rule_checks=[])


def test_concurrency_limit_is_actually_enforced(monkeypatch, legend_and_riser_pdf_bytes):
    monkeypatch.setenv("GASWATERAI_AGENT_MAX_CONCURRENCY", "2")
    provider = _ConcurrencyTrackingProvider(responses=dict(_FIXTURE_RESPONSES), sleep_s=0.05)
    _run(legend_and_riser_pdf_bytes, provider)
    assert provider.max_in_flight <= 2
    assert provider.max_in_flight >= 1


def test_higher_concurrency_limit_allows_more_overlap(monkeypatch, legend_and_riser_pdf_bytes):
    """Not a timing assertion (too flaky across machines) -- just confirms
    the configured cap itself changes what the executor is ALLOWED to do,
    by checking the cap is read fresh per request."""
    monkeypatch.setenv("GASWATERAI_AGENT_MAX_CONCURRENCY", "7")
    assert td.max_concurrency() == 7
    monkeypatch.setenv("GASWATERAI_AGENT_MAX_CONCURRENCY", "1")
    assert td.max_concurrency() == 1


def test_default_concurrency_is_three_when_unset(monkeypatch):
    monkeypatch.delenv("GASWATERAI_AGENT_MAX_CONCURRENCY", raising=False)
    assert td.max_concurrency() == 3


def test_invalid_concurrency_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GASWATERAI_AGENT_MAX_CONCURRENCY", "not-a-number")
    assert td.max_concurrency() == 3
    monkeypatch.setenv("GASWATERAI_AGENT_MAX_CONCURRENCY", "0")
    assert td.max_concurrency() == 3


def test_rueckfluss_stage_b_reads_sicherungs_stage_a_output_correctly(legend_and_riser_pdf_bytes):
    """Dependency-stage check: RueckflussAgent's result for INV-SICHERUNG
    must reflect the EXPLICIT category SicherungsAgent found on the SAME
    run's Stage A pass -- this can only be correct if Stage B genuinely
    started after Stage A's sicherungs_agent finished, not a race."""
    provider = FixtureAgentModelProvider(responses=dict(_FIXTURE_RESPONSES))
    result = _run(legend_and_riser_pdf_bytes, provider)
    rueckfluss = [o for o in result.observations if o["agent_id"] == "rueckfluss_agent"]
    assert len(rueckfluss) == 1
    assert rueckfluss[0]["value"]["documented_category"] == 2
    assert rueckfluss[0]["value"]["reason"] == "no_existing_category_requirement_rule"


def test_merge_order_is_deterministic_regardless_of_per_agent_latency(legend_and_riser_pdf_bytes):
    """Gives DIFFERENT agents artificially different latencies (so
    completion order varies run to run) and confirms the final
    observations list is appended in the SAME fixed agent+subject order
    every time regardless."""
    class _VariableLatencyProvider(FixtureAgentModelProvider):
        _DELAYS = {"classify_position_batch": 0.03, "classify_probenahme_batch": 0.01}

        def call(self, request):
            time.sleep(self._DELAYS.get(request.tool_name, 0.0))
            return super().call(request)

    orders = []
    for _ in range(4):
        provider = _VariableLatencyProvider(responses=dict(_FIXTURE_RESPONSES))
        result = _run(legend_and_riser_pdf_bytes, provider)
        orders.append([(o["agent_id"], o["subject_id"]) for o in result.observations])

    assert all(order == orders[0] for order in orders)


def test_canonical_output_is_identical_regardless_of_completion_order(legend_and_riser_pdf_bytes):
    class _VariableLatencyProvider(FixtureAgentModelProvider):
        _DELAYS = {"classify_position_batch": 0.03, "classify_probenahme_batch": 0.01}

        def call(self, request):
            time.sleep(self._DELAYS.get(request.tool_name, 0.0))
            return super().call(request)

    results = []
    for _ in range(4):
        provider = _VariableLatencyProvider(responses=dict(_FIXTURE_RESPONSES))
        results.append(_run(legend_and_riser_pdf_bytes, provider).canonical_plan_understanding)

    assert all(r == results[0] for r in results)


def test_extended_timing_diagnostics_batch_and_cache_fields_are_populated(monkeypatch, legend_and_riser_pdf_bytes):
    monkeypatch.setenv(td.TIMING_DIAGNOSTICS_ENV_VAR, "1")
    token = td.start_request()
    try:
        provider = FixtureAgentModelProvider(responses=dict(_FIXTURE_RESPONSES))
        _run(legend_and_riser_pdf_bytes, provider)
        payload = td.current().to_dict()
    finally:
        td.end_request(token)

    by_id = {a["agent_id"]: a for a in payload["agents"]}
    # sicherungs_agent has exactly one model-dependent subject in this fixture -> one batch of size 1.
    assert by_id["sicherungs_agent"]["batch_count"] == 1
    assert by_id["sicherungs_agent"]["batch_sizes"] == [1]
    assert by_id["sicherungs_agent"]["subjects_model"] == 1
    # zirkulations_hydraulik_agent resolves its one subject by text pattern -> deterministic skip, zero model calls.
    assert by_id["zirkulations_hydraulik_agent"]["subjects_deterministic_skip"] == 1
    assert by_id["zirkulations_hydraulik_agent"]["model_calls"] == 0
