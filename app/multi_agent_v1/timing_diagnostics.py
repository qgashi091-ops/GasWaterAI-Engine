"""Optional runtime timing diagnostics for POST /multi_agent_v1/analyze.

Gated entirely by GASWATERAI_TIMING_DIAGNOSTICS=1 (checked per request, not
cached at import time, so tests can toggle it with monkeypatch). With it
unset/disabled -- today's default -- every function here is a cheap no-op
(one contextvar lookup) and nothing is logged: this module changes no
agent, router, prompt, timeout, retry, model-call-count, or
parallelization behavior anywhere. It only measures durations and counts
around the EXISTING, unchanged serial call sequence
(pipeline.py/base_agent.py), the same way app/main.py's own `timing_ms`
blocks already measure /analyze and /check without changing them.

Logs exactly one structured line per request (Python logging, logger
"gaswaterai.timing_diagnostics") containing ONLY numbers, status enums,
and agent_id/claim_type identifiers that already exist as routing/schema
constants. Never logs image bytes, base64, prompts, plan text, secrets, or
any other request/response content -- `record_model_call()` below takes
only a byte COUNT and a duration, never the bytes themselves.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from .provider import AgentModelResponse

logger = logging.getLogger("gaswaterai.timing_diagnostics")

TIMING_DIAGNOSTICS_ENV_VAR = "GASWATERAI_TIMING_DIAGNOSTICS"

# pipeline.py calls every agent, and every subject within an agent, from one
# single top-to-bottom Python for-loop -- there is no thread pool, asyncio
# gather, or other concurrency anywhere in that module. This is a static
# fact about the existing code, not something measured per request, but it
# is included in every logged line so a reader never has to go re-check
# pipeline.py to answer "serial or parallel".
EXECUTION_MODE = "serial"


def enabled() -> bool:
    return os.environ.get(TIMING_DIAGNOSTICS_ENV_VAR) == "1"


def classify_status(response: AgentModelResponse) -> str:
    """SUCCESS / ERROR / TIMEOUT, derived only from the response's own
    `available`/`error` fields -- no new exception handling, no change to
    what a provider does on a real timeout (that already becomes a
    `URLError`/`TimeoutError` the provider itself catches and turns into
    `available=False, error="...timed out..."` -- see
    Base44AgentModelProvider.call and AnthropicAgentModelProvider.call)."""
    if response.available:
        return "SUCCESS"
    if response.error and "timeout" in response.error.lower():
        return "TIMEOUT"
    if response.error and "timed out" in response.error.lower():
        return "TIMEOUT"
    return "ERROR"


@dataclass
class ModelCallTiming:
    duration_ms: float
    images: int
    status: str  # "SUCCESS" | "ERROR" | "TIMEOUT"


@dataclass
class AgentTiming:
    agent_id: str
    run: bool
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    model_calls: list = field(default_factory=list)  # list[ModelCallTiming]

    def to_dict(self) -> dict:
        statuses = [c.status for c in self.model_calls]
        if not self.run:
            status = "SKIP"
        elif "TIMEOUT" in statuses:
            status = "TIMEOUT"
        elif "ERROR" in statuses:
            status = "ERROR"
        else:
            status = "SUCCESS"
        return {
            "agent_id": self.agent_id,
            "run": self.run,
            "start_ms": round(self.start_ms, 2) if self.start_ms is not None else None,
            "end_ms": round(self.end_ms, 2) if self.end_ms is not None else None,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms is not None else None,
            "model_calls": len(self.model_calls),
            "images": sum(c.images for c in self.model_calls),
            "provider_duration_ms_per_call": [round(c.duration_ms, 2) for c in self.model_calls],
            "status": status,
        }


class TimingRecorder:
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self._request_start = time.perf_counter()
        self.phases_ms: dict[str, float] = {}
        self.agents: dict[str, AgentTiming] = {}
        self._agent_order: list[str] = []

    def add_phase_ms(self, name: str, extra_ms: float) -> None:
        self.phases_ms[name] = self.phases_ms.get(name, 0.0) + extra_ms

    def _agent_timing(self, agent_id: str, run: bool) -> AgentTiming:
        timing = self.agents.get(agent_id)
        if timing is None:
            timing = AgentTiming(agent_id=agent_id, run=run)
            self.agents[agent_id] = timing
            self._agent_order.append(agent_id)
        return timing

    @contextmanager
    def agent_scope(self, agent_id: str, run: bool):
        timing = self._agent_timing(agent_id, run)
        timing.run = run
        if not run:
            yield timing
            return
        timing.start_ms = (time.perf_counter() - self._request_start) * 1000
        t0 = time.perf_counter()
        try:
            yield timing
        finally:
            timing.duration_ms = (time.perf_counter() - t0) * 1000
            timing.end_ms = (time.perf_counter() - self._request_start) * 1000

    def record_model_call(self, agent_id: str, duration_ms: float, images: int, status: str) -> None:
        timing = self._agent_timing(agent_id, run=True)
        timing.model_calls.append(ModelCallTiming(duration_ms=duration_ms, images=images, status=status))

    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self._request_start) * 1000

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "execution_mode": EXECUTION_MODE,
            "total_duration_ms": round(self.total_duration_ms(), 2),
            "phases_ms": {k: round(v, 2) for k, v in self.phases_ms.items()},
            "agents": [self.agents[a].to_dict() for a in self._agent_order],
        }


_current: "contextvars.ContextVar[Optional[TimingRecorder]]" = contextvars.ContextVar(
    "gaswaterai_timing_recorder", default=None,
)


def start_request() -> contextvars.Token:
    """Call once at the top of the /multi_agent_v1/analyze route. Creates
    and activates a new TimingRecorder only when the env var is set;
    otherwise activates nothing (every helper below becomes a no-op).
    Always returns a token -- pass it to `end_request()` in a `finally`."""
    recorder = TimingRecorder(correlation_id=uuid.uuid4().hex[:12]) if enabled() else None
    return _current.set(recorder)


def end_request(token: contextvars.Token) -> None:
    _current.reset(token)


def current() -> Optional[TimingRecorder]:
    return _current.get()


@contextmanager
def phase(name: str):
    """No-op when diagnostics are disabled or no request is active."""
    recorder = current()
    if recorder is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        recorder.add_phase_ms(name, (time.perf_counter() - t0) * 1000)


@contextmanager
def agent_scope(agent_id: str, run: bool):
    """No-op when diagnostics are disabled or no request is active."""
    recorder = current()
    if recorder is None:
        yield None
        return
    with recorder.agent_scope(agent_id, run) as timing:
        yield timing


def record_model_call(agent_id: str, duration_ms: float, images: int, status: str) -> None:
    """No-op when diagnostics are disabled or no request is active."""
    recorder = current()
    if recorder is None:
        return
    recorder.record_model_call(agent_id, duration_ms, images, status)


def finish_and_log() -> None:
    """No-op when diagnostics are disabled or no request is active. Emits
    exactly one structured log line -- see module docstring for the
    no-content-logged guarantee."""
    recorder = current()
    if recorder is None:
        return
    payload = recorder.to_dict()
    logger.info("multi_agent_v1_timing %s", json.dumps(payload))
