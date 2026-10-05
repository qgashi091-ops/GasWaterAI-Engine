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
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from .provider import AgentModelResponse

logger = logging.getLogger("gaswaterai.timing_diagnostics")

# Root cause of the live symptom (Render shows the uvicorn access line for
# /multi_agent_v1/analyze, never a timing line): `logging.getLogger(name)`
# creates a logger at level NOTSET, which defers to its parent's effective
# level -- and nothing in this process ever configures the root logger
# (no `logging.basicConfig()`/`dictConfig()` anywhere in app/, confirmed by
# inspection). Python's hardcoded root level is WARNING, so every
# `logger.info(...)` call below was being dropped by the standard level
# check before a handler, propagation, or Render's log capture ever entered
# the picture -- GASWATERAI_TIMING_DIAGNOSTICS=1 was read correctly and the
# log call was reached, it just never passed the level gate. Uvicorn's own
# default logging config (uvicorn.config.LOGGING_CONFIG) only configures
# its OWN "uvicorn"/"uvicorn.access"/"uvicorn.error" loggers, explicitly
# leaves `disable_existing_loggers: False`, and never touches the root
# logger or any application logger -- so it neither breaks nor fixes this.
#
# Fix: this module owns and configures ONLY its own named logger (never
# `logging.basicConfig()`, never root) -- an explicit level and a
# `StreamHandler` (stderr, matching the stream Render already captures for
# every other log line, including uvicorn's own default/error handler) so
# emission never depends on whatever the hosting process did or didn't
# configure elsewhere. `propagate` stays at its default (True): if the
# process ever gains its own root handler later, the line is still seen
# there too, exactly once (this handler never touches, replaces, or
# silences anything else).
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setLevel(logging.INFO)
    logger.addHandler(_handler)

TIMING_DIAGNOSTICS_ENV_VAR = "GASWATERAI_TIMING_DIAGNOSTICS"
MAX_CONCURRENCY_ENV_VAR = "GASWATERAI_AGENT_MAX_CONCURRENCY"
DEFAULT_MAX_CONCURRENCY = 3

# Performance-optimization epic: pipeline.py now runs Stage A's 10
# mutually-independent agents' batch calls concurrently, bounded by
# GASWATERAI_AGENT_MAX_CONCURRENCY (see pipeline.py's module docstring for
# the exact, code-derived dependency analysis -- RueckflussAgent needs
# SicherungsAgent's same-run output (Stage B), NachweisAgent needs every
# other agent's full observation list (Stage C), every other agent reads
# only already-computed deterministic context/inventory/plan_facts and is
# therefore independent of every other agent's output). Logged so a reader
# never has to go re-check pipeline.py to answer "serial or parallel", and
# so a run with GASWATERAI_AGENT_MAX_CONCURRENCY=1 is visibly distinct from
# true single-threaded execution.
EXECUTION_MODE = "staged-concurrent"


def max_concurrency() -> int:
    try:
        value = int(os.environ.get(MAX_CONCURRENCY_ENV_VAR, DEFAULT_MAX_CONCURRENCY))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONCURRENCY
    return value if value >= 1 else DEFAULT_MAX_CONCURRENCY


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
    subjects_total: int = 0
    subjects_deterministic_skip: int = 0  # resolved by a text-pattern/legend/keyword/PlanFacts-skip shortcut, never entered a batch
    subjects_model: int = 0               # subjects that actually needed a model call (cache hit or not)
    batch_count: int = 0
    batch_sizes: list = field(default_factory=list)  # list[int], one entry per provider call made
    cache_hits: int = 0

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
            "subjects_total": self.subjects_total,
            "subjects_deterministic_skip": self.subjects_deterministic_skip,
            "subjects_model": self.subjects_model,
            "batch_count": self.batch_count,
            "batch_sizes": list(self.batch_sizes),
            "model_calls": len(self.model_calls),
            "cache_hits": self.cache_hits,
            "images": sum(c.images for c in self.model_calls),
            "provider_duration_ms_per_call": [round(c.duration_ms, 2) for c in self.model_calls],
            "errors": sum(1 for s in statuses if s == "ERROR"),
            "timeouts": sum(1 for s in statuses if s == "TIMEOUT"),
            "status": status,
        }


class TimingRecorder:
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.max_concurrency = max_concurrency()
        self._request_start = time.perf_counter()
        self.phases_ms: dict[str, float] = {}
        self.agents: dict[str, AgentTiming] = {}
        self._agent_order: list[str] = []
        self._dropped_extra_results: int = 0
        # Guards every mutation below -- Stage A dispatches multiple
        # agents' batch calls onto a shared ThreadPoolExecutor
        # (pipeline.py), so several worker threads can call record_*
        # concurrently for different agents at once.
        self._lock = threading.Lock()

    def add_phase_ms(self, name: str, extra_ms: float) -> None:
        with self._lock:
            self.phases_ms[name] = self.phases_ms.get(name, 0.0) + extra_ms

    def _agent_timing_locked(self, agent_id: str, run: bool) -> AgentTiming:
        timing = self.agents.get(agent_id)
        if timing is None:
            timing = AgentTiming(agent_id=agent_id, run=run)
            self.agents[agent_id] = timing
            self._agent_order.append(agent_id)
        return timing

    def begin_agent(self, agent_id: str, run: bool, subjects_total: int = 0) -> None:
        """Explicit start, paired with `end_agent()` -- used instead of the
        `agent_scope()` context manager when an agent's actual work is
        dispatched to a thread pool and must stay "open" across a stage
        barrier rather than closing when the submitting code returns."""
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run)
            timing.run = run
            timing.subjects_total = subjects_total
            if run:
                timing.start_ms = (time.perf_counter() - self._request_start) * 1000

    def end_agent(self, agent_id: str) -> None:
        with self._lock:
            timing = self.agents.get(agent_id)
            if timing is None or not timing.run or timing.start_ms is None:
                return
            timing.end_ms = (time.perf_counter() - self._request_start) * 1000
            timing.duration_ms = timing.end_ms - timing.start_ms

    @contextmanager
    def agent_scope(self, agent_id: str, run: bool):
        """Simple synchronous sibling of begin_agent/end_agent, for a
        single agent whose own work is NOT split across a concurrency
        barrier (Stage B/C, and every unit test that still calls one
        agent directly)."""
        self.begin_agent(agent_id, run)
        try:
            yield self.agents.get(agent_id)
        finally:
            self.end_agent(agent_id)

    def record_model_call(self, agent_id: str, duration_ms: float, images: int, status: str) -> None:
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run=True)
            timing.model_calls.append(ModelCallTiming(duration_ms=duration_ms, images=images, status=status))

    def record_subjects_deterministic_skip(self, agent_id: str, count: int) -> None:
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run=True)
            timing.subjects_deterministic_skip += count

    def record_subjects_model(self, agent_id: str, count: int) -> None:
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run=True)
            timing.subjects_model += count

    def record_cache_hit(self, agent_id: str, count: int = 1) -> None:
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run=True)
            timing.cache_hits += count

    def record_batch_plan(self, agent_id: str, batch_count: int, batch_sizes: list) -> None:
        with self._lock:
            timing = self._agent_timing_locked(agent_id, run=True)
            timing.batch_count += batch_count
            timing.batch_sizes.extend(batch_sizes)

    def record_dropped_extra_results(self, agent_id: str, count: int) -> None:
        with self._lock:
            self._dropped_extra_results += count

    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self._request_start) * 1000

    def to_dict(self) -> dict:
        with self._lock:
            agent_order = list(self._agent_order)
            agents_dict = {k: v for k, v in self.agents.items()}
            phases = dict(self.phases_ms)
            dropped = self._dropped_extra_results
        return {
            "correlation_id": self.correlation_id,
            "execution_mode": EXECUTION_MODE,
            "max_concurrency": self.max_concurrency,
            "total_duration_ms": round(self.total_duration_ms(), 2),
            "phases_ms": {k: round(v, 2) for k, v in phases.items()},
            "agents": [agents_dict[a].to_dict() for a in agent_order],
            "dropped_extra_batch_results": dropped,
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


def begin_agent(agent_id: str, run: bool, subjects_total: int = 0) -> None:
    """No-op when diagnostics are disabled or no request is active. Pair
    with `end_agent()` -- see TimingRecorder.begin_agent's docstring for
    when to use this instead of `agent_scope()`."""
    recorder = current()
    if recorder is None:
        return
    recorder.begin_agent(agent_id, run, subjects_total=subjects_total)


def end_agent(agent_id: str) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.end_agent(agent_id)


def record_model_call(agent_id: str, duration_ms: float, images: int, status: str) -> None:
    """No-op when diagnostics are disabled or no request is active."""
    recorder = current()
    if recorder is None:
        return
    recorder.record_model_call(agent_id, duration_ms, images, status)


def record_subjects_deterministic_skip(agent_id: str, count: int) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.record_subjects_deterministic_skip(agent_id, count)


def record_subjects_model(agent_id: str, count: int) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.record_subjects_model(agent_id, count)


def record_cache_hit(agent_id: str, count: int = 1) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.record_cache_hit(agent_id, count)


def record_batch_plan(agent_id: str, batch_count: int, batch_sizes: list) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.record_batch_plan(agent_id, batch_count, batch_sizes)


def record_dropped_extra_results(agent_id: str, count: int) -> None:
    recorder = current()
    if recorder is None:
        return
    recorder.record_dropped_extra_results(agent_id, count)


def run_in_current_context(fn, /, *args, **kwargs):
    """Wraps `fn` so that, when invoked from a DIFFERENT thread (as
    ThreadPoolExecutor does), it runs with THIS thread's contextvars --
    specifically, the active TimingRecorder (and GASWATERAI_TIMING_
    DIAGNOSTICS's resolved state). Python does not propagate contextvars
    into a new thread automatically; `concurrent.futures.ThreadPoolExecutor
    .submit()` does not copy the caller's context either. Without this,
    every timing call made from a worker thread would silently see
    `current() is None` and record nothing, even with diagnostics
    enabled -- pipeline.py's Stage-A executor uses this for every
    submitted callable."""
    ctx = contextvars.copy_context()

    def _runner():
        return ctx.run(fn, *args, **kwargs)

    return _runner


def finish_and_log() -> None:
    """No-op when diagnostics are disabled or no request is active. Emits
    exactly one structured log line -- see module docstring for the
    no-content-logged guarantee."""
    recorder = current()
    if recorder is None:
        return
    payload = recorder.to_dict()
    logger.info("multi_agent_v1_timing %s", json.dumps(payload))
