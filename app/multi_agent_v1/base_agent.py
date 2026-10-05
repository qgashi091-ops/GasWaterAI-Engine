"""BaseAgent -- the shared plumbing every one of the 12 agents inherits so
each agent module contains only its own narrow domain logic, never repeated
boilerplate for hashing, error handling, or observation construction.

AGENT BOUNDARY ENFORCEMENT: `agent_id` and `claim_types` are declared once
per agent class and checked on every emitted observation
(`_observation()` below raises if a subclass tries to emit a claim_type it
did not declare, or an agent_id other than its own). This is what
tests/multi_agent_v1/test_agent_boundaries.py exercises -- an agent
literally cannot emit outside its declared responsibility, it is not just a
convention."""
from __future__ import annotations

import time
from abc import ABC
from typing import Optional

from . import timing_diagnostics
from .provider import AgentModelProvider, AgentModelRequest, AgentModelResponse
from .schema import AgentObservation, Confidence


class BaseAgent(ABC):
    agent_id: str = ""
    claim_types: tuple[str, ...] = ()

    def __init__(self, provider: AgentModelProvider):
        if not self.agent_id:
            raise ValueError(f"{type(self).__name__} must set a non-empty agent_id class attribute.")
        if not self.claim_types:
            raise ValueError(f"{type(self).__name__} must declare at least one claim_type.")
        self.provider = provider

    def call_model(self, request: AgentModelRequest) -> AgentModelResponse:
        """Every agent's only path to a model call -- the one central
        point that can time it (GASWATERAI_TIMING_DIAGNOSTICS=1 only) and
        attribute it to `self.agent_id`, without touching any agent's own
        logic, retry behavior, or the provider itself. When diagnostics
        are disabled, `timing_diagnostics.current()` is None and this is
        exactly `return self.provider.call(request)` -- same call, same
        count, same timeout, nothing added."""
        if timing_diagnostics.current() is None:
            return self.provider.call(request)
        t0 = time.perf_counter()
        response = self.provider.call(request)
        duration_ms = (time.perf_counter() - t0) * 1000
        timing_diagnostics.record_model_call(
            agent_id=self.agent_id, duration_ms=duration_ms,
            images=len(request.images), status=timing_diagnostics.classify_status(response),
        )
        return response

    def _observation(
        self, claim_type: str, subject_id: str, value: object,
        confidence: Optional[Confidence], evidence: Optional[list[str]] = None,
        ambiguity: Optional[str] = None, available: bool = True, error: Optional[str] = None,
        model: Optional[str] = None, prompt_hash: Optional[str] = None, input_hash: Optional[str] = None,
        cached: bool = False, latency_ms: Optional[float] = None, raw_response_id: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> AgentObservation:
        if claim_type not in self.claim_types:
            raise ValueError(
                f"{type(self).__name__} (agent_id={self.agent_id!r}) tried to emit claim_type "
                f"{claim_type!r}, which is outside its declared claim_types {self.claim_types!r}."
            )
        return AgentObservation(
            kind="AGENT_OBSERVATION", agent_id=self.agent_id, claim_type=claim_type, subject_id=subject_id,
            value=value, confidence=confidence, evidence=evidence or [], ambiguity=ambiguity,
            available=available, error=error, model=model, prompt_hash=prompt_hash, input_hash=input_hash,
            cached=cached, latency_ms=latency_ms, raw_response_id=raw_response_id, detail=detail or {},
        )

    def _unavailable(self, claim_type: str, subject_id: str, error: str) -> AgentObservation:
        return self._observation(claim_type, subject_id, value=None, confidence=None, available=False, error=error)
