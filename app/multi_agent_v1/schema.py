"""Shared contracts for the Multi-Agent Architecture v1.

CORE RULE (non-negotiable, enforced in evidence_merger.py, not just
documented here): DETERMINISTIC_FACT always outranks AGENT_OBSERVATION.
No agent output can overwrite a resolved PlanFact, a resolved rule result,
or any other value already produced by this engine's existing deterministic
modules (plan_facts.py, component_evidence.py, canonical_inventory.py,
rules/engine.py). An agent may only fill in a claim that deterministic
evidence left UNRESOLVED/absent, and even then its answer is an
AGENT_OBSERVATION, not a fact -- see AgentObservation.kind below, which is
always exactly "AGENT_OBSERVATION", mirroring the existing micro-agent POC's
own OBSERVATION/PLAN_FACT distinction (app/microagents/symbol_recognition_agent.py).

There is no super-/chief-agent anywhere in this package: AgentRouter and
EvidenceMerger are plain deterministic functions, not agents themselves --
neither ever calls a model, and neither can introduce a claim no individual
agent produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Observation / fact kinds
# ---------------------------------------------------------------------------

ObservationKind = Literal["AGENT_OBSERVATION"]
DeterministicKind = Literal["DETERMINISTIC_FACT"]
Confidence = Literal["supported", "uncertain"]

# The 12 agent ids, spelled exactly as the product order names them --
# used everywhere a value must name "which agent produced/owns this" so a
# typo can't silently create a 13th identity.
AGENT_IDS = (
    "planstruktur_agent",
    "symbol_agent",
    "text_agent",
    "leitungs_agent",
    "anschluss_agent",
    "schlaufungs_agent",
    "sicherungs_agent",
    "stagnations_agent",
    "rueckfluss_agent",
    "zirkulations_hydraulik_agent",
    "probenahme_agent",
    "nachweis_agent",
)


@dataclass
class AgentObservation:
    """One agent's answer to exactly one narrow question about exactly one
    subject. Never a plan-wide verdict, never a claim outside the agent's
    own declared `claim_type` (enforced by tests, see
    tests/multi_agent_v1/test_agent_boundaries.py)."""

    kind: ObservationKind
    agent_id: str
    claim_type: str
    subject_id: str
    value: object
    confidence: Optional[Confidence]
    evidence: list[str] = field(default_factory=list)
    ambiguity: Optional[str] = None
    available: bool = True
    error: Optional[str] = None
    model: Optional[str] = None
    prompt_hash: Optional[str] = None
    input_hash: Optional[str] = None
    cached: bool = False
    latency_ms: Optional[float] = None
    raw_response_id: Optional[str] = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != "AGENT_OBSERVATION":
            raise ValueError(f"AgentObservation.kind must be 'AGENT_OBSERVATION', got {self.kind!r}")
        if self.agent_id not in AGENT_IDS:
            raise ValueError(f"Unknown agent_id {self.agent_id!r}; not one of {AGENT_IDS}")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "agent_id": self.agent_id, "claim_type": self.claim_type,
            "subject_id": self.subject_id, "value": self.value, "confidence": self.confidence,
            "evidence": list(self.evidence), "ambiguity": self.ambiguity, "available": self.available,
            "error": self.error, "model": self.model, "prompt_hash": self.prompt_hash,
            "input_hash": self.input_hash, "cached": self.cached, "latency_ms": self.latency_ms,
            "raw_response_id": self.raw_response_id, "detail": self.detail,
        }


@dataclass
class DeterministicFactRef:
    """A thin, read-only pointer into an ALREADY-COMPUTED deterministic
    result (a PlanFact, a ComponentEvidence resolution, a rule CheckResult,
    a canonical-inventory field, ...). This package never recomputes any of
    those; it only wraps enough of the existing object for the merger to
    reason about priority and provenance. `source_module` names exactly
    which existing module produced it, so provenance is always traceable to
    real, already-tested code, never to this package inventing a fact."""

    fact_id: str
    claim_type: str
    subject_id: str
    value: object
    resolved: bool  # False for e.g. plan_facts.PlanFact(kind="UNRESOLVED") -- an unresolved deterministic
    #                  fact does NOT block an agent from being routed in; it only means there is
    #                  nothing here yet for the merger to prioritize over the agent's observation.
    source_module: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": "DETERMINISTIC_FACT", "fact_id": self.fact_id, "claim_type": self.claim_type,
            "subject_id": self.subject_id, "value": self.value, "resolved": self.resolved,
            "source_module": self.source_module, "detail": self.detail,
        }


Resolution = Literal[
    "DETERMINISTIC",           # a resolved DeterministicFactRef decided it; agent observations (if any) are pure corroboration/conflict record, never decisive
    "AGENT_AGREED",            # no resolved deterministic fact; >=2 agent observations agree
    "AGENT_SINGLE",            # no resolved deterministic fact; exactly one agent observation
    "AGENT_CONFLICT",          # no resolved deterministic fact; agent observations disagree -- reported, never silently picked
    "UNRESOLVED",              # neither a resolved deterministic fact nor any usable agent observation
]


@dataclass
class ConflictRecord:
    claim_type: str
    subject_id: str
    description: str
    competing_values: list[dict]  # [{"source": ..., "value": ..., "kind": "DETERMINISTIC_FACT"|"AGENT_OBSERVATION"}]

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type, "subject_id": self.subject_id,
            "description": self.description, "competing_values": self.competing_values,
        }


@dataclass
class MergedClaim:
    """One row of CanonicalPlanUnderstanding: the merger's final answer for
    one (claim_type, subject_id) pair, plus full provenance. `value` is
    NEVER computed by the merger itself -- it is always copied verbatim from
    either the winning DeterministicFactRef or the winning AgentObservation,
    so every value in the canonical output can be traced to one real
    producing module or agent."""

    claim_type: str
    subject_id: str
    value: object
    resolution: Resolution
    provenance: list[dict]  # ordered list of {"kind", "source", ...} entries that contributed
    conflicts: list[ConflictRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type, "subject_id": self.subject_id, "value": self.value,
            "resolution": self.resolution, "provenance": self.provenance,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


@dataclass
class RoutingDecision:
    agent_id: str
    run: bool
    reason: str
    subjects: list[str] = field(default_factory=list)  # subject_ids this agent will actually be asked about; empty if run=False

    def to_dict(self) -> dict:
        return {"agent_id": self.agent_id, "run": self.run, "reason": self.reason, "subjects": list(self.subjects)}


@dataclass
class CallDiagnostics:
    """Cost/call accounting for one agent invocation -- aggregated across a
    whole run in diagnostics.py. Never includes prompt/response text (only
    hashes), matching this codebase's existing privacy discipline (see
    vision_fallback/interface.py's VisionFallbackResult docstring)."""

    agent_id: str
    subject_id: str
    called_model: bool
    cached: bool
    available: bool
    latency_ms: Optional[float]
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id, "subject_id": self.subject_id, "called_model": self.called_model,
            "cached": self.cached, "available": self.available, "latency_ms": self.latency_ms, "error": self.error,
        }
