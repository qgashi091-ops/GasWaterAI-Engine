"""Abstract vision-fallback provider interface, kept deliberately small so a
concrete provider (Anthropic today) can be swapped without touching any
caller. A caller ALWAYS gets a `VisionFallbackResult` back, even on total
failure (`available=False`) -- component evidence fusion (see
app/plan_analysis/component_evidence.py) never crashes because a vision
call failed; it just records the source as unavailable and falls through to
COMPONENT_UNRESOLVED if nothing else resolves the candidate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class VisionFallbackResult:
    available: bool
    component_type: str | None  # one of the candidate_labels passed in, or None
    confidence: float | None  # 0..1, provider's own reported confidence -- never fabricated by the caller
    reasoning: str | None
    provider: str
    model: str | None
    raw_response_id: str | None = None  # provider's own response/request id, for audit -- never the full raw text (privacy: a plan crop can contain real text)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available, "component_type": self.component_type,
            "confidence": self.confidence, "reasoning": self.reasoning,
            "provider": self.provider, "model": self.model,
            "raw_response_id": self.raw_response_id, "error": self.error,
        }


class VisionFallbackProvider(ABC):
    """A provider identifies a component's TYPE from a cropped region image.
    It MUST NOT be asked, and must never be interpreted, to determine pipe
    connectivity, topology, real-world length, or a compliance verdict --
    those are enforced deterministic-only at the caller (canonical_inventory.py
    and the rule engine never read a vision result for any of those fields)."""

    @abstractmethod
    def classify_region(
        self, image_bytes: bytes, image_media_type: str, candidate_labels: list[str], context: dict,
    ) -> VisionFallbackResult:
        """`candidate_labels`: the closed set of component types this call may
        choose from (plus an implicit "unknown"/none-of-these outcome) -- this
        engine never asks vision to freely invent a label outside taxonomy.
        `context`: small, non-identifying hints only (e.g. nearby confirmed
        legend text already extracted deterministically), never the whole
        plan and never anything not already derived by this engine itself."""
        raise NotImplementedError
