"""AgentResponseCache -- a request-scoped (never cross-plan, never
cross-request) in-memory cache so identical model-call inputs within the
SAME /multi_agent_v1/analyze run are never paid for twice.

Scope is deliberately per-run, not persistent: a fresh instance is created
in `run_multi_agent_v1()` and discarded when that call returns. This
engine is stateless across HTTP requests (no database, no shared process
state -- see app/main.py's own docstring), and persisting model answers
across different plans would be both a correctness risk (a crop that
happens to hash the same across two different plans is extremely
unlikely but not provably impossible) and unnecessary: within one run,
batching already collapses most repeated work, and the only remaining
duplicate-input case this is for is two different subjects that happen
to produce byte-identical crops/text within the same plan.

Cache key covers everything two inputs must match on to be considered the
"same question": the agent asking (agent_id), which model would answer it
(model), the exact prompt/contract version (system_prompt + tool_schema,
hashed -- so changing a prompt or a schema can never hit a stale cache
entry), the per-subject text actually sent (captures nearby_text/
candidate_labels/etc that are baked into that text), any additional
caller-supplied "cache_facts" (hashable extra fields that influence
behaviour without being textual content, e.g. candidate-label tuples),
and every image's own content hash -- never a filename or index, so two
subjects with visually different crops never collide even if everything
else matches."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class CachedOutcome:
    tool_input: Optional[dict]
    available: bool
    error: Optional[str]
    model: Optional[str]
    raw_response_id: Optional[str]


def cache_key_for(
    agent_id: str, model: Optional[str], system_prompt: str, tool_schema: dict,
    text: str, cache_facts: tuple, images: list,
) -> str:
    h = hashlib.sha256()
    h.update(agent_id.encode("utf-8")); h.update(b"|")
    h.update(str(model).encode("utf-8")); h.update(b"|")
    h.update(hashlib.sha256(system_prompt.encode("utf-8")).digest()); h.update(b"|")
    h.update(hashlib.sha256(json.dumps(tool_schema, sort_keys=True, default=str).encode("utf-8")).digest())
    h.update(b"|")
    h.update(hashlib.sha256(text.encode("utf-8")).digest()); h.update(b"|")
    h.update(repr(cache_facts).encode("utf-8")); h.update(b"|")
    for img in images:
        h.update(hashlib.sha256(img).digest())
    return h.hexdigest()


class AgentResponseCache:
    """Thread-safe (batches from different agents may run concurrently
    under the Stage-A executor and share one cache instance for the whole
    run) -- a plain dict guarded by a lock, nothing fancier needed at this
    scale (one plan, a few dozen subjects at most)."""

    def __init__(self) -> None:
        self._store: dict[str, CachedOutcome] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[CachedOutcome]:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: CachedOutcome) -> None:
        with self._lock:
            self._store[key] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
