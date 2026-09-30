"""Development-time response cache -- CREDIT/EFFICIENCY RULE: "Identical
inputs should reuse the cached response during development unless a
deliberate reproducibility run requires a fresh model call."

A flat JSON-file-per-key store is deliberately the simplest thing that
works for a POC at this call volume (tens to low hundreds of calls) -- no
database, no TTL logic, no eviction policy.
"""
from __future__ import annotations

import json
from pathlib import Path


class ResponseCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # cache keys contain '|' and model ids may contain '/'-unsafe
        # characters in principle -- hash the key itself for the filename,
        # keep the real key inside the stored record for auditability.
        import hashlib
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record.get("cache_key") != key:
            return None  # hash collision safety net -- treat as a miss, never return the wrong record
        return record["value"]

    def set(self, key: str, value: dict) -> None:
        path = self._path(key)
        path.write_text(json.dumps({"cache_key": key, "value": value}, indent=2, ensure_ascii=False))
