"""Document fingerprinting and canonical-JSON helpers, used both by the API
response and by the reproducibility test suite (see tests/test_reproducibility.py).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def document_fingerprint(pdf_bytes: bytes) -> str:
    """SHA-256 of the raw PDF bytes -- identical input always yields an
    identical fingerprint, independent of filename or any parsing step."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def canonical_json(value: Any) -> str:
    """Deterministic JSON serialization for reproducibility comparison:
    sorted object keys, compact separators, no locale/float-repr surprises
    (Python's json module already renders floats deterministically for a
    given value). Two calls with semantically identical input always
    produce byte-identical output."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
