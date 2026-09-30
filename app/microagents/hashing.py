"""Shared, deterministic hashing utilities for the micro-agent POC --
records exactly what the credit/efficiency rule requires (model +
prompt_hash + crop_hash + text_hash) and what Phase 3 requires (exact
prompt hash, exact input crop hash).
"""
from __future__ import annotations

import hashlib


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combine_hashes(*hashes: str) -> str:
    """Deterministically combines several independent hashes into one --
    used to fold a tight-crop hash and an optional context-crop hash into
    the single `crop_hash` component of the cache key, without ever hiding
    which inputs contributed (the inputs are recorded separately in the
    agent's own provenance record regardless of this combined value)."""
    return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()


def cache_key(model: str, prompt_hash: str, crop_hash: str, text_hash: str) -> str:
    return f"{model}|{prompt_hash}|{crop_hash}|{text_hash}"
