"""Turns a legend entry's own raw label text into a presentable proposed
component name -- minimal transformation only (whitespace/casing), never a
rewrite or a guess. Per the domain expert's explicit instruction: when the
legend already names the component (e.g. "Wasserzähler"), the proposal IS
that name, not something an LLM infers instead.
"""
from __future__ import annotations

import re

# Tokens that keep their own internal casing (abbreviations, codes) rather
# than being title-cased -- extend as real examples surface during review.
_KEEP_AS_IS = {"pwc", "pwh", "dn", "kw", "ww", "up"}


def propose_component_name(raw_label: str) -> str:
    text = re.sub(r"\s+", " ", raw_label or "").strip()
    text = text.strip(" .,:;-–—")
    if not text:
        return ""
    words = text.split(" ")
    out = []
    for w in words:
        core = w.strip("().,:;")
        if core.casefold() in _KEEP_AS_IS:
            out.append(w.upper() if core.isupper() or len(core) <= 3 else w)
            continue
        if w and w[0].islower():
            w = w[0].upper() + w[1:]
        out.append(w)
    return " ".join(out)
