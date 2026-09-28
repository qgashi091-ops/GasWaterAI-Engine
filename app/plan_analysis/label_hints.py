"""Lightweight, non-authoritative pattern matching against SVGW W3 Anhang 4
labeling conventions (pipe media codes, nominal diameters, backflow/venting
device type codes). This gives Base44 a head start when it maps extracted
text onto its own symbol library, but it is a hint only — Base44's symbol
library and AI review remain the source of truth for actual classification.
"""
from __future__ import annotations

import re

# Pipe media codes from the SVGW symbol legend (Symbollegende), e.g. "PWC 80",
# "PWH-C 40", "PWW". Matched case-insensitively, optionally followed by a
# nominal diameter.
_PIPE_CODE = re.compile(r"\bPW[HCW](?:-[A-Z])?\b", re.IGNORECASE)

# Nominal diameter callouts, e.g. "DN20", "DN 50", or outer-diameter callouts
# using the diameter sign, e.g. "ø22", "Ø28" -- both conventions are common on
# Swiss/European plans (DN for larger fittings/valves, ø-notation for PE/PEX
# pipe runs), and OCR/native text can render the diameter sign as either case.
# Also: "Pex16"/"Pex 20" (PEX pipe material + size with no diameter sign, seen
# on several reference plans as the sole PE/PEX size notation), and a bare
# size directly followed by a parenthesised Lastenheiten/load-unit count, e.g.
# "16(2LU)", "28(15LU)" -- a distinct, real DIN 1988/SVGW convention for
# sizing a fixture branch by its load units, independent of the ø-notation.
_DN_SIZE = re.compile(
    r"\bDN\s?\d{1,4}\b"
    r"|[øØ]\s?\d{1,3}\b"
    r"|\bPex\s?\d{1,3}\b"
    r"|\b\d{1,3}\s?\(\s?\d{1,3}\s?LU\s?\)",
    re.IGNORECASE,
)

# Sicherungseinrichtung (backflow/venting device) type codes per W3/E1.
_SAFETY_DEVICE_CODE = re.compile(
    r"\b(AA|AB|AC|AD|AF|BA|CA|DA|DB|DC|EA|EB|HB|HC|HD|LA|LB)\b"
)


def guess_label_hint(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if _PIPE_CODE.search(text):
        return "pipe_media_code"
    if _DN_SIZE.search(text):
        return "nominal_diameter"
    if _SAFETY_DEVICE_CODE.fullmatch(text):
        return "safety_device_code"
    return None
