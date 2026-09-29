"""v0.4 Phase 2 -- privacy audit.

Scans a plan's own NATIVE TEXT (never the raw PDF bytes/images -- this is a
text-pattern scan, not OCR/vision) for patterns that typically identify a
real person, address, or company in a Swiss title block: a street name
ending in -strasse/-weg/-platz followed by a house number, a Swiss postal
code + town pair, common title-block field labels ("Bauherr", "Architekt",
"Kunde", ...) followed by a value, and company-suffix tokens (AG, GmbH).

This is deliberately a set of hand-checkable REGEX heuristics, not a named-
entity-recognition model and not an LLM call -- consistent with this
engine's whole "no LLM invents anything" discipline, just applied here to
privacy risk instead of component identity. It is intentionally biased
toward over-flagging (a false positive costs nothing -- the region is
already excluded from crops by construction, see candidates.py) rather than
under-flagging.

CASE-SENSITIVITY -- a real regression, found via taxonomy.py's own reuse of
this module: `legend_intelligence.py`'s `normalized_label` text is
lowercased, and `SWISS_PLZ_TOWN_PATTERN`/`COMPANY_SUFFIX_PATTERN` were
missing `re.IGNORECASE` (unlike `STREET_PATTERN`/`TITLE_BLOCK_FIELD_PATTERN`,
which already had it) -- so a real postal-code+town pair ("6023
Rothenburg") and a real company name+suffix, both present verbatim in one
of this batch's plans' legend text, silently passed every check once
lowercased, and would have reached a committed JSON file
(pipeline_summary.json/classes.json) before this fix. All patterns here are
now case-insensitive for exactly this reason: nothing calling `scan_text`
should have to already know or preserve the original casing of its input
for this module to do its job.

Source PDFs are never modified or deleted -- this module only reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

STREET_PATTERN = re.compile(
    r"\b[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ]+(?:strasse|straße|weg|platz|gasse|allee)[\s\-]*\d{1,4}[a-z]?\b",
    re.IGNORECASE,
)
SWISS_PLZ_TOWN_PATTERN = re.compile(r"\b(\d{4})\s+([A-ZÄÖÜ][a-zäöüA-ZÄÖÜ\-]+)\b", re.IGNORECASE)
TITLE_BLOCK_FIELD_PATTERN = re.compile(
    r"\b(Bauherr(?:schaft)?|Architekt(?:in)?|Kunde|Planer|Fachplaner|Bauleitung|Unternehmer|Auftraggeber)\s*:?\s*([A-ZÄÖÜ][\wäöüÄÖÜß .,&\-]{2,40})",
    re.IGNORECASE,
)
COMPANY_SUFFIX_PATTERN = re.compile(r"\b[A-ZÄÖÜ][\wäöüÄÖÜß .\-]{1,40}\s+(AG|GmbH|SA|Sàrl)\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b0\d{2}[ /.]?\d{3}\s?\d{2}\s?\d{2}\b")
EMAIL_PATTERN = re.compile(r"\b[\w.\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b")

FINDING_KINDS = {
    "street_address": STREET_PATTERN,
    "postal_code_town": SWISS_PLZ_TOWN_PATTERN,
    "title_block_field": TITLE_BLOCK_FIELD_PATTERN,
    "company_name": COMPANY_SUFFIX_PATTERN,
    "phone_number": PHONE_PATTERN,
    "email_address": EMAIL_PATTERN,
}


@dataclass
class PrivacyFinding:
    kind: str
    count: int
    # Deliberately NOT stored: the matched text itself. A privacy report
    # that repeats the very names/addresses it is warning about would defeat
    # its own purpose -- only the finding KIND and COUNT are kept.


@dataclass
class PlanPrivacyAudit:
    plan_id: str
    findings: list = field(default_factory=list)  # list[PrivacyFinding]
    filename_flagged: bool = False
    filename_flag_reason: str = ""

    @property
    def has_pii_risk(self) -> bool:
        return bool(self.findings) or self.filename_flagged

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "has_pii_risk": self.has_pii_risk,
            "findings": [{"kind": f.kind, "count": f.count} for f in self.findings],
            "filename_flagged": self.filename_flagged,
            "filename_flag_reason": self.filename_flag_reason,
        }


def scan_text(plan_id: str, native_text: str) -> PlanPrivacyAudit:
    audit = PlanPrivacyAudit(plan_id=plan_id)
    for kind, pattern in FINDING_KINDS.items():
        matches = pattern.findall(native_text)
        if matches:
            audit.findings.append(PrivacyFinding(kind=kind, count=len(matches)))
    return audit


# The ORIGINAL filename (never the pseudonym) is checked once, at ingest
# time, then discarded -- never logged verbatim anywhere under app/ or
# docs/. A filename embedding a real street name, a surname-shaped token,
# or a full address is exactly the kind of leak a "just look at the file
# list" workflow would miss if pseudonymization happened only at the report
# layer.
def scan_filename(original_filename: str) -> tuple[bool, str]:
    stem = original_filename.rsplit(".", 1)[0]
    if STREET_PATTERN.search(stem):
        return True, "filename appears to contain a street address"
    if re.search(r"[A-ZÄÖÜ][a-zäöü]+(?:_|-)?[A-ZÄÖÜ][a-zäöü]{3,}", stem) and not re.search(
        r"schema|plan|sanit|wasser|strang|versorg", stem, re.IGNORECASE
    ):
        return True, "filename token pattern resembles a personal or place name"
    return False, ""
