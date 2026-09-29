"""v0.4 Phase 5 -- annotation taxonomy proposal.

Deliberately NOT the 58-symbol SVGW library (task requirement). Proposes
roughly 10-15 classes purely from THIS batch's own measured evidence:

1. Icon frequency: every USABLE_TEMPLATE, non-swatch legend entry's own
   `normalized_label` across all 20 plans' legend_intelligence results
   (legend_entries.py) -- a class this frequent in real legends is a class
   worth teaching a detector.
2. Backflow/venting device codes (EA/CA/BA/...): v0.1's OWN, already-
   existing `label_hints.py` deterministic pattern match
   (`label_hint == "safety_device_code"`) over every plan's native text --
   reused, not reimplemented, exactly matching the task's suggested classes
   EA/CA/BA without inventing a new detector for them.
3. Drawing-side corroboration: the generic-library's own top suggested
   label (`generic_evidence.symbol_name`) across every plan's hybrid
   component facts -- a class the generic matcher proposes often for REAL
   drawing candidates (not just legend rows) is additional evidence the
   class actually appears in the drawings themselves, not just in legends
   that may go unused.

No class is invented that isn't backed by at least MIN_SUPPORT occurrences
somewhere in this evidence -- "Do not invent classes unsupported by the
plans" is enforced structurally, not just by intent.

MANDATORY PRIVACY FILTER -- found the hard way, on this exact batch:
`legend_entries.py`'s own `normalized_label` text is NOT limited to actual
device/icon rows. A real plan's legend region can also contain a supplier's
contact line printed directly under a materials list ("Tel. 041 348 00 60",
"info@ihts.ch", a street address, a postal-code+town, a company name) --
different from the title-block leak `candidates.py`'s three layers guard
against (that guards CROP IMAGES; this data flows as raw TEXT straight into
`classes.json`/`pipeline_summary.json`, both committed artifacts).
`is_plausible_component_label()` runs privacy.py's own regex scan (the SAME
patterns candidates.py's crops are checked against) over every legend label
before it is allowed to count toward -- or ever appear as the NAME of -- a
proposed class. A flagged label is dropped entirely, never partially
redacted, exactly like a PII-flagged candidate crop.

NOISE FILTER -- a real, measured taxonomy-quality problem, not just
privacy: on this batch, `legend_entries.py`'s classifier also picks up
fixture-unit / dimension TABLES (row headers like "kw: 3", "ø22 (9)",
"dn100", "strang -13") as USABLE_TEMPLATE rows alongside genuine device
legend entries, and these occur far more often per plan than any single
device row does. Ranking by raw frequency alone (the original, naive
version of this module) let them dominate every one of the top 15 slots,
crowding out real, task-relevant classes like "rückschlagklappe" or
"sicherheitsgruppe" that a human would recognize immediately.
`is_plausible_component_label()` also rejects labels that are clearly
measurements/identifiers rather than component names (diameter/DN codes,
abbreviation:number table cells, riser labels, mostly-digit fragments,
too-short fragments) using regex patterns and a digit-density check, not a
fixed list of known class names -- it still can't tell a device row from a
room/tenant-use label that happens to be made of ordinary words (e.g. "coop
laden", "putzraum") purely mechanically, so a human reviewer should still
skim the proposed class list in the annotation tool before large-scale
labeling; see docs/v04-dataset-report.md's taxonomy section.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from app.dataset_pipeline.privacy import scan_text
from app.plan_analysis import schema

MIN_SUPPORT = 2
MAX_PROPOSED_CLASSES = 15
FIXED_CLASSES = ["OTHER_RELEVANT_SYMBOL", "NOT_A_COMPONENT", "AMBIGUOUS"]
MAX_NOISE_DIGIT_RATIO = 0.3  # share of non-space characters that may be digits before a label is treated as a measurement/spec, not a component name
MIN_LABEL_ALPHA_CHARS = 3  # rejects single/double-letter table-header fragments ("m", "i", "(mm)")

_DIAMETER_OR_DN_PATTERN = re.compile(r"(ø|\bdn)\s?\d", re.IGNORECASE)
_ABBREV_COLON_NUMBER_PATTERN = re.compile(r"^(lu|du|kw|ww)\s*:\s*", re.IGNORECASE)
_RISER_LABEL_PATTERN = re.compile(r"^strang\s*-?\s*\d+$", re.IGNORECASE)
_MOSTLY_PUNCTUATION_OR_DIGITS_PATTERN = re.compile(r"^[\d\s./\"'\-()°,%]+$")
_TRAILING_INSTANCE_NUMBER_PATTERN = re.compile(r"^(.*\D)\s*\d+$")


def _digit_ratio(label: str) -> float:
    chars = [c for c in label if not c.isspace()]
    return (sum(c.isdigit() for c in chars) / len(chars)) if chars else 0.0


def is_plausible_component_label(label: str) -> bool:
    """See module docstring's "MANDATORY PRIVACY FILTER" and "NOISE FILTER"
    sections for why each check exists. Order matters only for cost (the
    cheap regex/digit checks run before the privacy scan), not correctness
    -- any single match rejects the label."""
    stripped = label.strip()
    if not stripped:
        return False
    if _DIAMETER_OR_DN_PATTERN.search(stripped):
        return False
    if _ABBREV_COLON_NUMBER_PATTERN.match(stripped):
        return False
    if _RISER_LABEL_PATTERN.match(stripped):
        return False
    if _MOSTLY_PUNCTUATION_OR_DIGITS_PATTERN.match(stripped):
        return False
    if _digit_ratio(stripped) > MAX_NOISE_DIGIT_RATIO:
        return False
    if sum(c.isalpha() for c in stripped) < MIN_LABEL_ALPHA_CHARS:
        return False
    if scan_text("_taxonomy_label_check", stripped).has_pii_risk:
        return False
    return True


def normalize_legend_label(label: str) -> str:
    """Strips a trailing standalone instance number ("pex-verteiler 45" ->
    "pex-verteiler") so repeated instances of the same real device family
    count as one class instead of splitting across many near-unique,
    individually-below-MIN_SUPPORT per-instance labels -- observed on this
    batch's plans, where each of several distributors sharing one legend
    icon is individually numbered on the drawing."""
    m = _TRAILING_INSTANCE_NUMBER_PATTERN.match(label)
    return m.group(1).strip() if m else label


@dataclass
class TaxonomyProposal:
    classes: list  # ordered list of class names, FIXED_CLASSES last
    evidence_counts: dict  # {class_name: {"legend": n, "safety_code": n, "generic_drawing": n, "total": n}}

    def to_dict(self) -> dict:
        return {"classes": self.classes, "evidence_counts": self.evidence_counts}


def aggregate_evidence(legend_counts: Counter, safety_code_counts: Counter, generic_counts: Counter) -> dict:
    """The actual ranking/filtering rule, factored out so both
    `propose_taxonomy` (below, given full per-plan doc/legend_result
    objects) and `scripts/run_v04_pipeline.py` (given the same three
    counters built from each plan's own worker-subprocess output, to avoid
    a third re-parse of every plan) apply IDENTICAL support/ranking logic
    rather than two hand-kept-in-sync copies of it. Callers are expected to
    have already run legend-derived label text through
    `is_plausible_component_label`/`normalize_legend_label` before it ever
    reaches this function -- this function only ranks, it does not filter."""
    all_names = set(legend_counts) | set(safety_code_counts) | set(generic_counts)
    evidence_counts = {}
    for name in all_names:
        legend_n, safety_n, generic_n = legend_counts.get(name, 0), safety_code_counts.get(name, 0), generic_counts.get(name, 0)
        total = legend_n + safety_n + generic_n
        if total >= MIN_SUPPORT:
            evidence_counts[name] = {"legend": legend_n, "safety_code": safety_n, "generic_drawing": generic_n, "total": total}
    return evidence_counts


def rank_classes(evidence_counts: dict) -> list:
    ranked = sorted(evidence_counts.items(), key=lambda kv: (-kv[1]["total"], kv[0]))
    return [name for name, _ in ranked[:MAX_PROPOSED_CLASSES]] + FIXED_CLASSES


def propose_taxonomy(per_plan: list[tuple[schema.DocumentAnalysis, dict]]) -> TaxonomyProposal:
    """per_plan: [(doc, legend_intelligence_result), ...] for every
    successfully-parsed plan in the batch."""
    legend_counts: Counter = Counter()
    safety_code_counts: Counter = Counter()
    generic_counts: Counter = Counter()

    for doc, legend_result in per_plan:
        for entry in legend_result.get("legend_entries", []):
            if entry["classification"] == "USABLE_TEMPLATE" and not entry["is_line_style_swatch"]:
                label = entry["normalized_label"].strip()
                if label and is_plausible_component_label(label):
                    legend_counts[normalize_legend_label(label)] += 1

        for page in doc.pages:
            for span in page.text_spans:
                if span.label_hint == "safety_device_code":
                    safety_code_counts[span.text.strip().upper()] += 1

        for fact in legend_result.get("component_facts", []):
            generic_ev = fact.get("generic_evidence")
            if generic_ev and generic_ev.get("symbol_name"):
                generic_counts[generic_ev["symbol_name"].strip().casefold()] += 1

    evidence_counts = aggregate_evidence(legend_counts, safety_code_counts, generic_counts)
    return TaxonomyProposal(classes=rank_classes(evidence_counts), evidence_counts=evidence_counts)
