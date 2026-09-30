"""Conservative potable-water relevance classifier for a legend entry's own
label text -- the ONLY evidence source this module uses, per the domain
expert's explicit instruction that the legend text is the primary signal
and a symbol legend's own naming should never be overridden.

DEFAULT IS EXCLUDE. An entry is included only when its label gives
POSITIVE evidence of naming a potable-water apparatus/fixture or the
potable-water system itself -- the mere absence of an exclusion keyword is
never sufficient ("if potable-water relevance cannot be established, do
not present the symbol as a drinking-water training example").

Exclusion patterns are checked first and always win: a label naming a
wastewater/heating/ventilation/electrical system explicitly is excluded
even if it also happens to contain a positive apparatus word (this never
actually occurs in the current dataset, but the ordering is a deliberate
safety property, not an accident of testing).
"""
from __future__ import annotations

import re

_EXCLUDE_PATTERNS: dict[str, re.Pattern] = {
    "wastewater_drainage": re.compile(
        r"abwasser|schmutzwasser|regenwasser|entw(ä|ae)sserung|drainage|"
        r"fallstrang|fallleitung|kanalisation|sickerleitung|meteorwasser"
    ),
    "insulation": re.compile(r"d(ä|ae)mm|armaflex|isolier"),
    # "heizungs*" as a prefix/compound catches any heating-SYSTEM fitting
    # (Heizungsverteiler, Heizungsfüllventil, Heizungspumpe, ...) even one
    # that water technically flows through -- conservative by design: only
    # an apparatus explicitly named with a potable-water/sanitary term
    # (see _POSITIVE_* below) overcomes this.
    "heating_only": re.compile(
        r"heizk(ö|oe)rper|heizungs\w*|\bradiator\b|\bfussbodenheizung\b|w(ä|ae)rmeerzeuger"
    ),
    "ventilation": re.compile(r"l(ü|ue)ftungsleitung|l(ü|ue)ftungskanal|\babluft\b|\bzuluft\b|entl(ü|ue)ftungsleitung"),
    "electrical": re.compile(r"\belektro\b|\bschalter\b|\bsicherung(?!sventil|sgruppe)\b|\bsteckdose\b|\bkabel\b"),
    "instruction_text": re.compile(r"anschliessen|anschlie(ß|ss)en|bauseits|\bsiehe\b|gem(ä|ae)ss\b|\bvorschrift\b"),
    "dimension_table": re.compile(r"^\s*\d+(\.\d+)?\s*(mm|cm|m)?\s*$|\blu\b|^\s*dn\s*\d+\s*$"),
    "titleblock_or_company": re.compile(
        r"plan-name|massstab|\brev\.|index.*art.*(ä|ae)nderung|gezeichnet|bauseits|"
        r"^\s*-\s*$|^\s*[a-b]\s*$|^\d\.\s*lage|strasse|str\.\s*\d|\bgmbh\b|\bag\b\.?$"
    ),
    # A label ending in "-leitung" (Zuleitung, Steigleitung, Fallleitung,
    # Sammelleitung, Verteilleitung, Anschlussleitung, ...) names a PIPE
    # SEGMENT, not a discrete apparatus -- even when it also contains an
    # apparatus word (e.g. "Wassererwärmer Zuleitung Cr 42" names that
    # heater's OWN supply pipe, not the heater itself as a symbol at this
    # legend position). Confirmed empirically, not just by pattern: of the
    # 387 existing human annotations, the specific label "warmwasserleitung
    # von haus 7 cr 35" was independently found to be NOT_A_COMPONENT 92% of
    # the time (n=12) -- see docs/v04-dataset-cleanup-report.md. Checked
    # AFTER the positive apparatus/potable checks below would otherwise fire,
    # so this always wins over them, matching the conservative default.
    "pipe_segment_label": re.compile(r"\w*leitung\b"),
}

# Compound/specific apparatus words only -- deliberately NOT a bare "ventil"
# or "klappe" (a heating or fire-safety damper/valve would also match those
# generically; see module docstring on staying conservative).
_POSITIVE_APPARATUS = re.compile(
    r"wasserz(ä|ae)hler|wasseruhr|r(ü|ue)ckflussverhinderer|systemtrenner|"
    r"absperr(armatur|ventil|hahn|klappe)|r(ü|ue)ckschlag(ventil|klappe)|"
    r"gartenventil|feinfilter|grobfilter|\bfilter\b|entleer(hahn|ventil)|"
    r"verteiler(batterie)?|batterieventil|misch(ventil|er|batterie)|"
    r"thermostat(isch)?.{0,15}misch|badewannenmisch|duschenmisch|"
    r"zirkulationsventil|zirkulationsregelventil|sicherheitsventil|"
    r"sicherheitsgruppe|druckminderer|wasserenth(ä|ae)rter|boiler|"
    r"wassererw(ä|ae)rmer|waschmaschine|waschtrog|up-ventil|"
    r"hauswasseranschluss|automatischer entl(ü|ue)fter|schnellentl(ü|ue)fter"
)
_POSITIVE_POTABLE_TERM = re.compile(
    r"trinkwasser|kaltwasser|warmwasser|zirkulation|\bkw\b|\bww\b|\bpwc\b|\bpwh\b|"
    r"sanit(ä|ae)r|hausanschluss(?!.*strom)"
)


def normalize_label(raw_label: str) -> str:
    return re.sub(r"\s+", " ", raw_label or "").strip().casefold()


def classify_potable_relevance(raw_label: str) -> dict:
    """Returns {"include": bool, "reason": str}. `reason` is always one of
    the named exclusion categories, "apparatus_terminology",
    "potable_water_terminology", or "relevance_not_established" (the
    conservative default when nothing positive was found)."""
    label = normalize_label(raw_label)
    if not label:
        return {"include": False, "reason": "empty_label"}
    for reason, pattern in _EXCLUDE_PATTERNS.items():
        if pattern.search(label):
            return {"include": False, "reason": f"excluded_{reason}"}
    if _POSITIVE_APPARATUS.search(label):
        return {"include": True, "reason": "apparatus_terminology"}
    if _POSITIVE_POTABLE_TERM.search(label):
        return {"include": True, "reason": "potable_water_terminology"}
    return {"include": False, "reason": "relevance_not_established"}
