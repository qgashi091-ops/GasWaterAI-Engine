"""Initial rule set -- see app/rules/__init__.py for the scope caveat these
rules operate under (no access to the original SVGW/Base44 rule corpus in
this session). Covers the capability families section 7 prioritizes:
explicit-text-based, topology/cycle, reserve-text/topology, and component-
identity checks. Dimension/real-world-length checks are represented by one
rule that is honestly always NOT_ASSESSABLE (this engine has no scale
calibration -- see RESERVE_LENGTH_NOTE) rather than silently absent.

Do not invent a professional requirement in this module. Every COMPLIANT/
NON_COMPLIANT branch below is defensible from either (a) this engine's own
documented PlanFacts semantics or (b) a general, widely-documented SVGW
hygiene principle named honestly as such, never a specific fabricated
clause number.
"""
from __future__ import annotations

from .engine import Rule

GENERAL_HYGIENE_NOTE = (
    "General SVGW/potable-water hygiene principle (avoid unintended, "
    "non-circulating stagnation loops); exact SVGW clause reference not "
    "available in this session -- the original rule corpus is not "
    "accessible from this repository (see engine final report)."
)
STRUCTURAL_NOTE = (
    "Structural corroboration of plan text against drawn topology, per "
    "this engine's own deterministic PlanFacts semantics "
    "(app/plan_analysis/plan_facts.py module docstring); not a cited SVGW "
    "clause."
)
NO_SCALE_CALIBRATION_NOTE = (
    "This engine computes topology in PDF/vector coordinate space only -- "
    "no real-world scale calibration exists yet (no rule anywhere in this "
    "codebase converts drawn length to metres), so any length-dependent "
    "check is unconditionally NOT_ASSESSABLE until that capability exists."
)


def _unintended_loop_evaluate(item: dict):
    medium = item.get("medium")
    if medium is None:
        return "NOT_ASSESSABLE", ["medium"], {"cycle_or_dead_end": item["cycle_or_dead_end"]}
    if medium == "Zirkulation":
        return "COMPLIANT", [], {"medium": medium, "reason": "labeled circulation loop -- expected topology"}
    return "NON_COMPLIANT", [], {"medium": medium, "reason": "closed loop on a non-circulation-labeled line -- stagnation-risk configuration"}


UNINTENDED_LOOP_RULE = Rule(
    rule_id="RULE-001-unintended-loop",
    name="Unintended non-circulating loop (stagnation risk)",
    required_facts=["cycle_or_dead_end", "medium"],
    source_reference=GENERAL_HYGIENE_NOTE,
    applicability=lambda item: item.get("cycle_or_dead_end") == "cycle",
    evaluate=_unintended_loop_evaluate,
)


def _dead_end_length_evaluate(item: dict):
    return "NOT_ASSESSABLE", ["real_world_length", "scale_calibration"], {"cycle_or_dead_end": item["cycle_or_dead_end"]}


DEAD_END_LENGTH_RULE = Rule(
    rule_id="RULE-002-dead-end-stagnation-length",
    name="Dead-end stub length (stagnation risk) -- not yet supported",
    required_facts=["real_world_length"],
    source_reference=NO_SCALE_CALIBRATION_NOTE,
    applicability=lambda item: item.get("cycle_or_dead_end") == "dead_end",
    evaluate=_dead_end_length_evaluate,
)


def _reserve_connection_evaluate(item: dict):
    cde = item.get("cycle_or_dead_end")
    if cde == "dead_end":
        return "COMPLIANT", [], {"cycle_or_dead_end": cde, "reason": "reserve connection drawn as an unconnected stub, as expected"}
    return "NOT_ASSESSABLE", ["cycle_or_dead_end"] if cde is None else [], {
        "cycle_or_dead_end": cde,
        "reason": "text names a reserve connection but drawn topology does not clearly corroborate an unconnected stub",
    }


RESERVE_CONNECTION_RULE = Rule(
    rule_id="RULE-003-reserve-connection",
    name="Explicit reserve connection corroborated by topology",
    required_facts=["reserve_evidence", "cycle_or_dead_end"],
    source_reference=STRUCTURAL_NOTE,
    applicability=lambda item: item.get("reserve_evidence") is True,
    evaluate=_reserve_connection_evaluate,
)


def _safety_device_identity_evaluate(item: dict):
    resolution = item["resolution"]
    if resolution == "COMPONENT_FACT":
        return "COMPLIANT", [], {"resolution": resolution, "component_type": item.get("component_type")}
    return "NOT_ASSESSABLE", ["component_type"], {
        "resolution": resolution,
        "reason": "text names a safety device but component identity is not confidently resolved",
    }


SAFETY_DEVICE_IDENTITY_RULE = Rule(
    rule_id="RULE-004-safety-device-identity",
    name="Named safety device has confidently resolved component identity",
    required_facts=["safety_device_evidence", "resolution", "component_type"],
    source_reference=STRUCTURAL_NOTE,
    applicability=lambda item: item.get("safety_device_evidence") is True,
    evaluate=_safety_device_identity_evaluate,
)


ALL_RULES = [UNINTENDED_LOOP_RULE, DEAD_END_LENGTH_RULE, RESERVE_CONNECTION_RULE, SAFETY_DEVICE_IDENTITY_RULE]
