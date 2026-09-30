"""Deterministic professional-check rule engine -- ENGINE v1 epic, section 6.

IMPORTANT SCOPE NOTE (read before adding a rule): the original GasWaterAI
Water rules/check types and reference cases (generatePruefstellen,
vektorGraphMapper.ts, the Base44 professional-rule layer referenced in this
project's own history) live on the Base44 platform itself, not in any git
repository available to this build session (`qgashi091-ops/GasWaterAI-Engine`
and `qgashi091-ops/gaswaterai` were both checked; neither contains that
code -- see the engine's final report). Every rule in water_rules.py is
therefore a FRESH, conservative implementation grounded only in (a) this
engine's own already-documented deterministic PlanFacts semantics
(app/plan_analysis/plan_facts.py's module docstring) and (b) well-established,
general SVGW/potable-water hygiene principles (e.g. "an unintended
non-circulating loop is a stagnation risk") -- never a specific invented
SVGW clause number. Each rule's `source_reference` says so explicitly where
that is the case, rather than fabricating a citation.
"""
