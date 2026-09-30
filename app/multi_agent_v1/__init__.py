"""GasWaterAI Engine -- Multi-Agent Architecture v1.

Additive, parallel to the existing engine: nothing here is imported by
app/main.py's existing `/analyze` or `/check` handlers, existing PlanFacts
topology code, or the golden-holdout harness. See app/main.py's own
`/multi_agent_v1/*` router mount and scripts/run_multi_agent_v1.py for the
separate API/CLI test path this epic asked for.

Core rule, enforced in code (not just documented): DETERMINISTIC_FACT always
outranks AGENT_OBSERVATION (see schema.py, evidence_merger.py). There is no
super-/chief-agent anywhere in this package.
"""
from .pipeline import ENGINE_VERSION, run_multi_agent_v1

__all__ = ["ENGINE_VERSION", "run_multi_agent_v1"]
