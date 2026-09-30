"""Cost/call diagnostics -- aggregates CallDiagnostics (schema.py) into a
plan-level summary. Purely arithmetic over already-recorded per-call data;
never estimates or guesses a cost this package cannot actually measure."""
from __future__ import annotations

from collections import defaultdict

from .schema import CallDiagnostics


def summarize(routing_decisions: dict, call_diagnostics: list[CallDiagnostics]) -> dict:
    by_agent: dict = defaultdict(lambda: {
        "routed": False, "route_reason": None, "subjects_routed": 0,
        "model_calls_made": 0, "cached_hits": 0, "unavailable": 0,
    })
    for agent_id, decision in routing_decisions.items():
        by_agent[agent_id]["routed"] = decision.run
        by_agent[agent_id]["route_reason"] = decision.reason
        by_agent[agent_id]["subjects_routed"] = len(decision.subjects)

    for d in call_diagnostics:
        entry = by_agent[d.agent_id]
        if d.called_model and not d.cached:
            entry["model_calls_made"] += 1
        if d.cached:
            entry["cached_hits"] += 1
        if not d.available:
            entry["unavailable"] += 1

    total_model_calls = sum(v["model_calls_made"] for v in by_agent.values())
    total_cached = sum(v["cached_hits"] for v in by_agent.values())
    skipped_agents = [aid for aid, d in routing_decisions.items() if not d.run]

    return {
        "by_agent": dict(by_agent),
        "totals": {
            "agents_routed": sum(1 for d in routing_decisions.values() if d.run),
            "agents_skipped": len(skipped_agents),
            "skipped_agent_ids": sorted(skipped_agents),
            "model_calls_made": total_model_calls,
            "cached_hits": total_cached,
            "subject_level_calls": len(call_diagnostics),
        },
    }
