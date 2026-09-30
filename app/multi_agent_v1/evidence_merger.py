"""EvidenceMerger -- a plain deterministic function module (not an agent,
not a chief/super-agent: it never calls a model and never invents a value).
Its ENTIRE job is priority + provenance + conflict bookkeeping over values
that some other, already-tested module produced:

    DETERMINISTIC_FACT (resolved)  >  AGENT_OBSERVATION

A resolved DeterministicFactRef always wins outright; competing agent
observations are recorded as provenance/conflict entries but NEVER change
the winning value. Only when there is no resolved deterministic fact for a
(claim_type, subject_id) pair does an agent observation's value become the
merged claim's value -- and even then, disagreement between agents is
reported as a ConflictRecord, never silently resolved by picking one
arbitrarily without saying so (same "never hide a disagreement" discipline
already established in component_evidence.py's `_fuse_one`).
"""
from __future__ import annotations

import json
from collections import defaultdict

from .context import PlanAgentContext
from .schema import AgentObservation, ConflictRecord, MergedClaim


def _value_key(value: object) -> str:
    """Stable equality key for possibly-dict/list-valued observations."""
    return json.dumps(value, sort_keys=True, default=str)


class EvidenceMerger:
    def merge(self, context: PlanAgentContext, observations: list[AgentObservation]) -> list[MergedClaim]:
        obs_by_pair: dict[tuple[str, str], list[AgentObservation]] = defaultdict(list)
        for obs in observations:
            obs_by_pair[(obs.claim_type, obs.subject_id)].append(obs)
        # Sort each subject's observations deterministically (never by input/
        # call order) so provenance lists, conflict-record ordering, and the
        # tie-broken representative value are all identical regardless of
        # what order the pipeline happened to invoke agents in this run.
        for key in obs_by_pair:
            obs_by_pair[key] = sorted(obs_by_pair[key], key=lambda o: (o.agent_id, _value_key(o.value)))

        det_pairs: dict[tuple[str, str], list] = defaultdict(list)
        for ref in context.all_fact_refs():
            det_pairs[(ref.claim_type, ref.subject_id)].append(ref)

        all_pairs = set(obs_by_pair) | set(det_pairs)
        claims: list[MergedClaim] = []

        for claim_type, subject_id in sorted(all_pairs):
            det_refs = det_pairs.get((claim_type, subject_id), [])
            agent_obs = obs_by_pair.get((claim_type, subject_id), [])
            resolved_refs = [r for r in det_refs if r.resolved]
            available_obs = [o for o in agent_obs if o.available]

            provenance: list[dict] = []
            conflicts: list[ConflictRecord] = []

            if resolved_refs:
                winner = resolved_refs[0]
                provenance.append({"kind": "DETERMINISTIC_FACT", "source": winner.source_module, "fact_id": winner.fact_id})
                for extra in resolved_refs[1:]:
                    provenance.append({"kind": "DETERMINISTIC_FACT", "source": extra.source_module, "fact_id": extra.fact_id, "role": "corroboration"})
                disagreeing = [o for o in available_obs if _value_key(o.value) != _value_key(winner.value)]
                for o in available_obs:
                    provenance.append({"kind": "AGENT_OBSERVATION", "source": o.agent_id, "role": "corroboration" if o not in disagreeing else "conflict_never_decisive"})
                if disagreeing:
                    conflicts.append(ConflictRecord(
                        claim_type=claim_type, subject_id=subject_id,
                        description="One or more agent observations disagreed with the resolved deterministic fact; the deterministic fact always wins and was never overridden.",
                        competing_values=[{"source": winner.source_module, "value": winner.value, "kind": "DETERMINISTIC_FACT"}]
                        + [{"source": o.agent_id, "value": o.value, "kind": "AGENT_OBSERVATION"} for o in disagreeing],
                    ))
                claims.append(MergedClaim(claim_type, subject_id, winner.value, "DETERMINISTIC", provenance, conflicts))
                continue

            if not available_obs:
                for r in det_refs:
                    provenance.append({"kind": "DETERMINISTIC_FACT", "source": r.source_module, "fact_id": r.fact_id, "role": "unresolved"})
                for o in agent_obs:
                    provenance.append({"kind": "AGENT_OBSERVATION", "source": o.agent_id, "role": "unavailable", "error": o.error})
                claims.append(MergedClaim(claim_type, subject_id, None, "UNRESOLVED", provenance, []))
                continue

            distinct_values = {_value_key(o.value) for o in available_obs}
            for o in available_obs:
                provenance.append({"kind": "AGENT_OBSERVATION", "source": o.agent_id, "confidence": o.confidence})

            if len(distinct_values) == 1:
                resolution = "AGENT_AGREED" if len(available_obs) > 1 else "AGENT_SINGLE"
                claims.append(MergedClaim(claim_type, subject_id, available_obs[0].value, resolution, provenance, []))
                continue

            # Genuine disagreement among agents, no deterministic tiebreaker
            # available: report it, never hide it. Deterministic, INPUT-
            # ORDER-INDEPENDENT tie-break for the representative value:
            # prefer a "supported"-confidence observation, else break ties
            # on (agent_id, value) -- never on agent_id alone, which fails
            # to discriminate (and so falls back to Python's stable-sort
            # input order) whenever the same agent produced more than one
            # competing observation for the same subject.
            supported = [o for o in available_obs if o.confidence == "supported"]
            pool = supported if len(supported) == 1 else available_obs
            chosen = sorted(pool, key=lambda o: (o.agent_id, _value_key(o.value)))[0]
            conflicts.append(ConflictRecord(
                claim_type=claim_type, subject_id=subject_id,
                description="Agent observations disagreed with no resolved deterministic fact to arbitrate; a representative value was chosen deterministically but the disagreement is preserved here.",
                competing_values=[{"source": o.agent_id, "value": o.value, "kind": "AGENT_OBSERVATION"} for o in available_obs],
            ))
            claims.append(MergedClaim(claim_type, subject_id, chosen.value, "AGENT_CONFLICT", provenance, conflicts))

        return claims
