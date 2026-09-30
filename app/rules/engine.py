"""Rule engine framework: a Rule declares required facts, applicability, and
compliant/non-compliant/insufficient-evidence conditions; the engine runs
every rule against every inventory item and never invents a result the rule
itself didn't produce. LLM prose is explicitly out of scope for this module
(section 6: "LLM may generate explanatory prose AFTER the deterministic
result, but may never change the result") -- this module produces only the
deterministic result; any prose generation is a separate, later concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Literal

RuleResultValue = Literal["COMPLIANT", "NON_COMPLIANT", "NOT_ASSESSABLE"]


@dataclass
class Rule:
    rule_id: str
    name: str
    required_facts: list[str]  # canonical-inventory field names this rule reads
    source_reference: str
    applicability: Callable[[dict], bool]  # (inventory_item) -> bool
    evaluate: Callable[[dict], tuple[RuleResultValue, list[str], dict]]  # -> (result, missing_facts, evidence)


@dataclass
class CheckResult:
    check_id: str
    rule_id: str
    result: RuleResultValue
    facts_used: list[str]
    missing_facts: list[str]
    evidence: dict
    source_reference: str
    inventory_id: str

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id, "rule_id": self.rule_id, "result": self.result,
            "facts_used": self.facts_used, "missing_facts": self.missing_facts,
            "evidence": self.evidence, "source_reference": self.source_reference,
            "inventory_id": self.inventory_id,
        }


def _check_id(rule_id: str, inventory_id: str) -> str:
    return "CK" + sha256(f"{rule_id}|{inventory_id}".encode("utf-8")).hexdigest()[:20]


def run_checks(inventory: list[dict], rules: list[Rule]) -> dict:
    results: list[CheckResult] = []
    for item in inventory:
        for rule in rules:
            if not rule.applicability(item):
                continue
            result_value, missing_facts, evidence = rule.evaluate(item)
            facts_used = [f for f in rule.required_facts if f not in missing_facts]
            results.append(CheckResult(
                check_id=_check_id(rule.rule_id, item["inventory_id"]), rule_id=rule.rule_id,
                result=result_value, facts_used=facts_used, missing_facts=missing_facts,
                evidence=evidence, source_reference=rule.source_reference, inventory_id=item["inventory_id"],
            ))

    counts = {"COMPLIANT": 0, "NON_COMPLIANT": 0, "NOT_ASSESSABLE": 0}
    for r in results:
        counts[r.result] += 1

    return {
        "checks": [r.to_dict() for r in results],
        "stats": {
            "rules_registered": len(rules), "checks_run": len(results), "by_result": counts,
            "by_rule": {
                rule.rule_id: {
                    "applicable_items": sum(1 for r in results if r.rule_id == rule.rule_id),
                    "by_result": {
                        v: sum(1 for r in results if r.rule_id == rule.rule_id and r.result == v)
                        for v in ("COMPLIANT", "NON_COMPLIANT", "NOT_ASSESSABLE")
                    },
                }
                for rule in rules
            },
        },
    }
