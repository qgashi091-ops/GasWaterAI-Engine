"""Canonical inventory -- ENGINE v1 epic, section 5. One stable inventory
built from PlanFacts (deterministic topology, unmodified) + ComponentEvidence
(the new fusion layer). Same deterministic inputs always produce the same
canonical inventory -- this module does no rendering, no randomness, and
reads only fields already computed by plan_facts.py/component_evidence.py.

Connectivity (cycle/dead-end/branch/terminal classification) is taken
EXCLUSIVELY from PlanFacts -- never from the detector or vision fallback
(see app/vision_fallback/interface.py's own contract: vision may name a
component, never its connectivity).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_MEDIUM_PATTERNS = [
    ("Zirkulation", re.compile(r"zirkulation|zirk\.?\b", re.IGNORECASE)),
    ("Warmwasser", re.compile(r"warmwasser|\bww\b", re.IGNORECASE)),
    ("Kaltwasser", re.compile(r"kaltwasser|\bkw\b", re.IGNORECASE)),
]
_RESERVE_PATTERN = re.compile(r"reserve|reserviert|reserveanschluss", re.IGNORECASE)
_SAFETY_DEVICE_PATTERN = re.compile(
    r"sicherheitsventil|r(ü|ue)ckflussverhinderer|r(ü|ue)ckschlagventil|r(ü|ue)ckschlagklappe|"
    r"druckminderer|absperrarmatur|entl(ü|ue)fter",
    re.IGNORECASE,
)

REQUIRED_FIELDS = [
    "component_type", "page", "bbox", "graph_node_ids", "graph_edge_ids",
    "medium", "dimension", "topology_status", "cycle_or_dead_end",
    "reserve_evidence", "safety_device_evidence",
]


@dataclass
class InventoryItem:
    inventory_id: str
    component_type: str | None
    resolution: str
    page: int
    bbox: tuple
    graph_node_ids: list
    graph_edge_ids: list
    medium: str | None
    dimension: str | None
    topology_status: str | None  # "cycle_present" | "acyclic_tree" | None
    cycle_or_dead_end: str | None  # "cycle" | "dead_end" | "branch" | "in_tree" | None
    reserve_evidence: bool
    safety_device_evidence: bool
    provenance: list
    unresolved_fields: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "inventory_id": self.inventory_id, "component_type": self.component_type,
            "resolution": self.resolution, "page": self.page, "bbox": list(self.bbox),
            "graph_node_ids": self.graph_node_ids, "graph_edge_ids": self.graph_edge_ids,
            "medium": self.medium, "dimension": self.dimension,
            "topology_status": self.topology_status, "cycle_or_dead_end": self.cycle_or_dead_end,
            "reserve_evidence": self.reserve_evidence, "safety_device_evidence": self.safety_device_evidence,
            "provenance": self.provenance, "unresolved_fields": self.unresolved_fields,
        }


def _text_near_component(page_model, bbox: tuple, margin: float = 80.0) -> list[str]:
    x0, y0, x1, y1 = bbox
    zone = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    out = []
    for span in page_model.text_spans:
        sx0, sy0, sx1, sy1 = span.bbox
        if sx0 < zone[2] and zone[0] < sx1 and sy0 < zone[3] and zone[1] < sy1:
            out.append(span.text)
    return out


def build_canonical_inventory(doc, plan_facts_result: dict, component_evidence_result: dict) -> dict:
    facts_by_page: dict[int, list[dict]] = {}
    for f in plan_facts_result["facts"]:
        facts_by_page.setdefault(f["page"], []).append(f)

    page_model_by_number = {p.page_number: p for p in doc.pages}

    node_cycle_membership: dict[tuple, bool] = {}
    node_dead_end: dict[tuple, bool] = {}
    node_branch: dict[tuple, bool] = {}
    dimension_by_edge: dict[tuple, str] = {}
    component_id_by_node: dict[tuple, str] = {}

    for page, facts in facts_by_page.items():
        for f in facts:
            if f["fact_type"] == "cycle_membership" and f["value"]:
                for n in f["supporting_nodes"]:
                    node_cycle_membership[(page, n)] = True
            if f["fact_type"] == "dead_end_path":
                for n in f["supporting_nodes"]:
                    node_dead_end[(page, n)] = True
            if f["fact_type"] == "branch" and f["value"]:
                for n in f["supporting_nodes"]:
                    node_branch[(page, n)] = True
            if f["fact_type"] == "dimension_evidence":
                for e in f["supporting_edges"]:
                    dimension_by_edge[(page, e)] = f["value"]

    components_by_page = {c["page"]: c for c in plan_facts_result["components"]}
    topology_status_by_node: dict[tuple, str] = {}
    for comp in plan_facts_result["components"]:
        for n in comp["node_ids"]:
            topology_status_by_node[(comp["page"], n)] = comp["topology_status"]

    items = []
    for ev in component_evidence_result["evidence"]:
        page = ev["page"]
        page_model = page_model_by_number.get(page)

        graph_assoc = ev.get("graph_association") or {}
        node_ids = list(graph_assoc.get("node_ids", []))
        edge_ids = list(graph_assoc.get("edge_ids", []))

        cycle_or_dead_end = None
        topology_status = None
        if node_ids:
            if any(node_cycle_membership.get((page, n)) for n in node_ids):
                cycle_or_dead_end = "cycle"
            elif any(node_dead_end.get((page, n)) for n in node_ids):
                cycle_or_dead_end = "dead_end"
            elif any(node_branch.get((page, n)) for n in node_ids):
                cycle_or_dead_end = "branch"
            else:
                cycle_or_dead_end = "in_tree"
            statuses = {topology_status_by_node.get((page, n)) for n in node_ids} - {None}
            if len(statuses) == 1:
                topology_status = next(iter(statuses))

        nearby_text = _text_near_component(page_model, tuple(ev["bbox"])) if page_model is not None else []
        nearby_text_joined = " ".join(nearby_text)

        medium = None
        for name, pattern in _MEDIUM_PATTERNS:
            if pattern.search(nearby_text_joined):
                medium = name
                break

        dimension = None
        for eid in edge_ids:
            if (page, eid) in dimension_by_edge:
                dimension = dimension_by_edge[(page, eid)]
                break
        # A component at_endpoint/at_branch (node_ids only, no edge_ids) has
        # no single incident edge to attribute a dimension to unambiguously
        # -- dimension stays unresolved rather than guessed, consistent with
        # v0.1's own documented scope limit (plan_facts.py never propagates
        # a dimension across multiple edges either).

        reserve_evidence = bool(_RESERVE_PATTERN.search(nearby_text_joined))
        safety_device_evidence = bool(_SAFETY_DEVICE_PATTERN.search(nearby_text_joined))

        unresolved = []
        if ev["component_type"] is None:
            unresolved.append("component_type")
        if medium is None:
            unresolved.append("medium")
        if dimension is None:
            unresolved.append("dimension")
        if cycle_or_dead_end is None:
            unresolved.append("cycle_or_dead_end")

        items.append(InventoryItem(
            inventory_id=ev["component_evidence_id"], component_type=ev["component_type"],
            resolution=ev["resolution"], page=page, bbox=tuple(ev["bbox"]),
            graph_node_ids=node_ids, graph_edge_ids=edge_ids, medium=medium, dimension=dimension,
            topology_status=topology_status, cycle_or_dead_end=cycle_or_dead_end,
            reserve_evidence=reserve_evidence, safety_device_evidence=safety_device_evidence,
            provenance=ev["provenance"], unresolved_fields=unresolved,
        ))

    return {
        "inventory": [i.to_dict() for i in items],
        "stats": {
            "item_count": len(items),
            "by_resolution": {
                k: sum(1 for i in items if i.resolution == k)
                for k in ("COMPONENT_FACT", "COMPONENT_SUPPORTED", "COMPONENT_UNRESOLVED")
            },
            "with_medium_identified": sum(1 for i in items if i.medium),
            "with_dimension_identified": sum(1 for i in items if i.dimension),
            "with_reserve_evidence": sum(1 for i in items if i.reserve_evidence),
            "with_safety_device_evidence": sum(1 for i in items if i.safety_device_evidence),
        },
    }
