"""PlanFacts -- deterministic topology derivation (new in this engine; there
is no equivalent module in the original GasWaterAI parser repo, which stops
at graph/symbol/text extraction and never computes cycles, dead-ends, or any
other topological classification -- see docs/architecture.md).

PROBLEM THIS SOLVES: on the Base44 application, the same W-003 crop, same
plan, same question was classified by a multimodal LLM as "Dead-End" /
"Dead-End" / "Loop" across three otherwise-identical runs, each reported at
high confidence. This module answers the same class of question --
cycle-vs-dead-end -- directly from the vector graph the parser already
extracts deterministically, so the LLM's free-form topology opinion is never
the source of truth for anything this module can prove from geometry alone.

ARCHITECTURAL RULE (non-negotiable): every emitted fact's `kind` is exactly
one of "FACT", "DERIVED_FACT", "UNRESOLVED" -- never a guess. A bridge
(reconstructed) edge can be recorded, but it never silently becomes evidence
for a directly-drawn claim (see bridging.py's own docstring for why: it is a
conservative geometric *inference*, not a drawn line).

============================================================================
SEMANTICS (read before changing this module)
============================================================================
- Direct edge: GraphEdge.is_bridge is False -- corresponds to an actually
  drawn vector segment. Sole basis for every geometric fact here.
- Bridge edge: GraphEdge.is_bridge is True -- reconstructed by bridging.py,
  never drawn. Reported only as diagnostic "possible continuity" evidence;
  never contributes to a cycle or dead-end fact.
- Directly-drawn subgraph: the graph restricted to direct edges. Every
  topology fact in this module is computed exclusively over this subgraph,
  per page (topology is never inferred across pages or files).
- Connected component: a maximal node set reachable via direct edges only.
- Degree (degree_direct): count of INCIDENT DIRECT EDGES ONLY. Bridge edges
  are deliberately excluded -- pipeline.py's own bridge_graph() increments
  GraphNode.degree for a bridged node (see graph.py/bridging.py), so trusting
  that field here would let a reconstructed connection manufacture a branch
  or hide one. This module always recomputes degree from scratch.
- Terminal endpoint: a node with degree_direct == 1 and NO incident bridge
  edge, classified PHYSICAL_END / DEVICE_BOUNDARY / DISTRIBUTOR_BOUNDARY.
  A degree-1 node touched by ANY bridge is always UNRESOLVED, regardless of
  what the classifier would otherwise say -- a bridge implies a possible,
  unproven continuation, so no confident "this pipe truly ends here" claim
  is warranted.
- Branch: a node with degree_direct >= 3.
- Cycle membership -- PURELY GRAPH-THEORETIC, NOT a hydraulic or SVGW
  compliance claim: an edge/node belongs to a cycle iff it lies on some
  closed walk of >=1 direct edges (no edge repeated) within its component.
  This says only "these drawn segments form a closed ring on this sheet."
  It says nothing about whether water actually recirculates, whether it is
  an intentional SVGW "Schlaufung", or whether it is compliant -- that
  interpretation is explicitly out of scope for this engine (and for
  Base44's professional-rule layer, not duplicated here).
- Dead-end/stub path: a maximal run of direct edges from a proven terminal
  leaf inward to a branch node or the boundary of a cycle. Purely structural
  ("this drawn run ends here with no further drawn continuation"), never a
  stagnation-risk or 4xID claim.
- Unresolved: a first-class, terminal answer -- not an error -- whenever a
  bridge or an unclassifiable endpoint prevents proof. Never guessed.
============================================================================
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from . import schema

FactKind = Literal["FACT", "DERIVED_FACT", "UNRESOLVED"]
EvidenceStatus = Literal["directly_drawn", "reconstructed", "unresolved"]
TerminalClass = Literal["PHYSICAL_END", "DEVICE_BOUNDARY", "DISTRIBUTOR_BOUNDARY", "UNRESOLVED"]

# Same conceptual category as the (independently maintained) Base44
# application's fokusRegionen.ts / vektorGraphMapper.ts -- reused as DESIGN,
# not as code: this is a fresh Python codebase with no dependency on the
# TypeScript application (see docs/architecture.md).
_DISTRIBUTOR_PATTERN = re.compile(r"verteiler|steigzone|steigleitung|verteilerbatterie|sammler", re.IGNORECASE)
_APPARATUS_PATTERN = re.compile(r"apparat|ger[aä]t|armaturengruppe|anschlussgruppe", re.IGNORECASE)


def _fact_id(*parts: str) -> str:
    """Stable, content-derived id -- never a counter, so identical input
    always yields an identical id regardless of processing order."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"PF{digest}"


@dataclass
class PlanFact:
    fact_id: str
    fact_type: str
    kind: FactKind
    value: object
    source: str
    evidence_status: EvidenceStatus
    supporting_edges: list[str] = field(default_factory=list)
    supporting_nodes: list[str] = field(default_factory=list)
    page: Optional[int] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "kind": self.kind,
            "value": self.value,
            "source": self.source,
            "evidence_status": self.evidence_status,
            "supporting_edges": sorted(self.supporting_edges),
            "supporting_nodes": sorted(self.supporting_nodes),
            "page": self.page,
            "detail": self.detail,
        }


class _UnionFind:
    def __init__(self, ids: list[str]):
        self.parent = {i: i for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic tie-break (lexicographically smaller root wins) --
        # independent of the order edges are processed in.
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _classify_endpoint(node_id: str, incident, symbol_by_port: dict, text_by_edge: dict) -> TerminalClass:
    if node_id in symbol_by_port:
        return "DEVICE_BOUNDARY"
    edge = incident[0][0] if incident else None
    texts = text_by_edge.get(edge.id, []) if edge else []
    if any(_DISTRIBUTOR_PATTERN.search(t) or _APPARATUS_PATTERN.search(t) for t in texts):
        return "DISTRIBUTOR_BOUNDARY"
    if edge is None:
        return "UNRESOLVED"
    return "PHYSICAL_END"


def _tree_path(u: str, v: str, parent: dict) -> list[str]:
    def path_to_root(n: str) -> list[str]:
        out = [n]
        while n in parent:
            n = parent[n]
            out.append(n)
        return out

    pu, pv = path_to_root(u), path_to_root(v)
    pu_set = set(pu)
    lca = next((n for n in pv if n in pu_set), None)
    if lca is None:
        return [u, v]
    u_side = pu[: pu.index(lca) + 1]
    v_side = pv[: pv.index(lca) + 1]
    return u_side + list(reversed(v_side[:-1]))


def build_page_facts(page: schema.PageAnalysis) -> tuple[list[PlanFact], dict]:
    """Returns (facts, structural_summary) for a single page. Pure function
    of the page's own graph/symbols/text/associations -- deterministic,
    O(V+E), no PDF/raster access."""
    page_key = f"p{page.page_number}"
    nodes = page.graph.nodes
    edges = page.graph.edges
    node_ids = sorted(n.id for n in nodes)
    point_by_id = {n.id: n.point for n in nodes}

    direct_edges = sorted([e for e in edges if not e.is_bridge], key=lambda e: e.id)
    bridge_edges = sorted([e for e in edges if e.is_bridge], key=lambda e: e.id)

    degree_direct: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[tuple]] = {nid: [] for nid in node_ids}
    self_loops = []
    for e in direct_edges:
        if e.source == e.target:
            self_loops.append(e)
            continue
        degree_direct[e.source] = degree_direct.get(e.source, 0) + 1
        degree_direct[e.target] = degree_direct.get(e.target, 0) + 1
        adjacency.setdefault(e.source, []).append((e, e.target))
        adjacency.setdefault(e.target, []).append((e, e.source))
    self_loop_nodes = {e.source for e in self_loops}

    bridge_incident: set[str] = set()
    for e in bridge_edges:
        bridge_incident.add(e.source)
        bridge_incident.add(e.target)

    symbol_by_port: dict[str, schema.SymbolCandidate] = {}
    for s in page.symbols:
        for pid in s.port_node_ids:
            symbol_by_port[pid] = s
    span_by_id = {t.id: t for t in page.text_spans}
    text_by_edge: dict[str, list[str]] = {}
    for a in page.associations:
        if a.target_type != "edge" or not a.target_id:
            continue
        span = span_by_id.get(a.text_id)
        if span is None:
            continue
        text_by_edge.setdefault(a.target_id, []).append(span.text)

    facts: list[PlanFact] = []

    # --- FACT tier: base geometric evidence (edges/text as drawn/extracted) ---
    for e in direct_edges:
        facts.append(PlanFact(
            fact_id=_fact_id("edge", page_key, e.id), fact_type="pipe_segment", kind="FACT",
            value=True, source="vector_graph", evidence_status="directly_drawn",
            supporting_edges=[e.id], supporting_nodes=[e.source, e.target], page=page.page_number,
            detail={"length": e.length, "color_rgb": list(e.color_rgb) if e.color_rgb else None},
        ))
    for e in bridge_edges:
        facts.append(PlanFact(
            fact_id=_fact_id("bridge", page_key, e.id), fact_type="pipe_segment", kind="UNRESOLVED",
            value=None, source="vector_graph_bridging", evidence_status="reconstructed",
            supporting_edges=[e.id], supporting_nodes=[e.source, e.target], page=page.page_number,
            detail={"bridge_reason": e.bridge_reason},
        ))
    # Dimension evidence: only text DIRECTLY associated (association.py) to an
    # edge and already pattern-matched as a nominal-diameter callout
    # (label_hints.py) is emitted as a FACT here. Propagating a symbol's
    # dimension label onto multiple nearby edges (as the Base44 application's
    # vektorGraphMapper.ts does, with dedicated ambiguity handling) is
    # explicitly out of scope for this v0.1 POC -- see docs/architecture.md.
    for a in page.associations:
        if a.target_type != "edge" or not a.target_id:
            continue
        span = span_by_id.get(a.text_id)
        if span is None or span.label_hint != "nominal_diameter":
            continue
        facts.append(PlanFact(
            fact_id=_fact_id("dimension", page_key, a.target_id, span.id), fact_type="dimension_evidence",
            kind="FACT", value=span.text, source="text_layer", evidence_status="directly_drawn",
            supporting_edges=[a.target_id], supporting_nodes=[], page=page.page_number,
            detail={"label_hint": span.label_hint, "distance": a.distance},
        ))

    # --- Connected components (direct edges only) ---
    uf = _UnionFind(node_ids)
    for e in direct_edges:
        if e.source != e.target:
            uf.union(e.source, e.target)

    cycle_edge_ids: set[str] = set()
    cycle_node_ids: set[str] = set()
    for e in self_loops:
        cycle_edge_ids.add(e.id)
        cycle_node_ids.add(e.source)

    members_by_root: dict[str, list[str]] = {}
    for nid in node_ids:
        members_by_root.setdefault(uf.find(nid), []).append(nid)

    component_summaries: list[dict] = []
    for root in sorted(members_by_root, key=lambda r: sorted(members_by_root[r])[0]):
        members = sorted(members_by_root[root])
        comp_edges = sorted([e for e in direct_edges if e.source != e.target and uf.find(e.source) == root], key=lambda e: e.id)
        v, e_count = len(members), len(comp_edges)
        cyclomatic = e_count - v + 1
        comp_id = _fact_id("component", page_key, ",".join(members))

        # Deterministic spanning tree (BFS from the smallest node id;
        # neighbors visited in sorted edge-id order) -> back edges mark
        # cycle membership via their tree-path to the least common ancestor.
        parent: dict[str, str] = {}
        parent_edge: dict[str, str] = {}
        tree_edge_ids: set[str] = set()
        visited = {members[0]}
        queue = [members[0]]
        while queue:
            u = queue.pop(0)
            for e, other in sorted(adjacency.get(u, []), key=lambda x: x[0].id):
                if other in visited:
                    continue
                visited.add(other)
                parent[other] = u
                parent_edge[other] = e.id
                tree_edge_ids.add(e.id)
                queue.append(other)
        back_edges = sorted([e for e in comp_edges if e.id not in tree_edge_ids], key=lambda e: e.id)
        for be in back_edges:
            path = _tree_path(be.source, be.target, parent)
            for n in path:
                cycle_node_ids.add(n)
            cycle_edge_ids.add(be.id)
            for a, b in zip(path, path[1:]):
                eid = parent_edge.get(a) if parent.get(a) == b else parent_edge.get(b)
                if eid:
                    cycle_edge_ids.add(eid)

        status = "cycle_present" if cyclomatic > 0 else "acyclic_tree"
        component_summaries.append({
            "component_id": comp_id, "page": page.page_number, "node_ids": members,
            "edge_ids": [e.id for e in comp_edges], "node_count": v, "direct_edge_count": e_count,
            "cyclomatic_number": cyclomatic, "topology_status": status,
        })

        # DERIVED_FACT tier: connectivity/degree/cycle are derived FROM the
        # base pipe_segment FACTs above, never asserted independently.
        facts.append(PlanFact(
            fact_id=_fact_id("connected_component", page_key, ",".join(members)), fact_type="connected_component",
            kind="DERIVED_FACT", value=members, source="vector_graph", evidence_status="directly_drawn",
            supporting_edges=[e.id for e in comp_edges], supporting_nodes=members, page=page.page_number,
        ))
        facts.append(PlanFact(
            fact_id=_fact_id("cycle_membership", page_key, comp_id), fact_type="cycle_membership",
            kind="DERIVED_FACT", value=cyclomatic > 0, source="vector_graph", evidence_status="directly_drawn",
            supporting_edges=[e.id for e in comp_edges if e.id in cycle_edge_ids] if cyclomatic > 0 else [e.id for e in comp_edges],
            supporting_nodes=[n for n in members if n in cycle_node_ids] if cyclomatic > 0 else members,
            page=page.page_number, detail={"cyclomatic_number": cyclomatic, "component_id": comp_id},
        ))

    component_id_by_node = {n: c["component_id"] for c in component_summaries for n in c["node_ids"]}

    # --- Per-node degree/branch/terminal DERIVED_FACTs ---
    for nid in node_ids:
        deg = degree_direct.get(nid, 0)
        facts.append(PlanFact(
            fact_id=_fact_id("node_degree", page_key, nid), fact_type="node_degree", kind="DERIVED_FACT",
            value=deg, source="vector_graph", evidence_status="directly_drawn",
            supporting_edges=[e.id for e, _ in adjacency.get(nid, [])], supporting_nodes=[nid], page=page.page_number,
        ))
        if nid in self_loop_nodes:
            # A self-loop already makes this node a (trivial) cycle member --
            # it is never additionally classified as a branch or a terminal
            # endpoint, regardless of its direct-edge degree.
            continue
        if deg >= 3:
            facts.append(PlanFact(
                fact_id=_fact_id("branch", page_key, nid), fact_type="branch", kind="DERIVED_FACT",
                value=True, source="vector_graph", evidence_status="directly_drawn",
                supporting_edges=[e.id for e, _ in adjacency.get(nid, [])], supporting_nodes=[nid], page=page.page_number,
            ))
        if deg == 1:
            has_bridge = nid in bridge_incident
            terminal_class: TerminalClass = "UNRESOLVED" if has_bridge else _classify_endpoint(
                nid, adjacency.get(nid, []), symbol_by_port, text_by_edge)
            kind: FactKind = "UNRESOLVED" if terminal_class == "UNRESOLVED" else "DERIVED_FACT"
            evidence: EvidenceStatus = "unresolved" if terminal_class == "UNRESOLVED" else "directly_drawn"
            facts.append(PlanFact(
                fact_id=_fact_id("terminal_endpoint", page_key, nid), fact_type="terminal_endpoint", kind=kind,
                value=None if terminal_class == "UNRESOLVED" else terminal_class, source="vector_graph",
                evidence_status=evidence, supporting_edges=[e.id for e, _ in adjacency.get(nid, [])],
                supporting_nodes=[nid], page=page.page_number,
                detail={"bridge_adjacent": has_bridge},
            ))

    # --- Dead-end/stub paths ---
    non_cycle_adjacency = {
        nid: [(e, other) for e, other in adjacency.get(nid, []) if e.id not in cycle_edge_ids]
        for nid in node_ids
    }
    leaves = sorted(
        nid for nid in node_ids
        if degree_direct.get(nid, 0) == 1 and nid not in cycle_node_ids and nid not in self_loop_nodes
    )
    for leaf in leaves:
        neighbors = non_cycle_adjacency.get(leaf, [])
        if len(neighbors) != 1:
            continue
        has_bridge = leaf in bridge_incident
        terminal_class: TerminalClass = "UNRESOLVED" if has_bridge else _classify_endpoint(
            leaf, adjacency.get(leaf, []), symbol_by_port, text_by_edge)
        hop_edge, cur = neighbors[0]
        path_nodes = [leaf]
        path_edges: list[str] = []
        while True:
            path_edges.append(hop_edge.id)
            path_nodes.append(cur)
            if cur in cycle_node_ids:
                break
            options = [(e, o) for e, o in non_cycle_adjacency.get(cur, []) if e.id != hop_edge.id]
            if len(options) != 1:
                break
            hop_edge, cur = options[0]
        kind: FactKind = "UNRESOLVED" if terminal_class == "UNRESOLVED" else "DERIVED_FACT"
        evidence: EvidenceStatus = "unresolved" if terminal_class == "UNRESOLVED" else "directly_drawn"
        facts.append(PlanFact(
            fact_id=_fact_id("dead_end_path", page_key, ",".join(sorted(path_edges))), fact_type="dead_end_path",
            kind=kind, value=terminal_class if terminal_class != "UNRESOLVED" else None,
            source="vector_graph", evidence_status=evidence,
            supporting_edges=path_edges, supporting_nodes=path_nodes, page=page.page_number,
            detail={"terminal_node": leaf, "bridge_adjacent": has_bridge},
        ))

    # --- Bridge continuity (diagnostic only, always UNRESOLVED) ---
    for be in bridge_edges:
        root_s, root_t = uf.find(be.source), uf.find(be.target)
        if root_s == root_t:
            continue  # bridge lies within an already directly-connected component -- no cross-component claim
        comp_a = component_id_by_node.get(be.source)
        comp_b = component_id_by_node.get(be.target)
        if not comp_a or not comp_b:
            continue
        facts.append(PlanFact(
            fact_id=_fact_id("bridge_continuity", page_key, be.id), fact_type="bridge_continuity",
            kind="UNRESOLVED", value=None, source="vector_graph_bridging", evidence_status="reconstructed",
            supporting_edges=[be.id], supporting_nodes=[be.source, be.target], page=page.page_number,
            detail={"component_a": comp_a, "component_b": comp_b, "bridge_reason": be.bridge_reason},
        ))

    return facts, {"components": component_summaries}


def build_document_facts(doc: schema.DocumentAnalysis) -> dict:
    """Aggregates build_page_facts() across every page of a DocumentAnalysis
    into the response's plan_facts block. Never combines topology across
    pages -- each page is processed and reported independently."""
    all_facts: list[dict] = []
    all_components: list[dict] = []
    pages_meta: list[dict] = []
    for page in doc.pages:
        page_facts, summary = build_page_facts(page)
        all_facts.extend(f.to_dict() for f in page_facts)
        all_components.extend(summary["components"])
        pages_meta.append({
            "page_number": page.page_number, "width": page.width, "height": page.height,
            "rotation": page.rotation, "text_source": page.text_source,
            "graph_nodes": len(page.graph.nodes), "graph_edges": len(page.graph.edges),
            "bridge_edges": sum(1 for e in page.graph.edges if e.is_bridge),
        })

    kind_counts = {"FACT": 0, "DERIVED_FACT": 0, "UNRESOLVED": 0}
    for f in all_facts:
        kind_counts[f["kind"]] = kind_counts.get(f["kind"], 0) + 1

    return {
        "pages": pages_meta,
        "components": all_components,
        "facts": all_facts,
        "stats": {
            "fact_count": len(all_facts),
            "by_kind": kind_counts,
            "components_with_cycle": sum(1 for c in all_components if c["topology_status"] == "cycle_present"),
            "components_acyclic": sum(1 for c in all_components if c["topology_status"] == "acyclic_tree"),
        },
    }
