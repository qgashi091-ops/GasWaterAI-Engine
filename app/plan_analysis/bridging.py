"""Bridges genuine junction gaps left after endpoint-snapping, using local
geometry and symbol evidence -- never by loosening the global snap
tolerance. A blanket increase would just as happily fuse two separate,
merely-nearby branches of a distributor/valve battery as it would repair a
real junction, since both look identical to a tolerance check alone.

Three independent, conservative mechanisms, all gated on matching color
(or both uncolored) as a first safety check:

- Collinear gap bridge: two dangling ends in different components, each
  the other's nearest qualifying candidate, whose connecting line closely
  continues *both* stubs' own directions. This is the signature of a
  dashed/interrupted linetype or a straight run split across two CAD
  entities with a small real gap -- not a branch meeting at an angle.
- Symbol-mediated bridge: dangling ends already identified as ports of the
  same detected symbol (symbols.detect_symbols) genuinely belong to that
  fitting. But a symbol's bounding box can just as easily enclose several
  parallel, functionally-separate branch stubs (e.g. a distributor
  manifold) as it can a real multi-way valve/tee -- both look like
  "several ports near one symbol." We only bridge when (a) there are just
  2-3 distinct components at the symbol, not a whole cluster, and (b)
  their stub directions are not all mutually parallel: a real junction's
  arms point in genuinely different directions, while several near-
  parallel stubs sharing a symbol's bounding box is the distributor
  pattern this is explicitly meant not to fuse.
- Corner convergence bridge: two dangling ends whose stubs meet at a real
  angle (not near-collinear, so mechanism A doesn't apply) and have no
  symbol nearby (so mechanism B doesn't apply) -- an elbow/bend with no
  separate fitting icon drawn on it. Bridged only when extending *both*
  stubs' own lines forward makes them cross at a point close to both ends
  -- the geometric signature of an actual corner. This is deliberately
  not just "close and same color": two parallel distributor branches are
  never bridged by this test because parallel lines don't converge at all
  (the cross product used to find the intersection is ~0, so there's no
  intersection to test), which is exactly the failure mode this whole
  module exists to avoid. Investigating issue #3 (the graph audit's
  "dash-pattern fragmentation" hypothesis) found this convergence test
  passes for only ~5% of the angled dangling-end pairs that are merely
  close and same-colored -- most such pairs are NOT safely explainable as
  a real corner from local geometry alone, and are correctly left
  dangling rather than force-connected.

Anything that doesn't clear these gates is left as-is -- ambiguous cases
are never guessed, they just stay dangling.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .geometry import Point, dist
from .graph import BuiltEdge, BuiltGraph
from .symbols import BuiltSymbol

# Mechanism A: collinear gap bridge
MAX_COLLINEAR_GAP_DIAGONAL_FRACTION = 0.02
COLLINEAR_MIN_ALIGNMENT = 0.97  # dot product of unit vectors; ~14 degrees

# Mechanism B: symbol-mediated bridge
SYMBOL_BRIDGE_MIN_COMPONENTS = 2
SYMBOL_BRIDGE_MAX_COMPONENTS = 3
SYMBOL_BRIDGE_MIN_ANGLE_DEVIATION_DEGREES = 25.0

# Mechanism C: corner convergence bridge
MAX_CORNER_GAP_DIAGONAL_FRACTION = 0.02
CORNER_MAX_OVERSHOOT_FRACTION = -0.05  # allow the intersection to fall slightly "behind" either stub
CORNER_MAX_INTERSECTION_SLACK = 2.0  # intersection must land within this multiple of the gap distance from both ends


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
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class BridgeEvent:
    node_a: str
    node_b: str
    reason: str  # "collinear_gap" | "symbol_junction"
    symbol_id: str | None = None


def _stub_direction(node_id: str, graph: BuiltGraph, edges_by_node: dict) -> Point | None:
    """Unit vector pointing from the node's own edge outward through it
    (i.e. the direction the pipe run continues if extended past this end)."""
    edges = edges_by_node.get(node_id)
    if not edges:
        return None
    e = edges[0]
    other = e.target if e.source == node_id else e.source
    points = {n.id: n.point for n in graph.nodes}
    p, op = points[node_id], points[other]
    v = (p[0] - op[0], p[1] - op[1])
    length = math.hypot(*v)
    if length == 0:
        return None
    return (v[0] / length, v[1] / length)


def _colors_match(a, b) -> bool:
    return a == b


def _line_intersect(
    a: Point, dir_a: Point, b: Point, dir_b: Point
) -> tuple[Point, float, float] | None:
    """Intersection of two lines given as point + direction. Returns
    (point, t, s) where t/s are how far along each direction (from a/b
    respectively) the intersection sits. None if the lines are parallel
    (or near enough that the intersection would be numerically unstable) --
    which is exactly the case for two side-by-side distributor branches,
    so they never produce a corner candidate."""
    cross = dir_a[0] * dir_b[1] - dir_a[1] * dir_b[0]
    if abs(cross) < 1e-6:
        return None
    bx, by = b[0] - a[0], b[1] - a[1]
    t = (bx * dir_b[1] - by * dir_b[0]) / cross
    s = (bx * dir_a[1] - by * dir_a[0]) / cross
    point = (a[0] + t * dir_a[0], a[1] + t * dir_a[1])
    return point, t, s


def _find_corner_bridges(
    dangling_ids: list[str], graph: BuiltGraph, page_width: float, page_height: float
) -> list[BridgeEvent]:
    if len(dangling_ids) < 2:
        return []
    diagonal = math.hypot(page_width, page_height)
    max_gap = diagonal * MAX_CORNER_GAP_DIAGONAL_FRACTION

    points = {n.id: n.point for n in graph.nodes}
    edges_by_node: dict[str, list[BuiltEdge]] = defaultdict(list)
    for e in graph.edges:
        edges_by_node[e.source].append(e)
        edges_by_node[e.target].append(e)

    uf = _UnionFind([n.id for n in graph.nodes])
    for e in graph.edges:
        uf.union(e.source, e.target)

    directions = {nid: _stub_direction(nid, graph, edges_by_node) for nid in dangling_ids}
    colors = {nid: (edges_by_node[nid][0].color_rgb if edges_by_node.get(nid) else None) for nid in dangling_ids}

    best_partner: dict[str, str] = {}
    for a in dangling_ids:
        da = directions.get(a)
        if da is None:
            continue
        root_a = uf.find(a)
        best = None
        for b in dangling_ids:
            if b == a or uf.find(b) == root_a:
                continue
            if not _colors_match(colors[a], colors[b]):
                continue
            db = directions.get(b)
            if db is None:
                continue
            gap = dist(points[a], points[b])
            if gap > max_gap:
                continue
            hit = _line_intersect(points[a], da, points[b], db)
            if hit is None:
                continue  # near-parallel -- e.g. two distributor branches; no corner to find
            corner_point, t, s = hit
            if t < CORNER_MAX_OVERSHOOT_FRACTION or s < CORNER_MAX_OVERSHOOT_FRACTION:
                continue  # the implied corner sits behind one of the stubs, not ahead of it
            slack = max(gap, 1e-6) * CORNER_MAX_INTERSECTION_SLACK
            if dist(corner_point, points[a]) > slack or dist(corner_point, points[b]) > slack:
                continue  # the lines cross somewhere unrelated to this specific gap
            if best is None or gap < best[0]:
                best = (gap, b)
        if best is not None:
            best_partner[a] = best[1]

    events: list[BridgeEvent] = []
    seen = set()
    for a, b in best_partner.items():
        if best_partner.get(b) != a:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        events.append(BridgeEvent(node_a=a, node_b=b, reason="corner_convergence"))
    return events


def _find_collinear_bridges(
    dangling_ids: list[str], graph: BuiltGraph, page_width: float, page_height: float
) -> list[BridgeEvent]:
    if len(dangling_ids) < 2:
        return []
    diagonal = math.hypot(page_width, page_height)
    max_gap = diagonal * MAX_COLLINEAR_GAP_DIAGONAL_FRACTION

    points = {n.id: n.point for n in graph.nodes}
    edges_by_node: dict[str, list[BuiltEdge]] = defaultdict(list)
    for e in graph.edges:
        edges_by_node[e.source].append(e)
        edges_by_node[e.target].append(e)

    uf = _UnionFind([n.id for n in graph.nodes])
    for e in graph.edges:
        uf.union(e.source, e.target)

    directions = {nid: _stub_direction(nid, graph, edges_by_node) for nid in dangling_ids}
    colors = {nid: (edges_by_node[nid][0].color_rgb if edges_by_node.get(nid) else None) for nid in dangling_ids}

    # For each dangling node, find its best (nearest, qualifying) partner.
    best_partner: dict[str, str] = {}
    for a in dangling_ids:
        da = directions.get(a)
        if da is None:
            continue
        root_a = uf.find(a)
        best = None
        for b in dangling_ids:
            if b == a or uf.find(b) == root_a:
                continue
            if not _colors_match(colors[a], colors[b]):
                continue
            db = directions.get(b)
            if db is None:
                continue
            gap = dist(points[a], points[b])
            if gap > max_gap:
                continue
            gap_vec = (points[b][0] - points[a][0], points[b][1] - points[a][1])
            gap_len = math.hypot(*gap_vec)
            if gap_len == 0:
                continue
            gap_dir = (gap_vec[0] / gap_len, gap_vec[1] / gap_len)
            # both stubs must point "into" the gap towards each other
            align_a = abs(da[0] * gap_dir[0] + da[1] * gap_dir[1])
            align_b = abs(db[0] * gap_dir[0] + db[1] * gap_dir[1])
            if align_a < COLLINEAR_MIN_ALIGNMENT or align_b < COLLINEAR_MIN_ALIGNMENT:
                continue
            if best is None or gap < best[0]:
                best = (gap, b)
        if best is not None:
            best_partner[a] = best[1]

    events: list[BridgeEvent] = []
    seen = set()
    for a, b in best_partner.items():
        # only bridge when it's a mutual best match -- otherwise ambiguous
        if best_partner.get(b) != a:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        events.append(BridgeEvent(node_a=a, node_b=b, reason="collinear_gap"))
    return events


def _find_symbol_bridges(
    symbols: list[BuiltSymbol], graph: BuiltGraph
) -> list[BridgeEvent]:
    edges_by_node: dict[str, list[BuiltEdge]] = defaultdict(list)
    for e in graph.edges:
        edges_by_node[e.source].append(e)
        edges_by_node[e.target].append(e)

    uf = _UnionFind([n.id for n in graph.nodes])
    for e in graph.edges:
        uf.union(e.source, e.target)

    events: list[BridgeEvent] = []
    for sym in symbols:
        if len(sym.port_node_ids) < SYMBOL_BRIDGE_MIN_COMPONENTS:
            continue
        by_component: dict[str, list[str]] = defaultdict(list)
        for nid in sym.port_node_ids:
            by_component[uf.find(nid)].append(nid)
        if not (SYMBOL_BRIDGE_MIN_COMPONENTS <= len(by_component) <= SYMBOL_BRIDGE_MAX_COMPONENTS):
            continue

        reps = [nids[0] for nids in by_component.values()]
        colors = [edges_by_node[nid][0].color_rgb if edges_by_node.get(nid) else None for nid in reps]
        if len(set(colors)) > 1:
            continue  # differing pipe colors at this symbol -- ambiguous, leave unresolved

        directions = [_stub_direction(nid, graph, edges_by_node) for nid in reps]
        if any(d is None for d in directions):
            continue

        max_deviation = 0.0
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                dot = max(-1.0, min(1.0, directions[i][0] * directions[j][0] + directions[i][1] * directions[j][1]))
                angle = math.degrees(math.acos(dot))
                deviation_from_parallel = min(angle, abs(180.0 - angle))
                max_deviation = max(max_deviation, deviation_from_parallel)

        if max_deviation < SYMBOL_BRIDGE_MIN_ANGLE_DEVIATION_DEGREES:
            continue  # stubs all point the same way -- looks like a distributor, not a junction

        # bridge the representatives pairwise as a star from the first
        anchor = reps[0]
        for other in reps[1:]:
            events.append(BridgeEvent(node_a=anchor, node_b=other, reason="symbol_junction", symbol_id=sym.id))

    return events


def bridge_graph(
    graph: BuiltGraph,
    symbols: list[BuiltSymbol],
    dangling_ids: list[str],
    page_width: float,
    page_height: float,
) -> tuple[BuiltGraph, list[BridgeEvent]]:
    """Returns a new graph with bridge edges added for confidently-resolved
    junction gaps, plus the list of bridges applied. Everything not covered
    by either mechanism is left exactly as it was -- still dangling."""
    collinear_events = _find_collinear_bridges(dangling_ids, graph, page_width, page_height)
    symbol_events = _find_symbol_bridges(symbols, graph)
    corner_events = _find_corner_bridges(dangling_ids, graph, page_width, page_height)

    events: list[BridgeEvent] = []
    seen_pairs: set[tuple[str, str]] = set()
    for ev in collinear_events + symbol_events + corner_events:
        key = tuple(sorted((ev.node_a, ev.node_b)))
        if key in seen_pairs:
            continue  # a pair independently qualifying under two mechanisms is bridged once
        seen_pairs.add(key)
        events.append(ev)
    if not events:
        return graph, []

    points = {n.id: n.point for n in graph.nodes}
    new_edges = list(graph.edges)
    degree_delta: dict[str, int] = defaultdict(int)
    for i, ev in enumerate(events):
        length = dist(points[ev.node_a], points[ev.node_b])
        new_edges.append(
            BuiltEdge(
                id=f"bridge{i}",
                source=ev.node_a,
                target=ev.node_b,
                length=length,
                color_rgb=None,
                stroke_width=None,
                polyline=[points[ev.node_a], points[ev.node_b]],
                is_bridge=True,
                bridge_reason=ev.reason,
            )
        )
        degree_delta[ev.node_a] += 1
        degree_delta[ev.node_b] += 1

    new_nodes = []
    for n in graph.nodes:
        if degree_delta.get(n.id):
            new_nodes.append(type(n)(id=n.id, point=n.point, degree=n.degree + degree_delta[n.id]))
        else:
            new_nodes.append(n)

    return BuiltGraph(nodes=new_nodes, edges=new_edges), events
