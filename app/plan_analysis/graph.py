"""Builds the pipe network graph from classified straight-line segments.

Endpoints within a small tolerance (relative to page size) are snapped to
a shared node, which is how a decomposed CAD polyline becomes a connected
topological graph again. Segments that merely cross in plan view without
sharing an endpoint are deliberately left unconnected -- that's a real
pipe-crossing-over-pipe situation, not a junction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import Point, PointSnapper, dist
from .vectors import LineSegment

# Endpoints closer than this fraction of the page diagonal are considered
# the same junction.
SNAP_DIAGONAL_FRACTION = 0.003


@dataclass
class BuiltNode:
    id: str
    point: Point
    degree: int = 0


@dataclass
class BuiltEdge:
    id: str
    source: str
    target: str
    length: float
    color_rgb: tuple[float, float, float] | None
    stroke_width: float | None
    polyline: list[Point]
    is_bridge: bool = False
    bridge_reason: str | None = None


@dataclass
class BuiltGraph:
    nodes: list[BuiltNode]
    edges: list[BuiltEdge]


def build_pipe_graph(segments: list[LineSegment], page_width: float, page_height: float) -> BuiltGraph:
    if not segments:
        return BuiltGraph(nodes=[], edges=[])

    diagonal = (page_width**2 + page_height**2) ** 0.5
    tolerance = max(diagonal * SNAP_DIAGONAL_FRACTION, 1e-6)

    endpoints: list[Point] = []
    for seg in segments:
        endpoints.append(seg.p0)
        endpoints.append(seg.p1)

    snapper = PointSnapper(endpoints, tolerance=tolerance)
    cluster_ids = snapper.node_ids()
    centroids = snapper.cluster_centroids()

    node_ids = [f"n{cid}" for cid in range(len(centroids))]
    nodes = [BuiltNode(id=node_ids[cid], point=centroids[cid]) for cid in range(len(centroids))]

    edges: list[BuiltEdge] = []
    seen_pairs: set[tuple[str, str, tuple]] = set()
    for i, seg in enumerate(segments):
        c0 = cluster_ids[2 * i]
        c1 = cluster_ids[2 * i + 1]
        if c0 == c1:
            continue  # degenerate segment collapsed by snapping
        n0, n1 = node_ids[c0], node_ids[c1]
        key = (min(n0, n1), max(n0, n1), (seg.color_rgb, seg.stroke_width))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        length = dist(nodes[c0].point, nodes[c1].point)
        edges.append(
            BuiltEdge(
                id=f"e{len(edges)}",
                source=n0,
                target=n1,
                length=length,
                color_rgb=seg.color_rgb,
                stroke_width=seg.stroke_width,
                polyline=[nodes[c0].point, nodes[c1].point],
            )
        )
        nodes[c0].degree += 1
        nodes[c1].degree += 1

    # Drop isolated nodes that ended up with no surviving edge.
    used_ids = {e.source for e in edges} | {e.target for e in edges}
    nodes = [n for n in nodes if n.id in used_ids]

    return BuiltGraph(nodes=nodes, edges=edges)
