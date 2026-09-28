"""Clusters leftover vector primitives (circles, arcs, hatching, filled
icons -- anything that isn't a pipe line or a text glyph) into symbol
candidates, and links dangling pipe ends ("ports") to the nearest symbol.

This module does not attempt to name the symbol (valve, backflow
preventer, water meter, ...): that classification belongs to Base44, which
has the actual SVGW symbol library and the ~150 reference cases to match
against. We only hand over geometry: where a component sits, how big it
is, and which pipe ends touch it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import (
    BBox,
    Point,
    SpatialBBoxClusterer,
    bbox_area,
    bbox_center,
    bbox_max_dim,
    bbox_union,
    point_in_or_near_bbox,
)
from .graph import BuiltGraph
from .vectors import VectorPrimitive

SYMBOL_CLUSTER_GAP_FRACTION = 0.004
MIN_SYMBOL_DIAGONAL_FRACTION = 0.0015
MAX_MERGED_SYMBOL_DIAGONAL_FRACTION = 0.16
PORT_SEARCH_MARGIN_FRACTION = 0.01


@dataclass
class BuiltSymbol:
    id: str
    bbox: BBox
    centroid: Point
    primitive_count: int
    has_curves: bool
    area: float
    port_node_ids: list[str] = field(default_factory=list)


def detect_symbols(
    primitives: list[VectorPrimitive],
    graph: BuiltGraph,
    page_width: float,
    page_height: float,
) -> tuple[list[BuiltSymbol], list[str]]:
    """Returns (symbols, dangling_pipe_end_node_ids)."""
    diagonal = (page_width**2 + page_height**2) ** 0.5
    if not primitives:
        dangling = [n.id for n in graph.nodes if n.degree == 1]
        return [], dangling

    boxes = [p.bbox for p in primitives]
    clusterer = SpatialBBoxClusterer(boxes, gap=diagonal * SYMBOL_CLUSTER_GAP_FRACTION)

    min_dim = diagonal * MIN_SYMBOL_DIAGONAL_FRACTION
    max_dim = diagonal * MAX_MERGED_SYMBOL_DIAGONAL_FRACTION

    symbols: list[BuiltSymbol] = []
    for cluster in clusterer.clusters():
        cluster_boxes = [boxes[i] for i in cluster]
        bbox = bbox_union(cluster_boxes)
        dim = bbox_max_dim(bbox)
        if dim < min_dim or dim > max_dim:
            continue
        has_curves = any(primitives[i].has_curves for i in cluster)
        item_count = sum(primitives[i].item_count for i in cluster)
        if len(cluster) < 2 and not has_curves:
            continue  # a single straight-edge fragment is more likely noise
        symbols.append(
            BuiltSymbol(
                id=f"sym{len(symbols)}",
                bbox=bbox,
                centroid=bbox_center(bbox),
                primitive_count=item_count,
                has_curves=has_curves,
                area=bbox_area(bbox),
            )
        )

    margin = diagonal * PORT_SEARCH_MARGIN_FRACTION
    matched_nodes: set[str] = set()
    for sym in symbols:
        for node in graph.nodes:
            if node.degree != 1 or node.id in matched_nodes:
                continue
            if point_in_or_near_bbox(node.point, sym.bbox, margin):
                sym.port_node_ids.append(node.id)
                matched_nodes.add(node.id)

    dangling = [n.id for n in graph.nodes if n.degree == 1 and n.id not in matched_nodes]
    return symbols, dangling
