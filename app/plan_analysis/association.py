"""Associates each extracted text label with the nearest symbol candidate
or, failing that, the nearest pipe edge (e.g. a "DN20" diameter callout
sitting on a pipe run with no symbol nearby). Labels further than the
search radius from anything are left unassigned rather than force-matched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Point, bbox_center, point_segment_distance
from .graph import BuiltGraph
from .symbols import BuiltSymbol

MAX_LABEL_DISTANCE_DIAGONAL_FRACTION = 0.02


@dataclass
class Association:
    text_id: str
    target_type: str  # "symbol" | "edge" | "unassigned"
    target_id: str | None
    distance: float | None


def _grid_key(point: Point, cell_size: float) -> tuple[int, int]:
    return (math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))


def _build_point_grid(points: list[Point], cell_size: float) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, p in enumerate(points):
        grid.setdefault(_grid_key(p, cell_size), []).append(idx)
    return grid


def _build_edge_grid(graph: BuiltGraph, node_points: dict[str, Point], cell_size: float) -> dict[tuple[int, int], list[int]]:
    """Indexes each edge into every grid cell its segment actually passes
    through, not just the cell containing its midpoint. A single-midpoint
    index misses a text label sitting near either END of a long edge (a
    common case on riser/Steigzonenschema drawings, where pipe runs of
    several hundred points are routine): the midpoint can be far outside the
    +/-1-cell neighborhood searched around the label, even though the true
    nearest point on the segment is well within the search radius. Sampling
    along the segment (capped at 50 samples, plenty relative to cell_size)
    fixes this without touching the search radius itself.
    """
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, e in enumerate(graph.edges):
        a = node_points[e.source]
        b = node_points[e.target]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, min(50, math.ceil(length / cell_size)))
        seen_cells: set[tuple[int, int]] = set()
        for step in range(steps + 1):
            t = step / steps
            sample = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            key = _grid_key(sample, cell_size)
            if key in seen_cells:
                continue
            seen_cells.add(key)
            grid.setdefault(key, []).append(idx)
    return grid


def associate_text_spans(
    text_items: list[tuple[str, Point]],
    symbols: list[BuiltSymbol],
    graph: BuiltGraph,
    page_width: float,
    page_height: float,
) -> list[Association]:
    diagonal = (page_width**2 + page_height**2) ** 0.5
    max_distance = diagonal * MAX_LABEL_DISTANCE_DIAGONAL_FRACTION
    cell_size = max(max_distance, 1e-6)

    symbol_points = [s.centroid for s in symbols]
    symbol_grid = _build_point_grid(symbol_points, cell_size)

    node_points = {n.id: n.point for n in graph.nodes}
    edge_grid = _build_edge_grid(graph, node_points, cell_size)

    associations: list[Association] = []
    for text_id, point in text_items:
        best_type, best_id, best_dist = "unassigned", None, None

        cx, cy = _grid_key(point, cell_size)
        candidate_symbols = [
            i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in symbol_grid.get((cx + dx, cy + dy), [])
        ]
        for i in candidate_symbols:
            d = math.hypot(point[0] - symbol_points[i][0], point[1] - symbol_points[i][1])
            if d <= max_distance and (best_dist is None or d < best_dist):
                best_type, best_id, best_dist = "symbol", symbols[i].id, d

        if best_type == "unassigned":
            candidate_edges = [
                i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in edge_grid.get((cx + dx, cy + dy), [])
            ]
            for i in candidate_edges:
                edge = graph.edges[i]
                d = point_segment_distance(point, node_points[edge.source], node_points[edge.target])
                if d <= max_distance and (best_dist is None or d < best_dist):
                    best_type, best_id, best_dist = "edge", edge.id, d

        associations.append(Association(text_id=text_id, target_type=best_type, target_id=best_id, distance=best_dist))

    return associations
