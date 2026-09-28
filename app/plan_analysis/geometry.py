"""Geometry helpers: distances, bounding boxes, and grid-based spatial clustering.

Plan PDFs in this pipeline vary wildly in page size (a CAD export at native
paper units vs. a rasterized A4 export), so clustering tolerances are always
expressed as a fraction of the page diagonal rather than an absolute point
value.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x0, y0, x1, y1


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bbox_of_points(points: Iterable[Point]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_union(boxes: Sequence[BBox]) -> BBox:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def bbox_center(b: BBox) -> Point:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def bbox_max_dim(b: BBox) -> float:
    return max(b[2] - b[0], b[3] - b[1])


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_expand(b: BBox, margin: float) -> BBox:
    return (b[0] - margin, b[1] - margin, b[2] + margin, b[3] + margin)


def bbox_distance(a: BBox, b: BBox) -> float:
    """0 if overlapping/touching, else the gap between the two boxes."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def point_in_or_near_bbox(p: Point, b: BBox, margin: float) -> bool:
    return (b[0] - margin) <= p[0] <= (b[2] + margin) and (b[1] - margin) <= p[1] <= (b[3] + margin)


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj = (ax + t * dx, ay + t * dy)
    return dist(p, proj)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class SpatialBBoxClusterer:
    """Groups items with bounding boxes into connected components using a
    uniform grid so we never do an O(n^2) comparison over tens of thousands
    of vector primitives.

    Two items are linked when their (margin-expanded) bounding boxes touch
    or overlap.
    """

    def __init__(self, boxes: Sequence[BBox], gap: float):
        self.boxes = boxes
        self.gap = max(gap, 1e-6)
        self.uf = UnionFind(len(boxes))
        self._cluster()

    def _cell_range(self, b: BBox):
        eb = bbox_expand(b, self.gap / 2.0)
        cx0 = math.floor(eb[0] / self.gap)
        cy0 = math.floor(eb[1] / self.gap)
        cx1 = math.floor(eb[2] / self.gap)
        cy1 = math.floor(eb[3] / self.gap)
        return cx0, cy0, cx1, cy1

    def _cluster(self) -> None:
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, box in enumerate(self.boxes):
            cx0, cy0, cx1, cy1 = self._cell_range(box)
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    grid.setdefault((cx, cy), []).append(idx)
        for cell_items in grid.values():
            if len(cell_items) < 2:
                continue
            first = cell_items[0]
            for other in cell_items[1:]:
                if bbox_distance(self.boxes[first], self.boxes[other]) <= self.gap:
                    self.uf.union(first, other)
                else:
                    # fall back to a direct check against all items already
                    # unioned in this cell so we don't miss links purely
                    # because the "first" pick happened to be far away.
                    for j in cell_items:
                        if j != other and bbox_distance(self.boxes[j], self.boxes[other]) <= self.gap:
                            self.uf.union(j, other)
                            break

    def clusters(self) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for idx in range(len(self.boxes)):
            root = self.uf.find(idx)
            groups.setdefault(root, []).append(idx)
        return list(groups.values())


class PointSnapper:
    """Snaps nearby points to shared cluster ids (for building graph nodes
    out of line endpoints that should coincide but suffer float noise)."""

    def __init__(self, points: Sequence[Point], tolerance: float):
        self.points = points
        self.tolerance = max(tolerance, 1e-6)
        boxes = [(p[0], p[1], p[0], p[1]) for p in points]
        self._clusterer = SpatialBBoxClusterer(boxes, gap=self.tolerance)

    def node_ids(self) -> list[int]:
        """Returns, for each input point, the id of its snapped cluster."""
        clusters = self._clusterer.clusters()
        point_to_cluster = [0] * len(self.points)
        for cluster_id, members in enumerate(clusters):
            for m in members:
                point_to_cluster[m] = cluster_id
        return point_to_cluster

    def cluster_centroids(self) -> list[Point]:
        clusters = self._clusterer.clusters()
        centroids = []
        for members in clusters:
            xs = [self.points[m][0] for m in members]
            ys = [self.points[m][1] for m in members]
            centroids.append((sum(xs) / len(xs), sum(ys) / len(ys)))
        return centroids
