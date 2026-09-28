"""Pydantic models for the structured JSON handed off to Base44.

This is the contract between "own Analyse-Service" (this service) and the
downstream Base44 review step: we only extract and structure geometry,
text and topology from the plan PDF. Semantic classification against the
symbol library, guideline checks and the AI review itself stay in Base44.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


class TextSpan(BaseModel):
    id: str
    text: str
    bbox: BBox
    source: Literal["native", "ocr"]
    confidence: Optional[float] = None
    font_size: Optional[float] = None
    label_hint: Optional[str] = Field(
        default=None,
        description=(
            "Non-authoritative pattern match against SVGW W3 Anhang 4 label "
            "conventions (pipe codes like PWC/PWH-C, DN sizes, backflow-device "
            "codes). Base44's symbol library remains the source of truth."
        ),
    )


class GraphNode(BaseModel):
    id: str
    point: Point
    degree: int
    is_endpoint: bool = Field(description="True if degree == 1 (a dangling pipe end)")


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    length: float
    color_rgb: Optional[tuple[float, float, float]] = None
    stroke_width: Optional[float] = None
    polyline: list[Point]
    is_bridge: bool = Field(
        default=False,
        description=(
            "True if this edge was inferred (not drawn) to bridge a junction gap left by "
            "endpoint-snapping -- see bridge_reason."
        ),
    )
    bridge_reason: Optional[Literal["collinear_gap", "symbol_junction", "corner_convergence"]] = None


class PipeGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class TextAssociation(BaseModel):
    text_id: str
    target_type: Literal["symbol", "edge", "unassigned"]
    target_id: Optional[str] = None
    distance: Optional[float] = None


class SymbolCandidate(BaseModel):
    id: str
    bbox: BBox
    centroid: Point
    primitive_count: int
    has_curves: bool
    area: float
    label: Optional[str] = None
    label_text_id: Optional[str] = None
    label_distance: Optional[float] = None
    port_node_ids: list[str] = Field(default_factory=list)


class PageAnalysis(BaseModel):
    page_number: int
    width: float
    height: float
    rotation: int = 0
    text_source: Literal["native", "ocr", "mixed", "none"]
    text_spans: list[TextSpan] = Field(default_factory=list)
    graph: PipeGraph = Field(default_factory=PipeGraph)
    symbols: list[SymbolCandidate] = Field(default_factory=list)
    associations: list[TextAssociation] = Field(default_factory=list)
    dangling_pipe_ends: list[str] = Field(
        default_factory=list,
        description="Node ids with degree 1 that were not matched to any symbol port",
    )
    warnings: list[str] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)


class DocumentAnalysis(BaseModel):
    filename: str
    page_count: int
    pages: list[PageAnalysis]
    warnings: list[str] = Field(default_factory=list)
