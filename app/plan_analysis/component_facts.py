"""Component recognition -- v0.2, new. Answers: what component is located
at or near this proven graph position? Built strictly on top of v0.1's
existing, UNCHANGED outputs (schema.DocumentAnalysis, plan_facts.py's
plan_facts block) plus the symbol reference library (symbol_library.py).

ARCHITECTURAL RULE (non-negotiable, same discipline as plan_facts.py):
every recognized component is exactly one of COMPONENT_FACT /
COMPONENT_CANDIDATE / UNRESOLVED -- a weak visual resemblance is never
silently promoted to a fact. Precision before recall: an ambiguous or
low-confidence match is reported UNRESOLVED or COMPONENT_CANDIDATE, never
forced into a specific, possibly wrong, type.

GRAPH ASSOCIATION: reuses v0.1's own `symbols.py::detect_symbols()` port
matching (a symbol's `port_node_ids` are already-proven, degree-1 graph
nodes within a small search margin of the symbol) -- this module invents no
new connectivity. A symbol with zero ports is checked for proximity to an
edge's *drawn* polyline (a mid-run component, e.g. a valve inline on a
pipe); if neither holds, it is honestly reported "near_not_connected",
never guessed into a connection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

import numpy as np
import pymupdf

from . import schema
from .geometry import point_segment_distance
from .symbol_library import SymbolLibrary, canonicalize, get_library

RENDER_DPI = 300
CROP_MARGIN_FRACTION = 0.20  # extra render margin around a symbol's own bbox
EDGE_PROXIMITY_DIAGONAL_FRACTION = 0.02  # same order of magnitude as association.py's own search radius

# Precision-first thresholds (see docs/symbol-audit.md and
# docs/w003-component-poc-report.md for how these were chosen/validated --
# not tuned to maximize apparent recall).
FACT_SCORE_MIN = 0.90
FACT_MARGIN_MIN = 0.12
CANDIDATE_SCORE_MIN = 0.75


def _stable_id(*parts: str) -> str:
    return "CF" + sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


@dataclass
class ComponentFact:
    component_fact_id: str
    kind: str  # "COMPONENT_FACT" | "COMPONENT_CANDIDATE" | "UNRESOLVED"
    page: int
    bbox: tuple
    centroid: tuple
    symbol_type: Optional[str]
    symbol_name: Optional[str]
    recognition_method: str
    confidence: float
    graph_association: dict
    supporting_evidence: dict
    ambiguity_candidates: list = field(default_factory=list)
    corroboration_status: str = "none"

    def to_dict(self) -> dict:
        return {
            "component_fact_id": self.component_fact_id,
            "kind": self.kind,
            "page": self.page,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "symbol_type": self.symbol_type,
            "symbol_name": self.symbol_name,
            "recognition_method": self.recognition_method,
            "confidence": self.confidence,
            "graph_association": self.graph_association,
            "supporting_evidence": self.supporting_evidence,
            "ambiguity_candidates": self.ambiguity_candidates,
            "corroboration_status": self.corroboration_status,
        }


def _render_symbol_crop(page: "pymupdf.Page", bbox: tuple) -> Optional[np.ndarray]:
    # IMPORTANT (found while building this module, not present in any v0.1
    # code path so it was never previously observable): for a rotated page
    # (page.rotation != 0), PyMuPDF's page.get_drawings()/get_text() --
    # which graph.py/symbols.py/text.py build every bbox from -- report
    # coordinates in the page's UNROTATED coordinate frame, while
    # page.get_pixmap()'s `clip` argument (like page.rect itself) is in the
    # ROTATED/display frame. schema.PageAnalysis.width/height are also
    # display-frame (from page.rect), so a v0.1 bbox and v0.1's own
    # width/height already silently disagree on which frame they're in --
    # this was invisible in v0.1 because plan_facts.py's topology algorithm
    # never renders a pixel or compares a bbox against page.rect. It matters
    # here because this is the first module that renders raster pixels from
    # a bbox. Fixed locally, without touching any v0.1 file, by mapping the
    # bbox through page.rotation_matrix (identity when rotation == 0) before
    # building the clip rect.
    x0, y0, x1, y1 = pymupdf.Rect(bbox) * page.rotation_matrix
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    margin = CROP_MARGIN_FRACTION * max(w, h)
    clip = pymupdf.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin) & page.rect
    if clip.is_empty:
        return None
    zoom = RENDER_DPI / 72.0
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
    arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n >= 3:
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    else:
        gray = arr[:, :, 0]
    return gray


def _graph_association(sym: schema.SymbolCandidate, edges: list, page_diagonal: float) -> dict:
    ports = sym.port_node_ids
    if len(ports) >= 2:
        return {"relation": "at_branch", "node_ids": list(ports), "edge_ids": []}
    if len(ports) == 1:
        return {"relation": "at_endpoint", "node_ids": list(ports), "edge_ids": []}

    # Zero proven ports: check proximity to an edge's DRAWN polyline (never
    # invent a connection -- only report one when the geometry itself is
    # within the same order-of-magnitude search radius association.py uses
    # for text-to-edge matching).
    margin = page_diagonal * EDGE_PROXIMITY_DIAGONAL_FRACTION
    best_edge_id, best_dist = None, None
    cx, cy = sym.centroid
    for e in edges:
        if len(e.polyline) < 2:
            continue
        d = point_segment_distance((cx, cy), e.polyline[0], e.polyline[-1])
        if best_dist is None or d < best_dist:
            best_dist, best_edge_id = d, e.id
    if best_edge_id is not None and best_dist <= margin:
        return {"relation": "on_edge", "node_ids": [], "edge_ids": [best_edge_id]}
    return {"relation": "near_not_connected", "node_ids": [], "edge_ids": []}


def _decide(sym: schema.SymbolCandidate, page_number: int, scores: list, library: SymbolLibrary,
            graph_assoc: dict) -> ComponentFact:
    base = dict(
        component_fact_id=_stable_id(str(page_number), sym.id, f"{sym.bbox}"),
        page=page_number, bbox=sym.bbox, centroid=sym.centroid,
        recognition_method="classical_cv_template_match_rotation_invariant",
        graph_association=graph_assoc,
    )
    if not scores:
        return ComponentFact(**base, kind="UNRESOLVED", symbol_type=None, symbol_name=None, confidence=0.0,
                              supporting_evidence={"reason": "no ink in rendered crop, or crop could not be rendered"})

    top, second = scores[0], (scores[1] if len(scores) > 1 else None)
    margin = top.score - (second.score if second else 0.0)
    family = library.ambiguous_family_by_symbol.get(top.symbol_id)
    # A family conflict means THIS SPECIFIC match, not just the templates in
    # the abstract, can't be told apart from a family-mate -- the audit's
    # family flag is only a prior; this checks it against the real result.
    family_conflict = family is not None and any(
        s.symbol_id in family and s.symbol_id != top.symbol_id for s in scores[:3]
    )
    text_required = top.symbol_id in library.text_legend_required
    ambiguity_candidates = [
        {"symbol_id": s.symbol_id, "name": library.templates[s.symbol_id].name, "score": round(s.score, 4)}
        for s in scores[:3]
    ]
    evidence = {
        "best_score": round(top.score, 4),
        "second_best_score": round(second.score, 4) if second else None,
        "margin": round(margin, 4),
        "rotation_degrees": top.rotation,
    }

    if top.score >= FACT_SCORE_MIN and margin >= FACT_MARGIN_MIN and not family_conflict and not text_required:
        kind, corroboration = "COMPONENT_FACT", "shape_evidence_sufficient"
    elif top.score >= CANDIDATE_SCORE_MIN:
        kind = "COMPONENT_CANDIDATE"
        corroboration = "text_or_legend_required" if text_required else ("ambiguous_family" if family_conflict else "below_fact_threshold")
    else:
        kind, corroboration = "UNRESOLVED", "insufficient_shape_evidence"

    return ComponentFact(
        **base, kind=kind,
        symbol_type=top.symbol_id if kind != "UNRESOLVED" else None,
        symbol_name=library.templates[top.symbol_id].name if kind != "UNRESOLVED" else None,
        confidence=top.score, ambiguity_candidates=ambiguity_candidates,
        corroboration_status=corroboration, supporting_evidence=evidence,
    )


def build_component_facts(pdf_bytes: bytes, doc: schema.DocumentAnalysis) -> dict:
    """Pure function of (pdf_bytes, doc) -- deterministic given identical
    input. Re-opens the PDF only to render raster crops at each symbol's own
    already-computed bbox (v0.1's symbol DETECTION is untouched; only this
    module's raster rendering is new)."""
    library = get_library()
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        facts: list[ComponentFact] = []
        seen_bbox_keys: set[tuple] = set()
        for page_model in doc.pages:
            page = pdf[page_model.page_number - 1]
            page_diagonal = (page_model.width ** 2 + page_model.height ** 2) ** 0.5
            for sym in page_model.symbols:
                # Defensive de-duplication: the SAME physical symbol must
                # never produce two component facts, even if it were ever
                # handed to this function twice (e.g. an upstream retry).
                bbox_key = (page_model.page_number,) + tuple(round(v, 1) for v in sym.bbox)
                if bbox_key in seen_bbox_keys:
                    continue
                seen_bbox_keys.add(bbox_key)

                crop_gray = _render_symbol_crop(page, sym.bbox)
                candidate_canon = canonicalize(crop_gray) if crop_gray is not None else None
                scores = library.match(candidate_canon)
                graph_assoc = _graph_association(sym, page_model.graph.edges, page_diagonal)
                facts.append(_decide(sym, page_model.page_number, scores, library, graph_assoc))
    finally:
        pdf.close()

    kind_counts = {"COMPONENT_FACT": 0, "COMPONENT_CANDIDATE": 0, "UNRESOLVED": 0}
    for f in facts:
        kind_counts[f.kind] += 1

    return {
        "facts": [f.to_dict() for f in facts],
        "stats": {
            "components_evaluated": len(facts),
            "by_kind": kind_counts,
            "library_eligible_symbols": len(library.eligible_ids),
            "library_total_symbols": len(library.meta_by_id),
        },
    }
