"""PlanAgentContext -- the single place that wraps this engine's EXISTING
deterministic outputs (schema.DocumentAnalysis, plan_facts.build_document_facts,
component_evidence, canonical_inventory, rule check results) for the
multi-agent pipeline to read. This module computes NOTHING new about the
plan itself -- it only indexes what already exists so each agent can pull a
small, relevant slice instead of the whole plan (per this epic's explicit
"Fachagenten erhalten nur kleinen Kontext" requirement).

Subject-id convention (stable, content-derived, reused everywhere in this
package so a claim_type + subject_id pair always names the same real plan
element): "p<page>:node:<node_id>", "p<page>:edge:<edge_id>",
"p<page>:symbol:<symbol_id>", "p<page>" for whole-page claims (e.g.
PlanstrukturAgent's plan-area classification), or an existing
component_evidence_id / inventory_id verbatim when wrapping those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pymupdf

from app.plan_analysis import schema
from .schema import DeterministicFactRef

CROP_MARGIN_FRACTION = 0.6
RENDER_DPI = 220.0


def _page_subject(page_number: int) -> str:
    return f"p{page_number}"


def _node_subject(page_number: int, node_id: str) -> str:
    return f"p{page_number}:node:{node_id}"


def _edge_subject(page_number: int, edge_id: str) -> str:
    return f"p{page_number}:edge:{edge_id}"


def _symbol_subject(page_number: int, symbol_id: str) -> str:
    return f"p{page_number}:symbol:{symbol_id}"


# Maps a plan_facts.py fact_type to (claim_type, subject-id builder). Only
# fact types this package's agents actually reconcile against are listed --
# anything else in plan_facts stays available via `facts_by_type` but has no
# claim_type mapping here, which is correct: not every deterministic fact is
# something an agent in this architecture opines about.
_FACT_TYPE_MAP = {
    "cycle_membership": "schlaufung_topology",
    "dead_end_path": "schlaufung_topology",
    "terminal_endpoint": "schlaufung_topology",
    "dimension_evidence": "leitung_dimension_evidence",
    "pipe_segment": "leitung_segment",
}


def _subject_for_fact(fact: dict) -> Optional[str]:
    page = fact.get("page")
    ftype = fact["fact_type"]
    if page is None:
        return None
    if ftype in ("terminal_endpoint", "dead_end_path"):
        node = fact.get("detail", {}).get("terminal_node") or (fact.get("supporting_nodes") or [None])[0]
        return _node_subject(page, node) if node else None
    if ftype == "cycle_membership":
        comp_id = fact.get("detail", {}).get("component_id")
        return comp_id
    if ftype in ("dimension_evidence", "pipe_segment"):
        edges = fact.get("supporting_edges") or []
        return _edge_subject(page, edges[0]) if edges else None
    return None


@dataclass
class PlanAgentContext:
    doc: schema.DocumentAnalysis
    pdf_bytes: Optional[bytes] = None
    plan_facts: dict = field(default_factory=dict)  # plan_facts.build_document_facts(doc) output
    component_evidence: list[dict] = field(default_factory=list)
    inventory: list[dict] = field(default_factory=list)
    rule_checks: list[dict] = field(default_factory=list)

    _facts_by_subject: dict = field(default_factory=dict, repr=False, init=False)
    _facts_by_claim_type: dict = field(default_factory=dict, repr=False, init=False)
    _pages_by_number: dict = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        self._pages_by_number = {p.page_number: p for p in self.doc.pages}
        self._facts_by_subject = {}
        self._facts_by_claim_type = {}
        for fact in self.plan_facts.get("facts", []):
            claim_type = _FACT_TYPE_MAP.get(fact["fact_type"])
            if claim_type is None:
                continue
            subject_id = _subject_for_fact(fact)
            if subject_id is None:
                continue
            ref = DeterministicFactRef(
                fact_id=fact["fact_id"], claim_type=claim_type, subject_id=subject_id,
                value=fact["value"], resolved=fact["kind"] != "UNRESOLVED",
                source_module="app.plan_analysis.plan_facts", detail=fact,
            )
            self._facts_by_subject.setdefault(subject_id, []).append(ref)
            self._facts_by_claim_type.setdefault(claim_type, []).append(ref)

        for ce in self.component_evidence:
            ref = DeterministicFactRef(
                fact_id=ce["component_evidence_id"], claim_type="component_type",
                subject_id=ce["component_evidence_id"], value=ce.get("component_type"),
                resolved=ce.get("resolution") == "COMPONENT_FACT",
                source_module="app.plan_analysis.component_evidence", detail=ce,
            )
            self._facts_by_subject.setdefault(ref.subject_id, []).append(ref)
            self._facts_by_claim_type.setdefault(ref.claim_type, []).append(ref)

        for check in self.rule_checks:
            subject_id = check.get("inventory_id")
            if not subject_id:
                continue
            claim_type = f"rule:{check['rule_id']}"
            ref = DeterministicFactRef(
                fact_id=check["check_id"], claim_type=claim_type, subject_id=subject_id,
                value=check["result"], resolved=check["result"] != "NOT_ASSESSABLE",
                source_module="app.rules.engine", detail=check,
            )
            self._facts_by_subject.setdefault(subject_id, []).append(ref)
            self._facts_by_claim_type.setdefault(claim_type, []).append(ref)

    # ---- narrow query surface every agent uses instead of touching raw data ----

    def page(self, page_number: int) -> Optional[schema.PageAnalysis]:
        return self._pages_by_number.get(page_number)

    def all_pages(self) -> list[schema.PageAnalysis]:
        return list(self.doc.pages)

    def facts_for_subject(self, subject_id: str) -> list[DeterministicFactRef]:
        return list(self._facts_by_subject.get(subject_id, []))

    def facts_for_claim_type(self, claim_type: str) -> list[DeterministicFactRef]:
        return list(self._facts_by_claim_type.get(claim_type, []))

    def all_fact_refs(self) -> list[DeterministicFactRef]:
        """Every DeterministicFactRef this context indexed, flattened --
        used by evidence_merger.py to enumerate every (claim_type,
        subject_id) pair that has deterministic evidence, without reaching
        into this class's private indices."""
        out: list[DeterministicFactRef] = []
        for refs in self._facts_by_subject.values():
            out.extend(refs)
        return out

    def is_resolved(self, subject_id: str, claim_type: Optional[str] = None) -> bool:
        refs = self.facts_for_subject(subject_id)
        if claim_type is not None:
            refs = [r for r in refs if r.claim_type == claim_type]
        return any(r.resolved for r in refs)

    def nearby_text(self, page_number: int, bbox: tuple, margin_pt: float = 80.0) -> list[str]:
        """Deterministic text within `margin_pt` of `bbox` on one page,
        DISPLAY-space bbox in, DISPLAY-space spans compared (this engine's
        v0.1 TextSpan.bbox is already display space post-extraction -- see
        app/plan_analysis/text.py). Small and bounded, never the full page
        text dump."""
        page = self.page(page_number)
        if page is None:
            return []
        x0, y0, x1, y1 = bbox
        zone = (x0 - margin_pt, y0 - margin_pt, x1 + margin_pt, y1 + margin_pt)
        out = []
        for span in page.text_spans:
            sx0, sy0, sx1, sy1 = span.bbox
            if sx0 < zone[2] and zone[0] < sx1 and sy0 < zone[3] and zone[1] < sy1:
                out.append(span.text)
        return sorted(set(out))

    def render_crop_png(self, page_number: int, bbox: tuple, margin_fraction: float = CROP_MARGIN_FRACTION) -> Optional[bytes]:
        """Renders one small PNG crop directly from the source PDF -- the
        only place in this package that touches raster pixels. Deliberately
        self-contained (does not call another module's private renderer) so
        this package's own coordinate-space handling stays auditable in one
        place. Mirrors the rotation-matrix discipline documented in
        app/plan_analysis/component_facts.py's _display_clip_rect: v0.1
        bboxes are in the page's UNROTATED frame, while get_pixmap()'s clip
        is in the ROTATED/display frame."""
        if not self.pdf_bytes:
            return None
        pdf = pymupdf.open(stream=self.pdf_bytes, filetype="pdf")
        try:
            if page_number < 1 or page_number > pdf.page_count:
                return None
            page = pdf[page_number - 1]
            x0, y0, x1, y1 = pymupdf.Rect(bbox) * page.rotation_matrix
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            margin = margin_fraction * max(w, h)
            clip = pymupdf.Rect(x0 - margin, y0 - margin, x1 + margin, y1 + margin) & page.rect
            if clip.is_empty:
                return None
            zoom = RENDER_DPI / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
            return pixmap.tobytes("png")
        finally:
            pdf.close()

    def render_full_page_png(self, page_number: int, dpi: float = 110.0) -> Optional[bytes]:
        """Whole-page overview crop (low DPI -- this is for coarse plan-area
        classification, not symbol-level detail). Uses page.rect directly
        (already DISPLAY space), so unlike render_crop_png no rotation-matrix
        conversion is needed here."""
        if not self.pdf_bytes:
            return None
        pdf = pymupdf.open(stream=self.pdf_bytes, filetype="pdf")
        try:
            if page_number < 1 or page_number > pdf.page_count:
                return None
            page = pdf[page_number - 1]
            zoom = dpi / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("png")
        finally:
            pdf.close()
