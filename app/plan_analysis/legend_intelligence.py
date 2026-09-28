"""Plan-specific legend intelligence -- v0.3, new (Phases 4-6 + orchestration).

Phase 4 -- SEARCH DRAWING AREA: matches each of v0.3's plan-specific
templates (symbol_templates.py) against every v0.1-detected symbol
candidate that lies OUTSIDE every detected legend region (never inside --
a legend row would otherwise trivially "recognize itself"). Reuses v0.2's
own rotation-invariant `cv2.matchTemplate` approach and its raster-render
helper (`component_facts._render_symbol_crop`, including its rotation-frame
fix) rather than re-implementing either.

Phase 5 -- COMPONENT FACTS: a plan-specific match becomes COMPONENT_FACT
only under the SAME precision-first discipline as v0.2 (score/margin
thresholds copied verbatim, not loosened -- see PLAN_FACT_SCORE_MIN /
PLAN_FACT_MARGIN_MIN below), plus: the match must sit outside the legend
(guaranteed by Phase 4's own exclusion) and its graph association must be
geometrically plausible (reuses v0.1's own proven ports/edges, never
invented -- component_facts._graph_association, unchanged).

Phase 6 -- HYBRID RECOGNITION: combines, per symbol candidate, (1) the
plan-specific legend match, (2) the existing v0.2 generic-library match,
(3) any text associated with the same symbol (v0.1's association.py,
unchanged), and (4) the deterministic graph association. An unambiguous
plan-specific match is never overridden by a weaker generic one -- but a
disagreement between the two is always recorded explicitly
(`hybrid_conflict`), never silently resolved by picking a side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

import cv2
import numpy as np
import pymupdf

from . import schema
from .component_facts import _decide as _generic_decide
from .component_facts import _graph_association, _render_symbol_crop
from .legend_detection import detect_legend_candidates
from .legend_entries import classify_candidate_rows
from .symbol_library import canonicalize, get_library
from .symbol_templates import PlanSpecificTemplate, build_templates

# Copied verbatim from component_facts.py's own precision-first thresholds
# (docs/symbol-audit.md / docs/w003-component-poc-report.md) -- explicit
# task requirement: "Do NOT lower recognition thresholds merely to create
# detections." A plan-specific match earns COMPONENT_FACT under exactly the
# same bar a generic-library match would.
PLAN_FACT_SCORE_MIN = 0.90
PLAN_FACT_MARGIN_MIN = 0.12
PLAN_CANDIDATE_SCORE_MIN = 0.75


def _stable_id(*parts: str) -> str:
    return "HC" + sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


@dataclass
class PlanMatchScore:
    template_id: str
    legend_entry_id: str
    label: str
    score: float
    rotation: int


@dataclass
class HybridComponentFact:
    component_fact_id: str
    kind: str  # COMPONENT_FACT | COMPONENT_CANDIDATE | UNRESOLVED
    page: int
    bbox: tuple
    centroid: tuple
    symbol_label: Optional[str]
    recognition_method: str
    confidence: float
    graph_association: dict
    supporting_evidence: dict
    plan_specific_evidence: Optional[dict]
    generic_evidence: Optional[dict]
    hybrid_conflict: bool
    corroboration_status: str

    def to_dict(self) -> dict:
        return {
            "component_fact_id": self.component_fact_id,
            "kind": self.kind,
            "page": self.page,
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "symbol_label": self.symbol_label,
            "recognition_method": self.recognition_method,
            "confidence": self.confidence,
            "graph_association": self.graph_association,
            "supporting_evidence": self.supporting_evidence,
            "plan_specific_evidence": self.plan_specific_evidence,
            "generic_evidence": self.generic_evidence,
            "hybrid_conflict": self.hybrid_conflict,
            "corroboration_status": self.corroboration_status,
        }


def _bbox_overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _symbol_in_any_legend(page: "pymupdf.Page", sym_bbox_native: tuple, legend_bboxes_display: list) -> bool:
    """sym_bbox_native is v0.1's own (unrotated-frame) bbox; transformed
    through the SAME page.rotation_matrix component_facts.py already uses,
    so it compares apples-to-apples against legend_detection.py's
    DISPLAY-space bboxes."""
    if not legend_bboxes_display:
        return False
    r = pymupdf.Rect(sym_bbox_native) * page.rotation_matrix
    disp = (r.x0, r.y0, r.x1, r.y1)
    return any(_bbox_overlaps(disp, lb) for lb in legend_bboxes_display)


def match_plan_templates(candidate_canon: Optional[np.ndarray], templates: list[PlanSpecificTemplate]) -> list[PlanMatchScore]:
    if candidate_canon is None or candidate_canon.std() == 0 or not templates:
        return []
    cf = candidate_canon.astype(np.float32)
    results: list[PlanMatchScore] = []
    for t in templates:
        best_score, best_rot = -1.0, 0
        for angle, mask in t.rotations.items():
            score = float(cv2.matchTemplate(cf, mask.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_score, best_rot = score, angle
        results.append(PlanMatchScore(t.template_id, t.legend_entry_id, t.label, best_score, best_rot))
    results.sort(key=lambda r: -r.score)
    return results


def _text_corroboration(sym: schema.SymbolCandidate, associations: list, text_spans_by_id: dict, label: str) -> bool:
    """Non-gating, purely reported evidence (Phase 6 evaluation criterion
    "nearby text"): does any text v0.1 already associated with this exact
    symbol textually contain the matched legend label? Never used to
    promote a kind on its own -- shape evidence must already clear its own
    threshold; this only gets recorded alongside it."""
    label_key = label.strip().casefold()
    if not label_key:
        return False
    for assoc in associations:
        if assoc.target_type != "symbol" or assoc.target_id != sym.id:
            continue
        span = text_spans_by_id.get(assoc.text_id)
        if span and label_key in span.text.strip().casefold():
            return True
    return False


def _decide_hybrid(sym: schema.SymbolCandidate, page_number: int, plan_scores: list, generic_scores: list,
                    library, graph_assoc: dict, text_corroborated: bool) -> HybridComponentFact:
    base = dict(
        component_fact_id=_stable_id(str(page_number), sym.id, f"{sym.bbox}"),
        page=page_number, bbox=sym.bbox, centroid=sym.centroid,
        graph_association=graph_assoc,
    )
    generic_result = _generic_decide(sym, page_number, generic_scores, library, graph_assoc)
    generic_evidence = {
        "kind": generic_result.kind, "symbol_name": generic_result.symbol_name,
        "confidence": generic_result.confidence, **generic_result.supporting_evidence,
    } if generic_scores else None

    plan_top = plan_scores[0] if plan_scores else None
    plan_second = plan_scores[1] if len(plan_scores) > 1 else None
    plan_margin = (plan_top.score - (plan_second.score if plan_second else 0.0)) if plan_top else 0.0
    plan_evidence = None
    if plan_top:
        plan_evidence = {
            "legend_entry_id": plan_top.legend_entry_id, "label": plan_top.label,
            "best_score": round(plan_top.score, 4),
            "second_best_score": round(plan_second.score, 4) if plan_second else None,
            "margin": round(plan_margin, 4), "rotation_degrees": plan_top.rotation,
            "text_corroborated": text_corroborated,
        }

    conflict = bool(
        plan_top and generic_result.kind != "UNRESOLVED" and generic_result.symbol_name
        and generic_result.symbol_name.strip().casefold() != plan_top.label.strip().casefold()
    )

    if plan_top and plan_top.score >= PLAN_FACT_SCORE_MIN and plan_margin >= PLAN_FACT_MARGIN_MIN:
        # An unambiguous plan-specific match is never overridden by a
        # weaker generic one -- but any disagreement is recorded, not hidden.
        return HybridComponentFact(
            **base, kind="COMPONENT_FACT", symbol_label=plan_top.label,
            recognition_method="plan_specific_legend_template_match",
            confidence=plan_top.score, supporting_evidence=plan_evidence,
            plan_specific_evidence=plan_evidence, generic_evidence=generic_evidence,
            hybrid_conflict=conflict,
            corroboration_status="plan_specific_shape_evidence_sufficient" + ("+text" if text_corroborated else ""),
        )

    if plan_top and plan_top.score >= PLAN_CANDIDATE_SCORE_MIN:
        # Weak/ambiguous plan-specific evidence: promote to COMPONENT_FACT
        # only if the generic path ALSO independently reached FACT for the
        # very same identity (agreement increases evidence strength); never
        # let this weaker plan-specific signal alone force a FACT.
        if generic_result.kind == "COMPONENT_FACT" and not conflict:
            return HybridComponentFact(
                **base, kind="COMPONENT_FACT", symbol_label=generic_result.symbol_name,
                recognition_method="hybrid_plan_specific_and_generic_agreement",
                confidence=max(plan_top.score, generic_result.confidence),
                supporting_evidence={**generic_result.supporting_evidence, "plan_specific": plan_evidence},
                plan_specific_evidence=plan_evidence, generic_evidence=generic_evidence,
                hybrid_conflict=False, corroboration_status="plan_specific_and_generic_agree",
            )
        return HybridComponentFact(
            **base, kind="COMPONENT_CANDIDATE", symbol_label=plan_top.label,
            recognition_method="plan_specific_legend_template_match",
            confidence=plan_top.score, supporting_evidence=plan_evidence,
            plan_specific_evidence=plan_evidence, generic_evidence=generic_evidence,
            hybrid_conflict=conflict,
            corroboration_status="plan_specific_below_fact_threshold",
        )

    # No usable plan-specific evidence: fall back to the unchanged v0.2
    # generic decision entirely -- v0.3 never removes v0.2 evidence, only adds to it.
    return HybridComponentFact(
        **base, kind=generic_result.kind, symbol_label=generic_result.symbol_name,
        recognition_method=generic_result.recognition_method if generic_result.kind != "UNRESOLVED" else "none",
        confidence=generic_result.confidence, supporting_evidence=generic_result.supporting_evidence,
        plan_specific_evidence=plan_evidence, generic_evidence=generic_evidence,
        hybrid_conflict=False, corroboration_status=generic_result.corroboration_status,
    )


@dataclass
class LegendIntelligenceResult:
    legend_candidates: list  # list[LegendCandidate.to_dict()]
    legend_entries: list  # list[LegendEntry.to_dict()]
    templates_built: int
    component_facts: list  # list[HybridComponentFact.to_dict()]
    stats: dict


def build_legend_intelligence(pdf_bytes: bytes, doc: schema.DocumentAnalysis) -> dict:
    """Pure function of (pdf_bytes, doc) -- deterministic given identical
    input, exactly like build_component_facts. Runs Phases 1-6 for every
    page and returns a single JSON-ready dict."""
    library = get_library()
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        all_legend_dicts: list[dict] = []
        all_entry_dicts: list[dict] = []
        all_facts: list[HybridComponentFact] = []
        total_templates = 0

        for page_model in doc.pages:
            page = pdf[page_model.page_number - 1]
            text_spans_by_id = {s.id: s for s in page_model.text_spans}

            legend_candidates = detect_legend_candidates(page_model, rotation_matrix=page.rotation_matrix)
            all_legend_dicts.extend(c.to_dict() for c in legend_candidates)
            legend_bboxes_display = [c.bbox for c in legend_candidates]

            page_templates: list[PlanSpecificTemplate] = []
            for cand in legend_candidates:
                entries, is_symbol_legend = classify_candidate_rows(page, cand)
                all_entry_dicts.extend(e.to_dict() for e in entries)
                if is_symbol_legend:
                    page_templates.extend(build_templates(entries))
            total_templates += len(page_templates)

            page_diagonal = (page_model.width ** 2 + page_model.height ** 2) ** 0.5
            seen_bbox_keys: set[tuple] = set()
            for sym in page_model.symbols:
                bbox_key = (page_model.page_number,) + tuple(round(v, 1) for v in sym.bbox)
                if bbox_key in seen_bbox_keys:
                    continue
                seen_bbox_keys.add(bbox_key)

                if _symbol_in_any_legend(page, sym.bbox, legend_bboxes_display):
                    continue  # Phase 4: never match a legend entry against itself

                crop_gray = _render_symbol_crop(page, sym.bbox)
                candidate_canon = canonicalize(crop_gray) if crop_gray is not None else None
                plan_scores = match_plan_templates(candidate_canon, page_templates)
                generic_scores = library.match(candidate_canon)
                graph_assoc = _graph_association(sym, page_model.graph.edges, page_diagonal)

                text_corroborated = False
                if plan_scores:
                    text_corroborated = _text_corroboration(
                        sym, page_model.associations, text_spans_by_id, plan_scores[0].label
                    )

                all_facts.append(_decide_hybrid(
                    sym, page_model.page_number, plan_scores, generic_scores,
                    library, graph_assoc, text_corroborated,
                ))
    finally:
        pdf.close()

    kind_counts = {"COMPONENT_FACT": 0, "COMPONENT_CANDIDATE": 0, "UNRESOLVED": 0}
    for f in all_facts:
        kind_counts[f.kind] += 1
    conflict_count = sum(1 for f in all_facts if f.hybrid_conflict)

    return {
        "legend_candidates": all_legend_dicts,
        "legend_entries": all_entry_dicts,
        "templates_built": total_templates,
        "component_facts": [f.to_dict() for f in all_facts],
        "stats": {
            "legend_candidates_detected": len(all_legend_dicts),
            "legend_entries_extracted": len(all_entry_dicts),
            "plan_specific_templates_built": total_templates,
            "components_evaluated": len(all_facts),
            "by_kind": kind_counts,
            "hybrid_conflicts": conflict_count,
        },
    }
