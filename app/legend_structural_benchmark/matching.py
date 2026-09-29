"""Matches v0.1 schema-area symbol candidates against a plan's own legend
catalog (see `legend_extraction.py`), using ONLY structural-geometry
similarity for identity, with text and graph evidence recorded as SEPARATE
corroboration signals -- never blended into the identity score itself, so a
weak structural match can never be laundered into a confident one just
because a nearby label or graph position happens to look right (per the
explicit task instruction this benchmark was commissioned under: "Graph
evidence may SUPPORT a symbol's identity, but must never turn a weak
structural match into a confident one artificially").

Exclusions applied BEFORE matching (per the same instructions), each with
its own honest status rather than a silent drop:
- EXCLUDED_LEGEND_REGION: candidate bbox sits inside the legend itself.
- EXCLUDED_TITLEBLOCK_OR_TABLE: candidate bbox sits inside another dense
  text/table region `legend_detection.py` also flagged on this page (a
  project title block or a coded lookup table often independently passes
  the same "dense block of rows" test as a real legend -- see that
  module's own docstring; every such region found on the page, not only
  the one chosen as the real legend, is excluded here).
- EXCLUDED_WASTEWATER: the candidate's nearest associated text names a
  wastewater/drainage system. A fixture that ALSO happens to have a
  wastewater connection is never excluded by this rule -- only text naming
  the wastewater/drainage system itself with no accompanying fixture
  keyword.
- EXCLUDED_NO_STRUCTURE: this benchmark's own vector re-read of the
  candidate's bbox found zero line/curve/rect/quad items at all (a stray
  dimension number or scanning artifact, not a real drawn component).

Every remaining candidate gets a `evidence_status` of exactly one of
COMPONENT_FACT / COMPONENT_CANDIDATE / UNRESOLVED -- the same three-tier
vocabulary v0.2's `component_facts.py` and v0.3's `legend_intelligence.py`
already use, chosen deliberately so this benchmark's results are directly
comparable to theirs (see docs/v04-legend-structural-benchmark-report.md
section 8). The decision thresholds below were fixed BEFORE this benchmark
ever ran Phase A on real data and were NEVER adjusted after seeing Phase C's
scores against human labels.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

from app.plan_analysis import schema

from .legend_extraction import LegendEntryRecord, PlanLegend
from .vector_features import PageVectorIndex, StructuralFingerprint, build_page_index, compute_structural_fingerprint, structural_similarity

# Frozen decision thresholds -- see module docstring. Chosen by analogy to
# v0.2/v0.3's own precision-first stance (PLAN_FACT_SCORE_MIN=0.90,
# PLAN_FACT_MARGIN_MIN=0.12 there), adapted to this benchmark's different
# [0,1] structural-similarity scale, and additionally requiring at least one
# corroborating signal (text OR graph) for the top tier -- structural
# evidence alone, however high its score, is capped at COMPONENT_CANDIDATE.
COMPONENT_FACT_SCORE_MIN = 0.75
COMPONENT_FACT_MARGIN_MIN = 0.12
COMPONENT_CANDIDATE_SCORE_MIN = 0.60
TEXT_SUPPORT_MIN_FOR_FACT = 0.5
AMBIGUITY_MARGIN_MAX = 0.05

_FIXTURE_KEYWORDS = re.compile(r"\bwc\b|\bdusch|\bwaschtisch\b|\bk(ü|ue)che\b|\bsp(ü|ue)le\b|\blavabo\b")
_WASTEWATER_KEYWORDS = re.compile(
    r"schmutzwasser|abwasser|regenwasser|entw(ä|ae)sserung|fallstrang|fallleitung"
)


def _stable_match_id(plan_id: str, page: int, bbox: tuple) -> str:
    payload = f"{plan_id}|{page}|{bbox[0]:.3f}|{bbox[1]:.3f}|{bbox[2]:.3f}|{bbox[3]:.3f}"
    return "SM" + sha256(payload.encode("utf-8")).hexdigest()[:20]


def is_wastewater_text(label: Optional[str]) -> bool:
    if not label:
        return False
    low = label.casefold()
    if not _WASTEWATER_KEYWORDS.search(low):
        return False
    return not _FIXTURE_KEYWORDS.search(low)  # a fixture's own wastewater connection is never excluded


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def graph_support_category(sym: schema.SymbolCandidate, nodes_by_id: dict[str, schema.GraphNode]) -> str:
    """Categorical, never a number: whether the candidate's own detected
    ports sit at a real topological feature, or the candidate is merely
    near graph geometry with no touching port at all. Spatial proximity
    alone is never treated as proof of connection (matches v0.1's own
    `symbols.detect_symbols`, which only ever links a port when a graph
    node's point falls inside/near the candidate's own bbox -- this
    function only reads that existing, already-conservative linkage)."""
    if not sym.port_node_ids:
        return "proximity_only"
    degrees = [nodes_by_id[nid].degree for nid in sym.port_node_ids if nid in nodes_by_id]
    if any(d == 1 for d in degrees):
        return "at_endpoint"
    if any(d >= 3 for d in degrees):
        return "at_branch"
    if degrees:
        return "on_line"
    return "proximity_only"


def text_support_score(sym_label: Optional[str], legend_label: str) -> Optional[float]:
    """difflib ratio (deterministic, no ML) between the candidate's own
    v0.1-associated nearest text and the legend entry's normalized label.
    None (not 0.0) when the candidate has no associated text at all -- an
    absence of evidence is recorded as such, never silently scored as
    'definitely doesn't match'."""
    if not sym_label:
        return None
    a = re.sub(r"\s+", " ", sym_label).strip().casefold()
    b = legend_label
    if not a or not b:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 4)


@dataclass
class MatchRecord:
    stable_match_id: str
    plan_id: str
    page: int
    bbox: tuple
    status: str  # EXCLUDED_* | COMPONENT_FACT | COMPONENT_CANDIDATE | UNRESOLVED
    exclusion_reason: Optional[str] = None
    matched_legend_entry: Optional[str] = None
    component_type: Optional[str] = None
    structural_score: Optional[float] = None
    structural_subscores: Optional[dict] = None
    text_support: Optional[float] = None
    graph_support: Optional[str] = None
    competing_match: Optional[str] = None
    competing_score: Optional[float] = None
    ambiguity: bool = False
    port_node_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stable_match_id": self.stable_match_id,
            "plan_id": self.plan_id,
            "page": self.page,
            "bbox": list(self.bbox),
            "status": self.status,
            "exclusion_reason": self.exclusion_reason,
            "matched_legend_entry": self.matched_legend_entry,
            "component_type": self.component_type,
            "structural_score": self.structural_score,
            "structural_subscores": self.structural_subscores,
            "text_support": self.text_support,
            "graph_support": self.graph_support,
            "competing_match": self.competing_match,
            "competing_score": self.competing_score,
            "ambiguity": self.ambiguity,
            "port_node_ids": self.port_node_ids,
        }


def match_page(
    plan_id: str,
    page_analysis: schema.PageAnalysis,
    index: PageVectorIndex,
    plan_legend: PlanLegend,
    other_dense_region_bboxes: list[tuple],
) -> list[MatchRecord]:
    catalog: dict[str, LegendEntryRecord] = plan_legend.usable_catalog()
    nodes_by_id = {n.id: n for n in page_analysis.graph.nodes}
    excluded_bboxes = list(other_dense_region_bboxes)  # already in raw/native space -- see Phase A caller
    if plan_legend.legend_bbox_raw is not None:
        excluded_bboxes.append(plan_legend.legend_bbox_raw)

    records: list[MatchRecord] = []
    for sym in page_analysis.symbols:
        match_id = _stable_match_id(plan_id, page_analysis.page_number, sym.bbox)

        if any(_bbox_overlap(sym.bbox, region) for region in excluded_bboxes):
            reason = "EXCLUDED_LEGEND_REGION" if plan_legend.legend_bbox_raw and _bbox_overlap(sym.bbox, plan_legend.legend_bbox_raw) else "EXCLUDED_TITLEBLOCK_OR_TABLE"
            records.append(MatchRecord(match_id, plan_id, page_analysis.page_number, sym.bbox, status=reason, exclusion_reason=reason, port_node_ids=sym.port_node_ids))
            continue

        if is_wastewater_text(sym.label):
            records.append(MatchRecord(match_id, plan_id, page_analysis.page_number, sym.bbox, status="EXCLUDED_WASTEWATER", exclusion_reason=f"associated text {sym.label!r} names a wastewater/drainage system", port_node_ids=sym.port_node_ids))
            continue

        fp = compute_structural_fingerprint(index, sym.bbox)
        if fp.total_item_count == 0:
            records.append(MatchRecord(match_id, plan_id, page_analysis.page_number, sym.bbox, status="EXCLUDED_NO_STRUCTURE", exclusion_reason="zero line/curve/rect/quad items found in this bbox", port_node_ids=sym.port_node_ids))
            continue

        if not catalog:
            records.append(MatchRecord(match_id, plan_id, page_analysis.page_number, sym.bbox, status="UNRESOLVED", exclusion_reason="no usable legend entries in this plan's catalog", port_node_ids=sym.port_node_ids))
            continue

        scored = []
        for entry_id, entry in catalog.items():
            sim = structural_similarity(fp, entry.fingerprint)
            scored.append((sim["structural_score"], entry_id, entry, sim))
        scored.sort(key=lambda t: -t[0])

        top_score, top_id, top_entry, top_sim = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = top_score - second_score

        text_sup = text_support_score(sym.label, top_entry.normalized_label)
        graph_sup = graph_support_category(sym, nodes_by_id)
        ambiguous = len(scored) > 1 and margin < AMBIGUITY_MARGIN_MAX

        corroborated = (text_sup is not None and text_sup >= TEXT_SUPPORT_MIN_FOR_FACT) or graph_sup in ("at_endpoint", "at_branch")

        if top_score >= COMPONENT_FACT_SCORE_MIN and margin >= COMPONENT_FACT_MARGIN_MIN and corroborated:
            status = "COMPONENT_FACT"
        elif top_score >= COMPONENT_CANDIDATE_SCORE_MIN:
            status = "COMPONENT_CANDIDATE"
        else:
            status = "UNRESOLVED"

        records.append(
            MatchRecord(
                stable_match_id=match_id,
                plan_id=plan_id,
                page=page_analysis.page_number,
                bbox=sym.bbox,
                status=status,
                matched_legend_entry=top_id if status != "UNRESOLVED" else None,
                component_type=top_entry.component_type if status != "UNRESOLVED" else None,
                structural_score=top_score,
                structural_subscores=top_sim,
                text_support=text_sup,
                graph_support=graph_sup,
                competing_match=scored[1][1] if len(scored) > 1 else None,
                competing_score=second_score if len(scored) > 1 else None,
                ambiguity=ambiguous,
                port_node_ids=sym.port_node_ids,
            )
        )
    return records
