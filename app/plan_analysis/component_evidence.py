"""Component evidence fusion -- ENGINE v1 epic, section 3: ONE canonical
component-resolution layer, combining (in priority/evidence order):

  1. explicit native PDF text + geometric association
  2. deterministic parser/vector evidence (generic shape template match)
  3. trained detector (Detector v1)
  4. plan-specific legend evidence
  5. controlled multimodal vision fallback

Sources 1, 2 and 4 are NOT reimplemented here -- they already exist,
unchanged, as `legend_intelligence.build_legend_intelligence`'s
HybridComponentFact per candidate (generic shape match = source 2,
plan-specific legend match = source 4, and its own `corroboration_status`
already records when a plan-specific match was ALSO corroborated by nearby
text = source 1). This module's job is strictly additive: layer the new
Detector v1 (source 3) and, only when still unresolved, a controlled vision
fallback (source 5) on top of that unchanged output, and make one final,
auditable decision -- never silently collapsing a disagreement between
sources.

OUTPUT KINDS (renames v0.2/v0.3's COMPONENT_FACT/COMPONENT_CANDIDATE/
UNRESOLVED to this epic's own vocabulary; identical ordering semantics):
  COMPONENT_FACT       -- strong, uncontested evidence (kept the strong
                          decision the older layer already made, OR two
                          independent sources agree).
  COMPONENT_SUPPORTED  -- some real evidence, not yet strong enough to be
                          a FACT, OR sources actively disagree (a conflict
                          is itself reported, never hidden by picking one).
  COMPONENT_UNRESOLVED -- no source produced usable evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from app.detector.infer import predict as detector_predict
from app.plan_analysis.component_facts import _render_symbol_crop
from app.vision_fallback.interface import VisionFallbackProvider, VisionFallbackResult

# Detector v1's own supported taxonomy -- see docs/v04-detector-poc-phase0-report.md
# and data/dev_plans_v04/detector_v1/metrics.json. Every other component type
# stays entirely outside Detector v1's opinion (source 3 simply has nothing
# to say about it), never forced through it.
DETECTOR_SUPPORTED_CLASSES = {"kueche", "wc_up"}
DETECTOR_BACKGROUND_LABEL = "background"

_DISPLAY_NAME = {"kueche": "küche", "wc_up": "wc up"}


@dataclass
class ComponentEvidence:
    component_evidence_id: str
    page: int
    bbox: tuple
    resolution: str  # "COMPONENT_FACT" | "COMPONENT_SUPPORTED" | "COMPONENT_UNRESOLVED"
    component_type: str | None
    graph_association: dict
    provenance: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "component_evidence_id": self.component_evidence_id,
            "page": self.page, "bbox": list(self.bbox),
            "resolution": self.resolution, "component_type": self.component_type,
            "graph_association": self.graph_association,
            "provenance": self.provenance, "conflicts": self.conflicts,
        }


def _fuse_one(hybrid_fact: dict, detector_available: bool, detector_class: str | None,
              detector_confidence: float | None, vision_result: VisionFallbackResult | None) -> ComponentEvidence:
    provenance: list[dict] = []
    candidates: dict[str, list[dict]] = {}  # component_type -> list of provenance entries proposing it

    def _propose(component_type: str, entry: dict):
        candidates.setdefault(component_type, []).append(entry)

    legend_kind = hybrid_fact["kind"]  # COMPONENT_FACT | COMPONENT_CANDIDATE | UNRESOLVED (v0.3's own vocabulary)
    legend_label = hybrid_fact.get("symbol_label")
    if legend_label and legend_kind in ("COMPONENT_FACT", "COMPONENT_CANDIDATE"):
        entry = {
            "source": "text_and_vector_and_legend_evidence", "value": legend_label,
            "strength": "strong" if legend_kind == "COMPONENT_FACT" else "moderate",
            "corroboration_status": hybrid_fact.get("corroboration_status"),
            "hybrid_conflict": hybrid_fact.get("hybrid_conflict", False),
        }
        provenance.append(entry)
        _propose(legend_label, entry)

    if detector_available and detector_class is not None and detector_class != DETECTOR_BACKGROUND_LABEL:
        entry = {
            "source": "trained_detector", "value": detector_class,
            "strength": "moderate", "confidence": detector_confidence,
        }
        provenance.append(entry)
        _propose(detector_class, entry)
    elif detector_available:
        provenance.append({"source": "trained_detector", "value": None, "note": "predicted background"})

    if vision_result is not None:
        if vision_result.available and vision_result.component_type:
            entry = {
                "source": "vision_fallback", "value": vision_result.component_type,
                "strength": "weak", "confidence": vision_result.confidence,
                "model": vision_result.model, "reasoning": vision_result.reasoning,
            }
            provenance.append(entry)
            _propose(vision_result.component_type, entry)
        else:
            provenance.append({"source": "vision_fallback", "value": None, "error": vision_result.error})

    conflicts = []
    resolution, component_type = "COMPONENT_UNRESOLVED", None
    if len(candidates) == 0:
        pass
    elif len(candidates) == 1:
        (component_type, entries), = candidates.items()
        has_strong = any(e["strength"] == "strong" for e in entries)
        has_corroboration = len(entries) >= 2  # independent sources agreeing
        resolution = "COMPONENT_FACT" if (has_strong or has_corroboration) else "COMPONENT_SUPPORTED"
    else:
        # Sources disagree on the component's identity -- reported as a
        # conflict, never silently resolved by picking the "best" one.
        conflicts.append({
            "candidates": {ct: [e["source"] for e in entries] for ct, entries in candidates.items()},
        })
        # The strongest single proposal is surfaced as the working type
        # (so downstream inventory has SOMETHING to key on), but the
        # resolution is capped at COMPONENT_SUPPORTED and the conflict is
        # always attached -- a caller must look at `conflicts`, not just
        # `component_type`, before trusting this.
        best_type, best_entries = max(candidates.items(), key=lambda kv: (
            any(e["strength"] == "strong" for e in kv[1]), len(kv[1]),
        ))
        component_type, resolution = best_type, "COMPONENT_SUPPORTED"

    return ComponentEvidence(
        component_evidence_id=hybrid_fact["component_fact_id"], page=hybrid_fact["page"],
        bbox=tuple(hybrid_fact["bbox"]), resolution=resolution, component_type=component_type,
        graph_association=hybrid_fact.get("graph_association", {}),
        provenance=provenance, conflicts=conflicts,
    )


def build_component_evidence(
    pdf_bytes: bytes, doc, legend_intelligence_result: dict,
    vision_provider: VisionFallbackProvider | None = None,
) -> dict:
    """Pure function of (pdf_bytes, doc, legend_intelligence_result) plus an
    OPTIONAL vision_provider (omit it -- the default -- to run detector-only
    fusion with zero network calls, e.g. for the reproducibility/regression
    suite; pass a real provider only when the deployment has one
    configured)."""
    hybrid_by_key = {
        (f["page"], tuple(round(v, 1) for v in f["bbox"])): f
        for f in legend_intelligence_result["component_facts"]
    }

    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        results: list[ComponentEvidence] = []
        for page_model in doc.pages:
            page = pdf[page_model.page_number - 1]
            for sym in page_model.symbols:
                key = (page_model.page_number, tuple(round(v, 1) for v in sym.bbox))
                hybrid_fact = hybrid_by_key.get(key)
                if hybrid_fact is None:
                    continue  # excluded upstream (inside a legend region) -- never a candidate here either, same rule as component_facts.py

                crop = _render_symbol_crop(page, sym.bbox)
                detector_available, detector_class, detector_confidence = False, None, None
                if crop is not None:
                    pred = detector_predict(crop)
                    detector_available = pred.available
                    detector_class = pred.predicted_class
                    detector_confidence = pred.probabilities.get(pred.predicted_class) if pred.predicted_class else None

                vision_result = None
                legend_kind = hybrid_fact["kind"]
                needs_fallback = (
                    vision_provider is not None
                    and legend_kind == "UNRESOLVED"
                    and (not detector_available or detector_class in (None, DETECTOR_BACKGROUND_LABEL))
                )
                if needs_fallback and crop is not None:
                    import cv2
                    ok, encoded = cv2.imencode(".png", crop)
                    if ok:
                        vision_result = vision_provider.classify_region(
                            encoded.tobytes(), "image/png",
                            candidate_labels=sorted(_DISPLAY_NAME.values()),
                            context={"page": page_model.page_number},
                        )

                results.append(_fuse_one(hybrid_fact, detector_available, detector_class, detector_confidence, vision_result))
    finally:
        pdf.close()

    kind_counts = {"COMPONENT_FACT": 0, "COMPONENT_SUPPORTED": 0, "COMPONENT_UNRESOLVED": 0}
    for r in results:
        kind_counts[r.resolution] += 1

    return {
        "evidence": [r.to_dict() for r in results],
        "stats": {
            "components_evaluated": len(results), "by_resolution": kind_counts,
            "conflicts_found": sum(1 for r in results if r.conflicts),
            "vision_fallback_invoked": vision_provider is not None,
        },
    }
