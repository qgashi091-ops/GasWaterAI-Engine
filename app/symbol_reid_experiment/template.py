"""Builds a structural TEMPLATE for one chosen legend symbol -- its own
vector geometry plus its own legend text -- to be searched for elsewhere on
the same page. Never invents a template from text alone: `native_bbox` must
be an ACTUAL legend symbol_bbox already extracted by the existing v0.1/v0.3
legend-detection code (see app/legend_symbol_dataset/extract.py), and the
template is rejected (raises) if that bbox does not itself pass
`graphical_symbol_gate` -- the same bar this experiment holds every
candidate OCCURRENCE to.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.legend_structural_benchmark.vector_features import (
    PageVectorIndex,
    StructuralFingerprint,
    compute_structural_fingerprint,
)
from app.legend_symbol_dataset.graphical_symbol_gate import (
    compute_text_area_ratio,
    evaluate_graphical_symbol,
)

BBox = tuple[float, float, float, float]


@dataclass
class SymbolTemplate:
    legend_symbol_id: str
    raw_legend_text: str
    native_bbox: BBox
    fingerprint: StructuralFingerprint

    @property
    def width(self) -> float:
        return self.native_bbox[2] - self.native_bbox[0]

    @property
    def height(self) -> float:
        return self.native_bbox[3] - self.native_bbox[1]


def build_template(
    legend_symbol_id: str,
    raw_legend_text: str,
    native_bbox: BBox,
    page_index: PageVectorIndex,
    text_spans_native: list,
) -> SymbolTemplate:
    fingerprint = compute_structural_fingerprint(page_index, native_bbox)
    text_area_ratio = compute_text_area_ratio(native_bbox, text_spans_native)
    gate = evaluate_graphical_symbol(fingerprint, text_area_ratio)
    if not gate.passes:
        raise ValueError(
            f"{legend_symbol_id} ({raw_legend_text!r}) does not itself pass "
            f"graphical_symbol_gate (reason={gate.reason}); refusing to build "
            f"a search template from it."
        )
    return SymbolTemplate(
        legend_symbol_id=legend_symbol_id,
        raw_legend_text=raw_legend_text,
        native_bbox=native_bbox,
        fingerprint=fingerprint,
    )
