"""Plan-specific symbol template construction -- v0.3, new (Phase 3). Turns
a legend's own USABLE_TEMPLATE entries (legend_entries.py) into
document-local match templates, reusing the exact canonicalization and
rotation-invariance approach already proven in v0.2's symbol_library.py --
deliberately, not reinvented: nothing here touches or overwrites the 58
generic reference templates (symbol_library.SymbolLibrary), which stay
separate, unmodified reference data. A plan-specific template is
document-local evidence only -- it exists to be matched within the SAME
plan it was extracted from, never persisted as a new generic reference
symbol.

Most of "normalize for matching" (crop tightly; normalize scale; preserve
structural line geometry) is already done by the time an entry reaches this
module: legend_entries.py builds every USABLE_TEMPLATE entry's
`symbol_canonical` via symbol_library.canonicalize() itself (ink-threshold,
crop to ink bbox, aspect-preserving resize onto a fixed canvas) -- the same
function, not a re-implementation, so a plan-specific template and a
generic template are directly comparable in shape. "Remove surrounding
label text" and "avoid learning table borders" are also already enforced
upstream, by construction: the lookback zone a template's ink comes from
stops exactly at the label's own left edge, and a border-shaped ink zone is
rejected in Phase 2, never reaching this module as USABLE_TEMPLATE.

What this module adds: rotation normalization -- pre-rotating each
template at 0/90/180/270 degrees so Phase 4 can match a candidate at
whatever orientation it was actually drawn in, exactly mirroring
SymbolLibrary's own rotation handling.

Line-style/pipe-type swatches (LegendEntry.is_line_style_swatch) are
intentionally excluded from becoming searchable templates here -- matching
a colored/dashed line sample against the drawing's own pipe segments would
match nearly every drawn pipe of that type, the opposite of "precision
before recall". They remain visible in legend_entries.py's own output for
provenance; this module just never turns them into a template.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import cv2
import numpy as np

from .legend_entries import LegendEntry

ROTATIONS = (0, 90, 180, 270)


def _stable_id(*parts: str) -> str:
    return "PT" + sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


@dataclass
class PlanSpecificTemplate:
    template_id: str
    legend_entry_id: str
    legend_id: str
    label: str
    source_bbox: tuple
    rotations: dict  # {0, 90, 180, 270: np.ndarray canonical mask}

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "legend_entry_id": self.legend_entry_id,
            "legend_id": self.legend_id,
            "label": self.label,
            "source_bbox": list(self.source_bbox),
        }


def build_templates(entries: list[LegendEntry]) -> list[PlanSpecificTemplate]:
    """Pure function of already-classified legend entries. Only
    USABLE_TEMPLATE, non-line-swatch entries with a canonical mask produce a
    searchable template -- everything else (TEXT_ONLY / AMBIGUOUS /
    REJECTED, and line-style swatches) is intentionally excluded, not
    silently dropped -- see legend_entries.py for why each was excluded."""
    templates: list[PlanSpecificTemplate] = []
    for e in entries:
        if e.classification != "USABLE_TEMPLATE" or e.is_line_style_swatch or e.symbol_canonical is None:
            continue
        canon = e.symbol_canonical
        rotations = {0: canon}
        cur = canon
        for angle in (90, 180, 270):
            cur = cv2.rotate(cur, cv2.ROTATE_90_CLOCKWISE)
            rotations[angle] = cur
        templates.append(PlanSpecificTemplate(
            template_id=_stable_id(e.legend_entry_id),
            legend_entry_id=e.legend_entry_id,
            legend_id=e.legend_id,
            label=e.normalized_label,
            source_bbox=e.symbol_bbox or e.bbox,
            rotations=rotations,
        ))
    return templates
