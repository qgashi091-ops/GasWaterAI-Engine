"""Unit tests for symbol_templates.py (v0.3 Phase 3): only USABLE_TEMPLATE,
non-swatch entries become searchable templates, template ids are stable,
and a plan-specific template matches its own source shape at any of the
four cardinal rotations -- the same rotation-invariance guarantee
symbol_library.py already provides for the 58 generic templates, now
extended to document-local ones.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.plan_analysis.legend_entries import LegendEntry
from app.plan_analysis.legend_intelligence import match_plan_templates
from app.plan_analysis.symbol_templates import build_templates


def _entry(label, classification="USABLE_TEMPLATE", is_line_style_swatch=False, canon=None):
    if canon is None and classification == "USABLE_TEMPLATE" and not is_line_style_swatch:
        canon = np.zeros((128, 128), dtype=np.uint8)
        canon[30:100, 40:90] = 255
    return LegendEntry(
        legend_entry_id=f"LE_{label}", legend_id="LG1_0", page=1,
        bbox=(0.0, 0.0, 10.0, 10.0), symbol_bbox=(0.0, 0.0, 10.0, 10.0),
        label_text=label, normalized_label=label, label_source="native",
        classification=classification, ambiguity_reason=None,
        symbol_ink_present=classification == "USABLE_TEMPLATE",
        is_line_style_swatch=is_line_style_swatch, symbol_canonical=canon,
    )


def test_only_usable_non_swatch_entries_become_templates():
    entries = [
        _entry("ventil"),
        _entry("warmwasser", is_line_style_swatch=True),
        _entry("some note", classification="TEXT_ONLY", canon=None),
        _entry("ambiguous row", classification="AMBIGUOUS", canon=None),
        _entry("verteilbatterie row", classification="REJECTED", canon=None),
    ]
    templates = build_templates(entries)
    labels = {t.label for t in templates}
    assert labels == {"ventil"}


def test_template_ids_are_stable_and_distinct_per_entry():
    entries = [_entry("ventil"), _entry("pumpe")]
    t1 = build_templates(entries)
    t2 = build_templates(entries)
    assert {t.template_id for t in t1} == {t.template_id for t in t2}
    assert len({t.template_id for t in t1}) == 2


def test_a_plan_specific_template_recognizes_its_own_shape_rotated_90_degrees():
    entries = [_entry("meinsymbol")]
    templates = build_templates(entries)
    source_canon = entries[0].symbol_canonical
    rotated_candidate = cv2.rotate(source_canon, cv2.ROTATE_90_CLOCKWISE)

    scores = match_plan_templates(rotated_candidate, templates)
    assert scores[0].label == "meinsymbol"
    assert scores[0].score > 0.95
