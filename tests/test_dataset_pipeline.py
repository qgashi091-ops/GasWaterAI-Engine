"""Unit tests for the v0.4 dataset-pipeline modules (audit, privacy,
families, splits, candidates, qc, taxonomy, export). These are synthetic/
unit-level tests -- the real 20-plan batch is processed by
scripts/run_v04_pipeline.py and reported in docs/v04-dataset-report.md, not
re-run inside the test suite (each plan can take 1-4 minutes; that isn't a
per-commit test budget, and the real plans are not committed to the repo
at all -- see .gitignore).
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter

from app.dataset_pipeline import privacy
from app.dataset_pipeline.candidates import (
    AnnotationCandidate,
    _crop_contains_pii,
    _is_safe_plan_suggestion_label,
    _ocr_pii_zones,
    _rect_distance,
    _UnionFind,
)
from app.dataset_pipeline.export import write_class_list, write_manifest
from app.dataset_pipeline.families import group_families
from app.dataset_pipeline.qc import class_imbalance_report, find_overlapping_candidates
from app.dataset_pipeline.splits import design_split
from app.plan_analysis import schema
from app.dataset_pipeline.taxonomy import (
    FIXED_CLASSES,
    aggregate_evidence,
    is_plausible_component_label,
    normalize_legend_label,
    rank_classes,
)


def _candidate(cid, plan_id="DEV-01", page=1, bbox=(0, 0, 10, 10), plan_label=None, generic_label=None):
    return AnnotationCandidate(
        candidate_id=cid, plan_id=plan_id, page=page, bbox=bbox, crop_bbox=bbox,
        bbox_in_crop_px=(0, 0, 10, 10), crop_path="", nearby_text=[],
        graph_association={"relation": "on_edge", "node_ids": [], "edge_ids": []},
        plan_specific_suggestion={"label": plan_label, "score": 0.9} if plan_label else None,
        generic_suggestion={"label": generic_label, "score": 0.8} if generic_label else None,
        hybrid_kind="UNRESOLVED",
    )


# ---- privacy.py ----

def test_street_address_pattern_is_detected():
    audit = privacy.scan_text("DEV-X", "Der Bauherr wohnt an der Musterstrasse 12 in der Nähe.")
    kinds = {f.kind for f in audit.findings}
    assert "street_address" in kinds


def test_swiss_postal_code_and_town_is_detected():
    audit = privacy.scan_text("DEV-X", "Lieferadresse: 6003 Luzern")
    kinds = {f.kind for f in audit.findings}
    assert "postal_code_town" in kinds


def test_company_suffix_is_detected():
    audit = privacy.scan_text("DEV-X", "Ausführung durch die Muster Sanitär AG.")
    kinds = {f.kind for f in audit.findings}
    assert "company_name" in kinds


def test_plain_pipe_labels_raise_no_privacy_finding():
    audit = privacy.scan_text("DEV-X", "PE-S 63 | Bad / WC 03_101_09 | Kueche 03_101_07 | Cr18")
    assert audit.has_pii_risk is False


def test_lowercase_postal_code_and_town_is_still_detected():
    # Regression test: SWISS_PLZ_TOWN_PATTERN was missing re.IGNORECASE, so
    # a real postal-code+town pair silently passed every check once
    # legend_intelligence.py's own normalization lowercased it -- caught by
    # manual review of the real batch's taxonomy evidence, not by any test
    # that existed before this one.
    audit = privacy.scan_text("DEV-X", "6023 rothenburg")
    assert audit.has_pii_risk is True
    assert {f.kind for f in audit.findings} == {"postal_code_town"}


def test_lowercase_company_suffix_is_still_detected():
    audit = privacy.scan_text("DEV-X", "josef ottiger + partner ag beratung und planung")
    assert audit.has_pii_risk is True


def test_filename_with_street_address_is_flagged():
    flagged, reason = privacy.scan_filename("Villenstrasse-1-Luzern-Sanitaerschema.pdf")
    assert flagged is True


def test_generic_filename_is_not_flagged():
    flagged, _ = privacy.scan_filename("Sanitaerschema.pdf")
    assert flagged is False


# ---- candidates.py's mandatory PII safety net ----
# Regression test for a real finding: a candidate on one of this batch's
# real plans landed in a GAP between two legend_detection.py-flagged
# regions of an unusually large, fragmented title block, and its rendered
# crop showed a property owner's name, a company's address, phone number
# and email. legend_intelligence's own geometric exclusion (which
# candidates.py's earlier docstring wrongly assumed was sufficient on its
# own) did not catch that gap -- this content-based check must, regardless
# of whether any legend region happens to cover the crop.

def _span(text, bbox):
    return schema.TextSpan(id=f"t{hash((text, bbox))}", text=text, bbox=bbox, source="native")


def test_a_crop_overlapping_a_title_block_field_is_flagged_as_pii():
    spans_display = [
        (_span("Projektverfasser: Iwan Bühler Architekten, Geissensteinring 41, 6005 Luzern", (0.0, 0.0, 400.0, 10.0)),
         (0.0, 0.0, 400.0, 10.0)),
    ]
    crop_bbox = (50.0, 0.0, 150.0, 10.0)  # overlaps the title-block text span above
    assert _crop_contains_pii(crop_bbox, spans_display) is True


def test_a_crop_over_ordinary_pipe_labels_is_not_flagged():
    spans_display = [
        (_span("PE-S 63", (0.0, 0.0, 40.0, 10.0)), (0.0, 0.0, 40.0, 10.0)),
        (_span("Bad / WC 03_101_09", (50.0, 0.0, 120.0, 10.0)), (50.0, 0.0, 120.0, 10.0)),
    ]
    crop_bbox = (0.0, 0.0, 120.0, 10.0)
    assert _crop_contains_pii(crop_bbox, spans_display) is False


def test_a_crop_that_does_not_overlap_the_pii_text_is_not_flagged():
    # Same PII-bearing span as the positive case, but the crop is elsewhere
    # on the page -- overlap, not mere presence on the page, is what matters.
    spans_display = [
        (_span("Bauherr: Muster Immobilien AG, Bahnhofstrasse 5, 8001 Zürich", (0.0, 0.0, 400.0, 10.0)),
         (0.0, 0.0, 400.0, 10.0)),
    ]
    crop_bbox = (1000.0, 1000.0, 1100.0, 1010.0)
    assert _crop_contains_pii(crop_bbox, spans_display) is False


# ---- families.py ----

def test_same_project_prefix_groups_into_one_family():
    names = {
        "P1": "13926-S-SS-M-3_20240719.pdf", "P2": "13926-S-SS-M-5_20240719.pdf",
        "P3": "13926-S-SS-M-9_20240719.pdf", "P4": "Unrelated-Other-Project.pdf",
    }
    families = group_families(names)
    assert families["P1"] == families["P2"] == families["P3"]
    assert families["P4"] != families["P1"]


def test_short_generic_stem_does_not_falsely_merge():
    names = {"A": "Schema.pdf", "B": "Schema Wasser_Binggeli.pdf"}
    families = group_families(names)
    assert families["A"] != families["B"]


# ---- splits.py ----

def test_family_members_never_split_across_sets():
    family_of = {"A": "FAM-1", "B": "FAM-1", "C": "FAM-1", "D": "STANDALONE-1", "E": "STANDALONE-2"}
    split = design_split(family_of)
    assert split.plan_to_split["A"] == split.plan_to_split["B"] == split.plan_to_split["C"]


def test_split_covers_every_plan_exactly_once():
    family_of = {f"P{i}": f"STANDALONE-{i}" for i in range(10)}
    split = design_split(family_of)
    assert set(split.plan_to_split.keys()) == set(family_of.keys())
    assert sum(split.counts.values()) == 10


# ---- qc.py ----

def test_heavily_overlapping_candidates_on_same_page_are_flagged():
    a = _candidate("A", bbox=(0, 0, 10, 10))
    b = _candidate("B", bbox=(1, 1, 11, 11))  # large IoU with A
    c = _candidate("C", bbox=(500, 500, 510, 510))  # far away, no overlap
    pairs = find_overlapping_candidates([a, b, c])
    ids = {(p.candidate_id_a, p.candidate_id_b) for p in pairs}
    assert ("A", "B") in ids
    assert not any("C" in pair for pair in ids)


def test_overlap_check_is_scoped_per_plan_and_page():
    a = _candidate("A", plan_id="DEV-01", page=1, bbox=(0, 0, 10, 10))
    b = _candidate("B", plan_id="DEV-02", page=1, bbox=(0, 0, 10, 10))  # identical bbox, DIFFERENT plan
    pairs = find_overlapping_candidates([a, b])
    assert pairs == []


def test_class_imbalance_report_counts_suggested_labels():
    cands = [_candidate(str(i), plan_label="Ventil") for i in range(8)] + [_candidate("x", plan_label="Pumpe")]
    report = class_imbalance_report(cands)
    assert report["suggested_label_counts"]["Ventil"] == 8
    assert report["imbalance_warning"] is True


# ---- export.py ----

def test_write_manifest_and_class_list_round_trip():
    cands = [_candidate("A", plan_label="Ventil"), _candidate("B", generic_label="Pumpe")]
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        info = write_manifest(cands, {"DEV-01": "STANDALONE-1"}, {"DEV-01": "train"}, manifest_path)
        assert info["record_count"] == 2
        with open(manifest_path, encoding="utf-8") as f:
            records = json.load(f)
        assert records[0]["verification_status"] == "UNLABELED"
        assert records[0]["class"] is None
        assert records[0]["style_family"] == "STANDALONE-1"
        assert records[0]["split"] == "train"

        classes_path = os.path.join(tmp, "classes.json")
        mapping = write_class_list(["Ventil", "Pumpe"] + FIXED_CLASSES, classes_path)
        assert mapping["Ventil"] == 0
        assert "NOT_A_COMPONENT" in mapping


# ---- candidates.py's THIRD safety net: OCR-based PII zones ----
# Regression test for a second, worse real finding on the same plan as
# above: after the text-layer check (_crop_contains_pii) shipped, the exact
# same offending candidate was STILL present, because v0.1's own native
# text extraction returns NOTHING for that part of the page -- confirmed
# directly with PyMuPDF's own get_text("words")/get_text("dict") -- since
# those specific title-block fields are drawn as vector outlines/curves
# ("convert text to paths", a common CAD export setting), not real PDF text
# objects. No text-layer method, however it's called, can ever see this.
# _ocr_pii_zones renders and OCRs the actual pixels instead, which is the
# only way to catch PII that was never text in the PDF to begin with.

def test_rect_distance_is_zero_when_overlapping_and_positive_when_apart():
    assert _rect_distance((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == 0.0
    assert _rect_distance((0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)) == 10.0


def test_union_find_groups_only_directly_or_transitively_linked_items():
    uf = _UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)


def test_ocr_pii_zone_catches_pii_rendered_as_a_raster_image_not_pdf_text(tmp_path):
    # Replicates the real failure exactly: text baked into an inserted
    # image has NO PDF text object at all (get_text() returns ''), the same
    # as vector-outlined title-block text -- both are invisible to every
    # text-layer method, and only pixel-level OCR can catch either.
    import pymupdf
    from PIL import Image, ImageDraw

    img_path = os.path.join(str(tmp_path), "fake_titleblock.png")
    img = Image.new("RGB", (300, 80), "white")
    ImageDraw.Draw(img).text((5, 5), "Bauherr: Acme AG, Bahnhofstrasse 5, 8001 Zuerich", fill="black")
    img.save(img_path)

    pdf_path = os.path.join(str(tmp_path), "fake_plan.pdf")
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    image_rect = pymupdf.Rect(10, 10, 310, 90)
    page.insert_image(image_rect, filename=img_path)
    doc.save(pdf_path)
    doc.close()

    reopened = pymupdf.open(pdf_path)
    page = reopened[0]
    assert page.get_text() == ""  # confirms this really is invisible to the text layer

    zones = _ocr_pii_zones(page, [(10.0, 10.0, 310.0, 90.0)])
    assert len(zones) == 1
    assert _bbox_overlaps_for_test(zones[0], (10.0, 10.0, 310.0, 90.0))


def test_ocr_timeout_fails_safe_as_pii_rather_than_hanging(monkeypatch):
    # Regression test for a real finding: the same multi-minute tesseract
    # pathology already known from the offline batch pipeline (worked
    # around there with a process-group wall-clock timeout) also hit a
    # single OCR zone live, through the annotation tool's wide-context
    # endpoint, with no timeout protection at all -- observed hanging one
    # real request for 6+ minutes with no recovery. pytesseract's own
    # `timeout` kwarg now bounds every OCR call; on expiry it raises
    # RuntimeError, which must be caught by the existing fail-safe except
    # clause (treat as PII) rather than propagating and hanging the caller.
    import pymupdf

    def _raise_timeout(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("app.dataset_pipeline.candidates.pytesseract.image_to_string", _raise_timeout)
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    zones = _ocr_pii_zones(page, [(10.0, 10.0, 190.0, 190.0)])
    assert len(zones) == 1


def test_ocr_pii_zone_finds_nothing_when_the_region_has_no_legend_candidates():
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    assert _ocr_pii_zones(page, []) == []


def _bbox_overlaps_for_test(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ---- candidates.py's THIRD, DIFFERENT PII vector: plan_specific_suggestion ----
# Regression test for a real finding distinct from the crop-image leaks
# above: rounds 1-2 guard the rendered crop, but v0.3's own best-matching-
# legend-entry suggestion (shown as a one-click button in the annotation
# tool and stored per-candidate in manifest.json) can independently be a
# company's phone number or address if legend_detection.py's clustering
# swept a contact line into legend_entries and v0.3's matcher picked it as
# a symbol's best match. Found by auditing this batch's real qc.py
# class-imbalance report, which showed a phone number as the single most
# common "suggested label" across all 20 plans' candidates.

def test_a_phone_number_plan_suggestion_is_rejected():
    assert _is_safe_plan_suggestion_label("t 041 818 09 09") is False


def test_a_company_name_and_phone_plan_suggestion_is_rejected():
    assert _is_safe_plan_suggestion_label("planinno gmbh tel.: 041 521 30 00") is False


def test_a_genuine_device_plan_suggestion_is_accepted():
    assert _is_safe_plan_suggestion_label("Rückschlagklappe") is True


# ---- taxonomy.py's own mandatory PII filter + noise filter ----
# Regression test for a real finding on this exact batch: a plan's LEGEND
# region (not its title block -- candidates.py's own three-layer net only
# guards crop images, not this module's raw text) contained a supplier's
# contact line -- a street address, a phone number, a company name --
# printed directly under a materials list, and that raw text would
# otherwise have flowed straight into classes.json/pipeline_summary.json,
# both committed artifacts. Caught by manual review of the real batch's
# evidence_counts, not by any automated check that existed before this fix.

def test_a_legend_label_with_a_phone_number_is_rejected():
    assert is_plausible_component_label("Tel. 041 348 00 60") is False


def test_a_legend_label_with_a_street_address_is_rejected():
    assert is_plausible_component_label("Ettiswilerstrasse 39") is False


def test_a_legend_label_with_a_postal_code_and_town_is_rejected():
    assert is_plausible_component_label("6023 Rothenburg") is False


def test_a_genuine_device_name_is_accepted():
    assert is_plausible_component_label("Rückschlagklappe") is True
    assert is_plausible_component_label("Sicherheitsgruppe") is True
    assert is_plausible_component_label("Wasserzähler") is True


def test_a_diameter_or_nominal_width_spec_is_rejected_as_noise():
    assert is_plausible_component_label("ø63") is False
    assert is_plausible_component_label("DN100") is False
    assert is_plausible_component_label("vpe ø25(4)") is False


def test_an_abbreviation_colon_number_fixture_table_cell_is_rejected_as_noise():
    assert is_plausible_component_label("lu: 1") is False
    assert is_plausible_component_label("du: 0.8") is False
    assert is_plausible_component_label("kw: 3") is False


def test_a_riser_label_is_rejected_as_noise():
    assert is_plausible_component_label("strang -13") is False


def test_a_mostly_digit_fragment_is_rejected_as_noise():
    assert is_plausible_component_label("1 1 1 1/2\"") is False
    assert is_plausible_component_label("16") is False


def test_a_too_short_fragment_is_rejected_as_noise():
    assert is_plausible_component_label("m") is False
    assert is_plausible_component_label(".") is False


def test_normalize_legend_label_strips_a_trailing_instance_number():
    assert normalize_legend_label("pex-verteiler 45") == "pex-verteiler"
    assert normalize_legend_label("pex-verteiler") == "pex-verteiler"


def test_aggregate_and_rank_respects_min_support_and_fixed_classes_last():
    evidence = aggregate_evidence(
        legend_counts=Counter({"Rückschlagklappe": 3, "Einzelfund": 1}),
        safety_code_counts=Counter({"EA": 2}),
        generic_counts=Counter(),
    )
    assert "Einzelfund" not in evidence  # below MIN_SUPPORT=2
    classes = rank_classes(evidence)
    assert classes[:2] == ["Rückschlagklappe", "EA"] or classes[:2] == ["EA", "Rückschlagklappe"]
    assert classes[-3:] == FIXED_CLASSES
