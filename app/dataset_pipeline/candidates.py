"""v0.4 Phase 4 -- automatic annotation-candidate generation.

Turns every hybrid component fact `legend_intelligence.py` already computed
(v0.1 symbol candidate + v0.2 generic match + v0.3 plan-specific match +
graph association -- all reused, nothing re-detected here) into a
human-reviewable annotation candidate: a stable id, a rendered crop with
enough surrounding context to identify the component, the nearby text v0.1
already associated with it, its graph relation, and the top predicted
labels from both recognition paths.

TITLE-BLOCK / PII EXCLUSION -- NOT free, found the hard way: the first
version of this module assumed legend_intelligence.py's own Phase-4
self-match exclusion (it only evaluates v0.1 symbol candidates OUTSIDE
every detected legend/dense-text region) would also keep every candidate
out of a plan's title block, since a title block is exactly such a
dense-text region on W-003. That assumption is FALSE in general: on one of
this batch's real plans, legend_detection.py's clustering fragmented an
unusually large, complex title block (company names, addresses, phone
numbers, emails across a dozen contact-role rows) into several separate
dense-text candidates with real gaps between them, and a v0.1 symbol
candidate (a stray vector primitive near a "Gezeichnet: ..." field) landed
in exactly such a gap -- fully inside the title block visually, but outside
every individual detected legend bbox. That crop, rendered and inspected
during development, showed a property owner's name, a company's full
address, phone number and email in plain text. Never shipped: caught by
manual visual review before being added to any committed dataset.

Fix, round 1: a SECOND, independent, content-based safety net --
`_crop_contains_pii()` -- runs privacy.py's own regex scan (street address /
postal-code+town / title-block-field / company-suffix / phone / email)
against every text span that falls inside a candidate's OWN rendered crop
region, not just the region legend_detection.py happened to flag. Any hit
drops that candidate ENTIRELY (no crop is rendered or saved) rather than
just warning -- geometric exclusion is convenient when it works, but this
privacy guarantee does not depend on getting the clustering geometry right.

Fix, round 2 -- a WORSE gap than round 1 fixed: re-checking that exact
candidate after round 1 shipped showed it was STILL not excluded, and still
showed the same PII. Root cause: v0.1's own native text extraction
(`text.py::extract_native_text_spans`, via PyMuPDF's `get_text()`) finds
NOTHING at all in that part of the page -- not a garbled span, an outright
ABSENT one, confirmed with `get_text("words")` and `get_text("dict")`
directly. Some of this plan's title-block fields are drawn as vector
outlines/curves (a common CAD-export setting, "convert text to paths"),
which render as perfectly readable letters but are not text objects in the
PDF at all -- invisible to every text-layer extraction method there is, no
matter how it's called. Layer 2 cannot see what isn't text.

Fix, round 2's actual fix -- `_ocr_pii_zones()`: OCRs a handful of
EXPANDED regions once per PAGE (not once per candidate, which would be
prohibitively slow across hundreds of candidates): every detected
legend/dense-text region (from legend_intelligence.py, already computed) is
grouped with its neighbors within OCR_ZONE_MERGE_DISTANCE_PT and padded by
OCR_ZONE_PADDING_PT, since a real title block's fragments (as in round 1's
finding) can sit farther apart than legend_detection.py's own, deliberately
tight clustering distance. Each resulting zone is OCR'd (small, a few
seconds each, not the multi-minute full-page-tile pathology found earlier
in this same batch's OCR fallback path) and scanned for PII; any zone that
matches becomes a hard exclusion region for every candidate on that page,
independent of what round 1 saw in the (in this case, absent) text layer.

Fix, round 3 -- a DIFFERENT vector than rounds 1-2, found while auditing this
batch's proposed taxonomy: rounds 1-2 guard the rendered CROP IMAGE, but
each candidate also carries a `plan_specific_suggestion` -- v0.3's own
best-matching LEGEND ENTRY label for that symbol, `fact["plan_specific_
evidence"]["label"]`, shown to the human reviewer in the annotation tool as
a one-click suggestion button and stored verbatim in the committed
`manifest.json`. `legend_detection.py`'s clustering can sweep a company's
contact line ("Tel. 041 521 30 00", a street address, an email) into
`legend_entries` alongside genuine device rows (the same root cause as the
crop-image gap above, seen from a different angle), and v0.3's matcher will
happily match a nearby symbol against THAT entry if it scores best -- with
nothing about that matching logic knowing the "legend entry" it just picked
is actually a contact line, not a device label. On this batch, a phone
number and a company name+phone ended up as the suggested label on dozens
of real candidates before this fix. `plan_specific_suggestion` is now
dropped (not the whole candidate -- the crop itself already passed rounds
1-2 independently) whenever its label fails the same `privacy.scan_text`
check; the candidate keeps its `generic_suggestion` (always a fixed
library name, never scraped plan text, so never at risk here) and remains
fully annotatable with no suggestion pre-filled.

OCR ROBUSTNESS -- a separate, non-privacy bug found the same way (running
the annotation tool live against this batch's real, final manifest): the
same multi-minute tesseract pathology already known from the offline batch
pipeline's OCR-fallback path (worked around there with a process-group
wall-clock timeout, see `scripts/run_v04_pipeline.py`) also hit a single
zone inside `_ocr_scan_rect_for_pii`/`_ocr_pii_zones`, with NO timeout
protection at this level at all -- observed hanging one real request to the
annotation tool's `/wide_crop` endpoint for 6+ minutes with no recovery,
and the same call also runs once per page during candidate generation
itself. Fixed with pytesseract's own `timeout=` kwarg (`OCR_TIMEOUT_SECONDS`
-- on expiry it raises `RuntimeError` and kills its own tesseract
subprocess, cleanly, no orphaned process), which the existing fail-SAFE
`except Exception` clause already treats as PII, exactly like any other OCR
failure -- no new exception-handling path needed.

COORDINATE FRAMES (the same rotation issue v0.2/v0.3 already found and
worked around -- see component_facts.py's `_render_symbol_crop` docstring):
v0.1's own `SymbolCandidate.bbox` is in the PDF's UNROTATED ("native")
frame, but `page.get_pixmap()` always renders in the ROTATED/DISPLAY frame.
Over half of this batch's 20 real plans are rotated 90/270 degrees, so this
is not a corner case here -- every bbox this module hands to a human
reviewer or a future detector (`bbox`, `crop_bbox`, `bbox_in_crop_px`) is
computed in DISPLAY space (via `page.rotation_matrix`, identity when
rotation is 0) so it lines up with the actual rendered crop pixels; only
the render call itself still takes the native-frame bbox, matching
`_render_symbol_crop`'s existing, unchanged contract.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Optional

import cv2
import numpy as np
import pymupdf
import pytesseract

from app.dataset_pipeline.privacy import scan_text
from app.plan_analysis import schema
from app.plan_analysis.component_facts import _display_clip_rect, _render_symbol_crop

CONTEXT_MARGIN_FRACTION = 3.0  # multiple of the candidate's own max(width, height)
CONTEXT_MARGIN_MIN_PT = 60.0
CONTEXT_MARGIN_MAX_PT = 400.0
# Round-2 OCR safety net (see module docstring): two dense-text regions
# within this distance of each other are treated as fragments of the same
# title block for OCR-zone purposes -- deliberately looser than
# legend_detection.py's own COLUMN_MERGE_MAX_X_GAP (100pt), since that
# real title block's fragments sat ~470pt apart, farther than
# legend_detection's own (correctly conservative, for its own clustering
# purpose) merge distance.
OCR_ZONE_MERGE_DISTANCE_PT = 600.0
OCR_ZONE_PADDING_PT = 120.0
OCR_ZONE_DPI = 150  # low enough to OCR a zone in a few seconds, high enough for reliable text
# A single zone normally OCRs in a few seconds (see module docstring) -- but
# the SAME multi-minute tesseract pathology already found and worked around
# at the offline batch level (run_v04_pipeline.py's process-group timeout,
# for a scanned page's OCR fallback) can also hit a single zone here, with
# no batch-level safety net to catch it: found live, hanging the annotation
# tool's wide-context endpoint (and, inside _ocr_pii_zones, one page's whole
# candidate-generation pass) for 6+ minutes on one real zone. pytesseract's
# own `timeout` kwarg bounds this at the OCR-call level -- on expiry it
# raises, which the existing fail-SAFE except-clause below already treats
# as PII, exactly like any other OCR failure.
OCR_TIMEOUT_SECONDS = 20
# Hard cap on the rendered crop's own width/height, centered on the
# candidate's centroid. Needed because v0.1's own symbol-candidate
# clustering (symbols.py, unchanged -- a pre-existing, documented
# limitation, see docs/architecture.md's "Known v0.2 limitations") can
# occasionally merge a large run of nearby annotations into one oversized
# candidate spanning several building storeys; without this cap, a
# proportional margin around such a candidate produces a crop that shows
# nearly the whole drawing -- the opposite of "avoid unnecessarily large
# crops." Capping trades completeness for usability for that rare case: the
# crop is centered on the candidate so its most identifiable part is still
# visible, even if the full oversized extent is not.
MAX_CROP_SIDE_PT = 450.0
# Crops are rendered via component_facts._render_symbol_crop, which always
# renders at RENDER_DPI=300 (its own fixed constant) -- reused here (not
# duplicated as a separate DPI setting) so bbox_in_crop_px always matches
# the actual saved PNG's pixel grid.
_RENDER_DPI_SCALE = 300.0 / 72.0


def _stable_candidate_id(plan_id: str, page: int, bbox: tuple) -> str:
    key = f"{plan_id}|{page}|{','.join(f'{v:.2f}' for v in bbox)}"
    return "AC" + sha256(key.encode("utf-8")).hexdigest()[:20]


@dataclass
class AnnotationCandidate:
    candidate_id: str
    plan_id: str
    page: int
    bbox: tuple  # DISPLAY-space bbox (the actual candidate, tight)
    crop_bbox: tuple  # DISPLAY-space bbox actually rendered (candidate + context margin)
    bbox_in_crop_px: tuple  # the candidate's own bbox, in pixels WITHIN the saved crop image
    crop_path: str  # relative path to the saved PNG crop
    nearby_text: list  # list[str] -- text v0.1 associated with this exact symbol
    graph_association: dict
    plan_specific_suggestion: Optional[dict]  # {label, score, legend_entry_id} or None
    generic_suggestion: Optional[dict]  # {label, score} or None
    hybrid_kind: str  # the v0.2/v0.3 engine's own kind for this candidate, for reference only
    status: str = "UNLABELED"
    assigned_class: Optional[str] = None
    corrected_bbox: Optional[tuple] = None
    annotator_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "plan_id": self.plan_id,
            "page": self.page,
            "bbox": list(self.bbox),
            "crop_bbox": list(self.crop_bbox),
            "bbox_in_crop_px": list(self.bbox_in_crop_px),
            "crop_path": self.crop_path,
            "nearby_text": self.nearby_text,
            "graph_association": self.graph_association,
            "plan_specific_suggestion": self.plan_specific_suggestion,
            "generic_suggestion": self.generic_suggestion,
            "hybrid_kind": self.hybrid_kind,
            "status": self.status,
            "assigned_class": self.assigned_class,
            "corrected_bbox": list(self.corrected_bbox) if self.corrected_bbox else None,
            "annotator_note": self.annotator_note,
        }


def _to_display(bbox: tuple, rotation_matrix) -> tuple:
    r = pymupdf.Rect(*bbox) * rotation_matrix
    return (r.x0, r.y0, r.x1, r.y1)


def _native_bounds(page: "pymupdf.Page") -> tuple:
    """The v0.1 native (unrotated) frame's bounds -- page.mediabox width/height
    swapped relative to page.rect when the page is rotated 90/270 degrees."""
    if page.rotation in (90, 270):
        return (0.0, 0.0, page.rect.height, page.rect.width)
    return (0.0, 0.0, page.rect.width, page.rect.height)


def _context_bbox_display(bbox_display: tuple, page_rect_display: "pymupdf.Rect") -> tuple:
    x0, y0, x1, y1 = bbox_display
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    margin = min(max(CONTEXT_MARGIN_FRACTION * max(w, h), CONTEXT_MARGIN_MIN_PT), CONTEXT_MARGIN_MAX_PT)
    half_side = min(max(w, h) / 2.0 + margin, MAX_CROP_SIDE_PT / 2.0)
    clip = pymupdf.Rect(cx - half_side, cy - half_side, cx + half_side, cy + half_side) & page_rect_display
    return (clip.x0, clip.y0, clip.x1, clip.y1)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _rect_distance(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _ocr_scan_rect_for_pii(page: "pymupdf.Page", clip: "pymupdf.Rect") -> bool:
    """Renders exactly `clip` (DISPLAY space) at OCR_ZONE_DPI and scans the
    OCR'd text for PII -- the actual mechanism behind `_ocr_pii_zones`,
    pulled out so a caller that already knows the one region it cares about
    (the annotation tool's live wide-context endpoint, which has no
    precomputed legend boxes to cluster) can reuse it directly on that
    region instead of on a legend-derived zone list. Fails SAFE (treats a
    render/OCR exception as PII) rather than silently passing an unreadable
    region through -- this is a safety net, not the main pipeline."""
    if clip.is_empty:
        return False
    zoom = OCR_ZONE_DPI / 72.0
    try:
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        gray = arr[:, :, 0]
        text = pytesseract.image_to_string(gray, lang="deu+eng", config="--psm 6", timeout=OCR_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 -- see docstring: OCR failure must fail SAFE, not silently pass through
        return True
    return scan_text("_ocr_zone_check", text).has_pii_risk


def _ocr_pii_zones(page: "pymupdf.Page", legend_bboxes_display: list) -> list:
    """Round-2 OCR safety net -- see module docstring. Groups the page's own
    already-detected legend/dense-text regions (any confidence, any
    heading -- legend_intelligence.py already computed these, nothing new
    is detected here) into clusters by proximity, pads each cluster's
    envelope, and OCRs it directly via `_ocr_scan_rect_for_pii` (bypassing
    the PDF text layer entirely, since that layer can be flatly missing the
    very content being checked for). Returns the padded envelope of every
    cluster whose OCR'd text matches privacy.py's patterns -- a hard
    exclusion zone list for the caller, independent of anything round 1
    (`_crop_contains_pii`) saw."""
    if not legend_bboxes_display:
        return []

    uf = _UnionFind(len(legend_bboxes_display))
    for i in range(len(legend_bboxes_display)):
        for j in range(i + 1, len(legend_bboxes_display)):
            if _rect_distance(legend_bboxes_display[i], legend_bboxes_display[j]) <= OCR_ZONE_MERGE_DISTANCE_PT:
                uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(legend_bboxes_display)):
        clusters.setdefault(uf.find(i), []).append(i)

    page_rect = page.rect
    pii_zones = []
    for members in clusters.values():
        boxes = [legend_bboxes_display[i] for i in members]
        x0 = min(b[0] for b in boxes) - OCR_ZONE_PADDING_PT
        y0 = min(b[1] for b in boxes) - OCR_ZONE_PADDING_PT
        x1 = max(b[2] for b in boxes) + OCR_ZONE_PADDING_PT
        y1 = max(b[3] for b in boxes) + OCR_ZONE_PADDING_PT
        clip = pymupdf.Rect(x0, y0, x1, y1) & page_rect
        if _ocr_scan_rect_for_pii(page, clip):
            pii_zones.append((clip.x0, clip.y0, clip.x1, clip.y1))
    return pii_zones


def _nearby_text(sym: schema.SymbolCandidate, associations: list, text_spans_by_id: dict) -> list:
    """v0.1's association.py search radius is normally well inside the crop's
    own context margin, so `_crop_contains_pii`'s check over the crop region
    already covers this text too -- but that's a fact about typical radii,
    not a guarantee, so each associated text string is ALSO scanned
    individually here before being placed in a committed manifest. A single
    flagged string is dropped from the list, not the whole candidate (unlike
    the crop-region check): the candidate's shape evidence and other,
    unflagged nearby text remain legitimate, usable annotation context."""
    out = []
    for assoc in associations:
        if assoc.target_type == "symbol" and assoc.target_id == sym.id:
            span = text_spans_by_id.get(assoc.text_id)
            text = span.text.strip() if span else ""
            if text and not scan_text("_nearby_text_check", text).has_pii_risk:
                out.append(text)
    return out


def _bbox_overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _is_safe_plan_suggestion_label(label: str) -> bool:
    """Round 3's own check (see module docstring) -- kept as its own tiny,
    directly-testable function rather than inlined, exactly like every
    other privacy check in this module."""
    return not scan_text("_plan_suggestion_check", label).has_pii_risk


def _crop_contains_pii(crop_bbox_display: tuple, all_spans_display: list) -> bool:
    """The mandatory second safety net described in this module's docstring:
    checks EVERY text span whose bbox overlaps the crop region actually
    being rendered (not just text v0.1 happened to associate with the
    symbol) against privacy.py's own regex patterns. Independent of
    legend_detection.py's clustering, so a title block that clustering
    fragmented or missed is still caught here directly from what the crop
    would actually show."""
    texts = [s.text for s, dbbox in all_spans_display if _bbox_overlaps(dbbox, crop_bbox_display)]
    if not texts:
        return False
    return scan_text("_crop_check", " ".join(texts)).has_pii_risk


def build_candidates_for_plan(
    plan_id: str, pdf_bytes: bytes, doc: schema.DocumentAnalysis,
    legend_intelligence_result: dict, crop_output_dir: str,
) -> tuple[list[AnnotationCandidate], int]:
    """Returns (candidates, pii_excluded_count) -- the count is reported by
    the pipeline's privacy section so a candidate dropped for containing
    PII-shaped text is visible in the audit trail, not silently vanished."""
    os.makedirs(crop_output_dir, exist_ok=True)
    facts_by_key = {}
    for f in legend_intelligence_result["component_facts"]:
        key = (f["page"], tuple(round(v, 1) for v in f["bbox"]))
        facts_by_key[key] = f

    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    candidates: list[AnnotationCandidate] = []
    pii_excluded_count = 0
    try:
        for page_model in doc.pages:
            page = pdf[page_model.page_number - 1]
            text_spans_by_id = {s.id: s for s in page_model.text_spans}
            rotation_matrix = page.rotation_matrix
            page_rect_display = page.rect
            all_spans_display = [(s, _to_display(s.bbox, rotation_matrix))
                                  for s in page_model.text_spans if s.text.strip()]
            legend_bboxes_display = [tuple(c["bbox_display_space"])
                                      for c in legend_intelligence_result["legend_candidates"]
                                      if c["page"] == page_model.page_number]
            ocr_pii_zones = _ocr_pii_zones(page, legend_bboxes_display)

            for sym in page_model.symbols:
                key = (page_model.page_number, tuple(round(v, 1) for v in sym.bbox))
                fact = facts_by_key.get(key)
                if fact is None:
                    continue  # excluded by Phase 4 (inside a legend region) -- never a candidate here either

                cand_id = _stable_candidate_id(plan_id, page_model.page_number, sym.bbox)

                bbox_display = _to_display(sym.bbox, rotation_matrix)
                crop_bbox_display = _context_bbox_display(bbox_display, page_rect_display)

                if _crop_contains_pii(crop_bbox_display, all_spans_display) or \
                        any(_bbox_overlaps(crop_bbox_display, z) for z in ocr_pii_zones):
                    pii_excluded_count += 1
                    continue  # mandatory safety net -- see module docstring; never rendered, never saved

                # _render_symbol_crop's own contract takes a NATIVE-frame bbox
                # and applies rotation_matrix itself -- recover the native
                # frame's crop rect by inverting the display transform.
                crop_bbox_native_rect = pymupdf.Rect(*crop_bbox_display) * ~rotation_matrix
                crop_bbox_native = (crop_bbox_native_rect.x0, crop_bbox_native_rect.y0,
                                     crop_bbox_native_rect.x1, crop_bbox_native_rect.y1)
                crop_gray = _render_symbol_crop(page, crop_bbox_native)

                crop_filename = f"{cand_id}.png"
                crop_path = os.path.join(crop_output_dir, crop_filename)
                if crop_gray is not None:
                    cv2.imwrite(crop_path, crop_gray)
                else:
                    crop_path = ""

                # _render_symbol_crop adds its OWN extra margin on top of
                # whatever bbox it's given (component_facts.CROP_MARGIN_FRACTION)
                # -- the saved image's actual origin is NOT crop_bbox_display's
                # own corner. _display_clip_rect computes the exact same clip
                # rect _render_symbol_crop renders, so bbox_in_crop_px lines up
                # with the real saved pixels instead of being offset by that
                # extra margin.
                actual_clip = _display_clip_rect(page, crop_bbox_native)
                bx0, by0, bx1, by1 = bbox_display
                bbox_in_crop_px = (
                    round((bx0 - actual_clip.x0) * _RENDER_DPI_SCALE, 1), round((by0 - actual_clip.y0) * _RENDER_DPI_SCALE, 1),
                    round((bx1 - actual_clip.x0) * _RENDER_DPI_SCALE, 1), round((by1 - actual_clip.y0) * _RENDER_DPI_SCALE, 1),
                )

                plan_ev = fact.get("plan_specific_evidence")
                plan_suggestion = None
                if plan_ev and plan_ev.get("label") and _is_safe_plan_suggestion_label(plan_ev["label"]):
                    plan_suggestion = {"label": plan_ev["label"], "score": plan_ev["best_score"],
                                        "legend_entry_id": plan_ev.get("legend_entry_id")}
                generic_ev = fact.get("generic_evidence")
                generic_suggestion = None
                if generic_ev and generic_ev.get("symbol_name"):
                    generic_suggestion = {"label": generic_ev["symbol_name"], "score": generic_ev.get("confidence")}

                candidates.append(AnnotationCandidate(
                    candidate_id=cand_id, plan_id=plan_id, page=page_model.page_number,
                    bbox=bbox_display, crop_bbox=crop_bbox_display, bbox_in_crop_px=bbox_in_crop_px,
                    crop_path=crop_path,
                    nearby_text=_nearby_text(sym, page_model.associations, text_spans_by_id),
                    graph_association=fact["graph_association"],
                    plan_specific_suggestion=plan_suggestion,
                    generic_suggestion=generic_suggestion,
                    hybrid_kind=fact["kind"],
                ))
    finally:
        pdf.close()
    return candidates, pii_excluded_count
