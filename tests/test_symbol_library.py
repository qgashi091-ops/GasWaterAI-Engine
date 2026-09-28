"""Unit tests for classical rotation/scale-invariant template matching
(symbol_library.py) -- operate directly on the 58 reference images, no PDF
rendering needed. Covers the "exact / rotated / scaled / visually similar
but different" cases required for v0.2.
"""
from __future__ import annotations

import cv2

from app.plan_analysis.symbol_library import canonicalize, get_library

LIB = get_library()


def _load_gray(symbol_id: str):
    import os
    from app.plan_analysis.symbol_library import LIBRARY_DIR
    path = os.path.join(LIBRARY_DIR, "templates", f"{symbol_id}.png")
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def test_exact_symbol_is_recognized_with_near_perfect_score():
    gray = _load_gray("SYM-038")  # Wasserzaehler -- distinctive, not in any ambiguous family
    canon = canonicalize(gray)
    scores = LIB.match(canon)
    assert scores[0].symbol_id == "SYM-038"
    assert scores[0].score > 0.99


def test_rotated_symbol_is_still_recognized():
    gray = _load_gray("SYM-038")
    for cv_rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(gray, cv_rotation)
        canon = canonicalize(rotated)
        scores = LIB.match(canon)
        assert scores[0].symbol_id == "SYM-038", f"failed at cv_rotation={cv_rotation}"
        assert scores[0].score > 0.95, f"failed at cv_rotation={cv_rotation}: {scores[0].score}"


def test_scaled_symbol_is_still_recognized():
    gray = _load_gray("SYM-038")
    h, w = gray.shape
    for scale in (0.6, 1.6):
        resized = cv2.resize(gray, (int(w * scale), int(h * scale)))
        canon = canonicalize(resized)
        scores = LIB.match(canon)
        assert scores[0].symbol_id == "SYM-038", f"failed at scale={scale}"
        assert scores[0].score > 0.85, f"failed at scale={scale}: {scores[0].score}"


def test_visually_similar_but_different_symbol_shows_up_as_a_genuine_ambiguous_family():
    # SYM-008/013/014 (Absperrarmatur/Schieber/Absperrklappe) were flagged by
    # the data audit as a near-duplicate family -- an exact copy of one must
    # still show its family-mates scoring almost as high, proving the
    # ambiguity is real at match time, not just a template-time artifact.
    gray = _load_gray("SYM-008")
    canon = canonicalize(gray)
    scores = LIB.match(canon)
    top3_ids = {s.symbol_id for s in scores[:3]}
    family = LIB.ambiguous_family_by_symbol.get(scores[0].symbol_id)
    assert family is not None
    assert len(top3_ids & set(family)) >= 2, "an ambiguous family's members must show up close together, not falsely separated"


def test_excluded_symbols_never_appear_as_a_match_hypothesis():
    # SYM-001..005 (not_a_component) and the low-distinctiveness set must
    # never be offered as an answer, regardless of what's being matched.
    gray = _load_gray("SYM-038")
    canon = canonicalize(gray)
    scores = LIB.match(canon)
    returned_ids = {s.symbol_id for s in scores}
    assert not (returned_ids & LIB.not_a_component)
    assert not (returned_ids & LIB.low_distinctiveness)


def test_a_near_blank_crop_scores_essentially_nothing_against_every_template():
    import numpy as np
    blank = np.full((128, 128), 255, dtype="uint8")
    canon = canonicalize(blank)
    assert canon is None  # no ink at all -- caller must treat this as UNRESOLVED, not call match()
