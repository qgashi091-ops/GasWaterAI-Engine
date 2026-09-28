"""Loads the 58 SVGW symbol reference images + metadata + the data-audit
result (docs/symbol-audit.md, app/data/symbol_library/audit.json) and
exposes rotation-invariant classical template matching against a candidate
crop. New in v0.2 -- no equivalent in v0.1 or in the original GasWaterAI
parser repo (which explicitly defers all symbol *naming* to Base44's
library, see symbols.py's own module docstring).

WHY RASTER TEMPLATE MATCHING, NOT VECTOR-TO-VECTOR: the 58 reference images
are themselves scanned/rasterized legend graphics embedded in the source
PDF (see docs/symbol-audit.md) -- there is no vector path data to compare
against on the reference side, regardless of how the target plan itself was
authored. Both sides are therefore canonicalized to a binary raster mask
before comparison.

CANONICALIZATION (scale-invariance): every image (template or candidate) is
thresholded to an ink mask, cropped to its own ink bounding box, and resized
(aspect-preserving, padded) onto a fixed square canvas. This removes
absolute size/DPI differences by construction -- a "scaled symbol" test
passes for free, not through any special-cased logic.

ROTATION-INVARIANCE: each template is pre-rotated at the four cardinal
angles (0/90/180/270 degrees) -- covering the vast majority of real
orientations in an orthogonal pipe schematic -- and a candidate is scored
against all four, keeping the best. Arbitrary-angle rotation and mirroring
were deliberately NOT added: more transform hypotheses raise the chance of
an incorrect high-scoring match, which directly fights the "precision
before recall" requirement for this version.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "symbol_library")
CANVAS = 128
ROTATIONS = (0, 90, 180, 270)


def canonicalize(gray_or_mask: np.ndarray, already_binary: bool = False) -> Optional[np.ndarray]:
    if gray_or_mask is None or gray_or_mask.size == 0:
        return None
    if already_binary:
        mask = gray_or_mask
    else:
        _, mask = cv2.threshold(gray_or_mask, 200, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape
    scale = (CANVAS - 8) / max(h, w, 1)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    oy, ox = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


@dataclass
class SymbolTemplate:
    symbol_id: str
    name: str
    beschreibung: str
    quelle: str
    rotations: dict = field(default_factory=dict)  # {angle: canonical mask}


@dataclass
class MatchScore:
    symbol_id: str
    score: float
    rotation: int


class SymbolLibrary:
    def __init__(self, library_dir: str = LIBRARY_DIR):
        meta = json.load(open(os.path.join(library_dir, "symbole58.json"), encoding="utf-8"))
        audit = json.load(open(os.path.join(library_dir, "audit.json"), encoding="utf-8"))
        self.meta_by_id = {m["id"]: m for m in meta}

        # Data-audit results (docs/symbol-audit.md) -- computed once, offline,
        # over the 58 templates themselves (never over W-001..W-010, which
        # stay holdout). Drives which symbols even enter the matching pool
        # and which matches must be capped below COMPONENT_FACT.
        self.not_a_component = set(audit["not_a_component"])
        self.low_distinctiveness = set(audit["low_distinctiveness"])
        self.text_legend_required = set(audit["text_legend_required"])
        self.ambiguous_family_by_symbol: dict[str, tuple] = {}
        for fam in audit["ambiguous_families"]:
            fam_t = tuple(fam)
            for sid in fam:
                self.ambiguous_family_by_symbol[sid] = fam_t

        self.templates: dict[str, SymbolTemplate] = {}
        for sid, m in self.meta_by_id.items():
            path = os.path.join(library_dir, "templates", f"{sid}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            canon = canonicalize(img)
            if canon is None:
                continue
            rotations = {0: canon}
            cur = canon
            for angle in (90, 180, 270):
                cur = cv2.rotate(cur, cv2.ROTATE_90_CLOCKWISE)
                rotations[angle] = cur
            self.templates[sid] = SymbolTemplate(sid, m["name"], m["beschreibung"], m["quelle"], rotations)

        # The actual hypothesis pool for matching -- excludes symbols the
        # data audit found are not physical components at all, or are so
        # visually weak (bare line/arrow/dot) they would mostly match plan
        # clutter rather than their intended meaning.
        self.eligible_ids = [
            sid for sid in self.templates
            if sid not in self.not_a_component and sid not in self.low_distinctiveness
        ]

    def match(self, candidate_canon: Optional[np.ndarray]) -> list[MatchScore]:
        if candidate_canon is None or candidate_canon.std() == 0:
            return []
        cf = candidate_canon.astype(np.float32)
        results: list[MatchScore] = []
        for sid in self.eligible_ids:
            template = self.templates[sid]
            best_score, best_rot = -1.0, 0
            for angle, mask in template.rotations.items():
                score = float(cv2.matchTemplate(cf, mask.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
                if score > best_score:
                    best_score, best_rot = score, angle
            results.append(MatchScore(sid, best_score, best_rot))
        results.sort(key=lambda r: -r.score)
        return results


_SINGLETON: Optional[SymbolLibrary] = None


def get_library() -> SymbolLibrary:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = SymbolLibrary()
    return _SINGLETON
