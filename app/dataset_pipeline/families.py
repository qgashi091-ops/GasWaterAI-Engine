"""v0.4 -- project/style-family grouping and near-duplicate detection.

Two plans from the SAME project (e.g. different sheets of one building, or
a later revision of the same drawing) must never end up split across
train/validation/test (task requirement) -- a detector could otherwise
"cheat" by memorizing a project's specific drawing style rather than
learning general component appearance, inflating validation/test scores.

GROUPING SIGNAL: a shared filename prefix. This uses the ORIGINAL filename
(available only from the local, gitignored id-mapping file -- see
audit.py's module docstring) purely as an internal computation; the
grouping OUTPUT this module returns is just {plan_id: family_id}, which
carries no identifying text at all, so it is safe to persist in a committed
report. Two plans are linked when their filename stems (extension
stripped) share a longest-common-prefix of at least MIN_LCP_CHARS AND that
prefix covers more than MIN_LCP_RATIO of the shorter stem -- both
thresholds are needed together: a short, generic stem (e.g. "Schema") can
trivially be a full prefix of a longer one by coincidence (ratio 1.0) with
almost no shared information (see MIN_LCP_CHARS), while two long,
genuinely-unrelated stems rarely share a long literal prefix by chance.
Verified on this batch's real 20 filenames: this correctly finds both real
families (a 3-sheet project sharing a project-number prefix, and a 2-plan
project sharing a project-number prefix across a 2023/2025 revision) and
produces zero false merges among the other 16, unrelated single-plan
projects.

NEAR-DUPLICATE DETECTION: exact content near-duplicates (the same page
re-exported/re-saved) are caught independently via a perceptual hash of a
downsampled page render -- content-based, so it also catches a duplicate
that was renamed and would otherwise be invisible to the filename signal.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymupdf

MIN_LCP_CHARS = 8
MIN_LCP_RATIO = 0.3
PHASH_SIZE = 16  # 16x16 downsampled grayscale -> a coarse but effective near-duplicate signature
PHASH_HAMMING_DUPLICATE_THRESHOLD = 12  # out of 256 bits


def _stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _lcp_len(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class _UnionFind:
    def __init__(self, keys: list[str]):
        self.parent = {k: k for k in keys}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_families(original_filenames: dict[str, str]) -> dict[str, str]:
    """original_filenames: {plan_id: original_filename} (from the LOCAL
    mapping only). Returns {plan_id: family_id} where family_id is an
    opaque, non-identifying label ("FAM-1", "FAM-2", ...) -- safe to persist
    anywhere, including committed reports."""
    stems = {pid: _stem(fn) for pid, fn in original_filenames.items()}
    uf = _UnionFind(list(stems.keys()))
    ids = list(stems.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            lcp = _lcp_len(stems[a], stems[b])
            if lcp >= MIN_LCP_CHARS and lcp / min(len(stems[a]), len(stems[b])) > MIN_LCP_RATIO:
                uf.union(a, b)

    groups: dict[str, list[str]] = {}
    for pid in ids:
        groups.setdefault(uf.find(pid), []).append(pid)

    family_of: dict[str, str] = {}
    for idx, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1])), start=1):
        label = f"FAM-{idx}" if len(members) > 1 else f"STANDALONE-{idx}"
        for pid in members:
            family_of[pid] = label
    return family_of


def _page_phash(pdf_bytes: bytes) -> Optional[np.ndarray]:
    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[0]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.05, 0.05), alpha=False)
        arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        gray = arr[:, :, 0].astype(np.float32) if pixmap.n >= 1 else None
        pdf.close()
        if gray is None or gray.size == 0:
            return None
        import cv2
        small = cv2.resize(gray, (PHASH_SIZE, PHASH_SIZE), interpolation=cv2.INTER_AREA)
        return (small > small.mean()).astype(np.uint8)
    except Exception:  # noqa: BLE001
        return None


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


@dataclass
class NearDuplicatePair:
    plan_id_a: str
    plan_id_b: str
    hamming_distance: int


def detect_near_duplicates(plans: dict[str, bytes]) -> list[NearDuplicatePair]:
    hashes = {pid: _page_phash(data) for pid, data in plans.items()}
    ids = [pid for pid, h in hashes.items() if h is not None]
    pairs: list[NearDuplicatePair] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            d = _hamming(hashes[a], hashes[b])
            if d <= PHASH_HAMMING_DUPLICATE_THRESHOLD:
                pairs.append(NearDuplicatePair(a, b, d))
    return pairs
