"""Feature extraction for Detector v1, shared identically by train.py and
infer.py so training-time and inference-time features can never drift apart.

Chosen deliberately: a hand-rolled Histogram-of-Oriented-Gradients-style
descriptor (Sobel gradients -> per-cell orientation histogram) over a
fixed-size grayscale resize. HOG-style descriptors are shape descriptors --
appropriate here because the candidate crops are plan-symbol regions (line
art, not photographs), need no GPU, produce small fixed-size output
regardless of the source crop's resolution, and are fully deterministic (no
random init). This is the "smallest practical" choice for a dataset of this
size (49 positive instances total across 2 classes) -- a CNN would have
orders of magnitude more parameters than training examples and would simply
memorize the train split.

NOTE: this is hand-rolled (Sobel + numpy histogram) rather than
`cv2.HOGDescriptor` because the installed opencv-python-headless build in
this environment does not expose the objdetect module's HOGDescriptor
binding (`AttributeError: module 'cv2' has no attribute 'HOGDescriptor'`,
confirmed live in this environment; cv2.Sobel is available in every opencv
build since it's core imgproc). The output is a lower-fidelity but
equivalent-in-spirit orientation-histogram shape descriptor.
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_IMAGE_SIZE = 96  # square, pixels
CELL_SIZE = 12  # -> 8x8 grid of cells over a 96x96 image
NBINS = 9  # unsigned orientation bins, 0-180 degrees, same convention as classical HOG


def _cell_orientation_histogram(gray: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0)  # unsigned gradient direction

    n_cells = FEATURE_IMAGE_SIZE // CELL_SIZE
    bin_edges = np.linspace(0.0, 180.0, NBINS + 1)
    out = np.zeros((n_cells, n_cells, NBINS), dtype=np.float32)
    for i in range(n_cells):
        for j in range(n_cells):
            cell_angle = angle[i * CELL_SIZE:(i + 1) * CELL_SIZE, j * CELL_SIZE:(j + 1) * CELL_SIZE]
            cell_mag = magnitude[i * CELL_SIZE:(i + 1) * CELL_SIZE, j * CELL_SIZE:(j + 1) * CELL_SIZE]
            hist, _ = np.histogram(cell_angle, bins=bin_edges, weights=cell_mag)
            norm = np.linalg.norm(hist) + 1e-6
            out[i, j] = hist / norm
    return out.flatten()


def extract_features(image: np.ndarray) -> np.ndarray:
    """`image` may be grayscale or BGR/RGB; returns a fixed-length 1D float
    vector. Fully deterministic: fixed resize interpolation, no random
    state anywhere in the pipeline."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    hog_vec = _cell_orientation_histogram(resized)
    # Append coarse intensity statistics (mean/std of 4x4 grid cells) --
    # cheap, deterministic, and gives the classifier a signal HOG alone
    # discards entirely: how much ink/fill is in each region (distinguishes
    # a filled fixture outline from a sparse line-only symbol).
    grid = 4
    step = FEATURE_IMAGE_SIZE // grid
    stats = []
    for i in range(grid):
        for j in range(grid):
            cell = resized[i * step:(i + 1) * step, j * step:(j + 1) * step]
            stats.append(cell.mean() / 255.0)
            stats.append(cell.std() / 255.0)
    return np.concatenate([hog_vec, np.array(stats, dtype=np.float32)])


def load_and_extract(image_path: str) -> np.ndarray | None:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return extract_features(image)
