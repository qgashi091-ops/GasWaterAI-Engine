"""Detector v1 inference -- loads the frozen model trained by
scripts/train_detector_v1.py and classifies a single symbol-candidate crop.

CPU-only (scikit-learn RandomForestClassifier.predict on a ~700-dim feature
vector for one crop is sub-millisecond on any modern CPU; no batching
needed for this engine's per-document call volume).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import extract_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v1" / "model.joblib"
CONFIG_PATH = REPO_ROOT / "data" / "dev_plans_v04" / "detector_v1" / "training_config.json"

_model = None
_config = None


@dataclass
class DetectorPrediction:
    available: bool
    predicted_class: str | None  # "kueche" | "wc_up" | "background" | None if unavailable
    probabilities: dict[str, float]
    model_version: str | None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "available": self.available, "predicted_class": self.predicted_class,
            "probabilities": self.probabilities, "model_version": self.model_version, "note": self.note,
        }


def _load():
    global _model, _config
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        _config = json.loads(CONFIG_PATH.read_text())
    return _model, _config


def predict(image: np.ndarray) -> DetectorPrediction:
    model, config = _load()
    if model is None:
        return DetectorPrediction(
            available=False, predicted_class=None, probabilities={}, model_version=None,
            note="Detector v1 model not found -- run scripts/train_detector_v1.py first.",
        )
    feats = extract_features(image).reshape(1, -1)
    proba = model.predict_proba(feats)[0]
    probs = {cls: round(float(p), 4) for cls, p in zip(model.classes_, proba)}
    predicted = model.classes_[int(np.argmax(proba))]
    return DetectorPrediction(
        available=True, predicted_class=predicted, probabilities=probs,
        model_version=config.get("dataset_version") if config else None,
    )
