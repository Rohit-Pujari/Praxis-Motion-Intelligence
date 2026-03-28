from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from .models import JointSeries
from .reference_data import load_injury_profile, load_normal_profile, load_stroke_profile


FEATURE_ORDER: List[str] = [
    "left_elbow_flexion",
    "right_elbow_flexion",
    "left_shoulder_abduction",
    "right_shoulder_abduction",
    "left_hip_flexion",
    "right_hip_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
]


def _model_path(base_dir: Path) -> Path:
    return base_dir / "data" / "basic_condition_model.json"


def _flatten_profile(payload: dict) -> List[float]:
    joint_profiles = payload.get("joint_profiles") or {}
    vector: List[float] = []
    for joint_name in FEATURE_ORDER:
        joint = joint_profiles.get(joint_name, {})
        vector.append(float(joint.get("mean_angle", 0.0)))
        vector.append(float(joint.get("rom_mean", 0.0)))
    return vector


def _flatten_series(series: Dict[str, JointSeries]) -> List[float]:
    vector: List[float] = []
    for joint_name in FEATURE_ORDER:
        joint = series.get(joint_name)
        if joint is None:
            vector.extend([0.0, 0.0])
            continue
        vector.extend([float(joint.mean), float(joint.rom)])
    return vector


def build_basic_condition_model(base_dir: Path) -> dict:
    normal = load_normal_profile(base_dir)
    injury = load_injury_profile(base_dir)
    stroke = load_stroke_profile(base_dir)
    model = {
        "model_type": "nearest_centroid_profile_classifier",
        "feature_order": FEATURE_ORDER,
        "labels": ["Normal", "Injury Recovery", "Severe Limitation"],
        "centroids": {
            "Normal": _flatten_profile(normal),
            "Injury Recovery": _flatten_profile(injury),
            "Severe Limitation": _flatten_profile(stroke),
        },
        "sources": {
            "normal": normal.get("source", ""),
            "injury": injury.get("source", ""),
            "stroke": stroke.get("source", ""),
        },
    }
    _model_path(base_dir).write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_basic_condition_model(base_dir: Path) -> dict:
    path = _model_path(base_dir)
    if not path.exists():
        return build_basic_condition_model(base_dir)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def predict_condition_with_basic_model(series: Dict[str, JointSeries], base_dir: Path) -> Tuple[str, Dict[str, float]]:
    model = load_basic_condition_model(base_dir)
    sample = _flatten_series(series)
    distances: Dict[str, float] = {}
    for label, centroid in (model.get("centroids") or {}).items():
        squared_error = 0.0
        for sample_value, centroid_value in zip(sample, centroid):
            squared_error += (float(sample_value) - float(centroid_value)) ** 2
        distances[label] = math.sqrt(squared_error)

    if not distances:
        return "Normal", {}

    best_label = min(distances, key=distances.get)
    return best_label, {label: round(value, 2) for label, value in distances.items()}
