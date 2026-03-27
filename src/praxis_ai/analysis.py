from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .models import AnalysisReport, JointSeries, Landmark, Limitation, PoseFrame, PoseSequence
from .reference_data import load_reference_patterns


ANGLE_TRIPLETS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow_flexion": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_flexion": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_abduction": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_abduction": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip_flexion": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_flexion": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee_flexion": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_flexion": ("right_hip", "right_knee", "right_ankle"),
}


def _vector(a: Landmark, b: Landmark) -> np.ndarray:
    return np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=float)


def compute_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    ab = _vector(a, b)
    cb = _vector(c, b)
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom == 0:
        return 0.0
    cosine = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def compute_joint_series(sequence: PoseSequence) -> Dict[str, JointSeries]:
    series: Dict[str, List[float]] = {name: [] for name in ANGLE_TRIPLETS}
    for frame in sequence.frames:
        for name, triplet in ANGLE_TRIPLETS.items():
            if all(key in frame.landmarks for key in triplet):
                a, b, c = (frame.landmarks[key] for key in triplet)
                series[name].append(compute_angle(a, b, c))
    return {name: JointSeries(name=name, values=values) for name, values in series.items() if values}


def resample(values: Iterable[float], target_len: int = 32) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return np.zeros(target_len, dtype=float)
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=float)
    old_index = np.linspace(0.0, 1.0, len(values))
    new_index = np.linspace(0.0, 1.0, target_len)
    return np.interp(new_index, old_index, values)


def normalized_distance(a: Iterable[float], b: Iterable[float]) -> float:
    aa = resample(a)
    bb = resample(b)
    spread = max(np.ptp(bb), 1.0)
    return float(np.mean(np.abs(aa - bb)) / spread)


def symmetry_score(series: Dict[str, JointSeries]) -> float:
    pairs = [
        ("left_elbow_flexion", "right_elbow_flexion"),
        ("left_shoulder_abduction", "right_shoulder_abduction"),
        ("left_hip_flexion", "right_hip_flexion"),
        ("left_knee_flexion", "right_knee_flexion"),
    ]
    scores: List[float] = []
    for left, right in pairs:
        if left in series and right in series:
            distance = normalized_distance(series[left].values, series[right].values)
            scores.append(max(0.0, 100.0 - distance * 100.0))
    return sum(scores) / len(scores) if scores else 0.0


def smoothness_score(series: Dict[str, JointSeries]) -> float:
    scores: List[float] = []
    for joint in series.values():
        values = resample(joint.values, 48)
        velocity = np.diff(values)
        acceleration = np.diff(velocity)
        jerk = np.diff(acceleration)
        cost = float(np.mean(np.abs(jerk))) if len(jerk) else 0.0
        scores.append(max(0.0, 100.0 - cost * 3.0))
    return sum(scores) / len(scores) if scores else 0.0


def match_reference(series: Dict[str, JointSeries], base_dir: Path) -> Tuple[str, Dict[str, float], float]:
    references = load_reference_patterns(base_dir)
    best_name = "unknown"
    best_score = -1.0
    best_joint_scores: Dict[str, float] = {}
    for name, reference in references.items():
        joint_scores: Dict[str, float] = {}
        for joint_name, pattern in reference["joint_patterns"].items():
            if joint_name not in series:
                continue
            distance = normalized_distance(series[joint_name].values, pattern)
            joint_scores[joint_name] = max(0.0, 100.0 - distance * 100.0)
        if not joint_scores:
            continue
        score = sum(joint_scores.values()) / len(joint_scores)
        if score > best_score:
            best_name = name
            best_score = score
            best_joint_scores = joint_scores
    return best_name, best_joint_scores, max(best_score, 0.0)


def infer_feedback(
    matched_reference: str,
    reference_score: float,
    symmetry: float,
    smoothness: float,
    joint_scores: Dict[str, float],
) -> List[str]:
    feedback = [
        f"Matched movement pattern: {matched_reference.replace('_', ' ')}.",
        f"Reference similarity is {reference_score:.1f}/100, indicating how closely the movement follows the UCF101-derived template.",
        f"Left-right coordination score is {symmetry:.1f}/100.",
        f"Motion smoothness score is {smoothness:.1f}/100.",
    ]
    low_joints = [name for name, score in sorted(joint_scores.items(), key=lambda item: item[1]) if score < 75]
    if low_joints:
        feedback.append(
            "Lowest-quality joints: " + ", ".join(low_joints[:3]).replace("_", " ") + "."
        )
    return feedback


def analyze_pose(sequence: PoseSequence, limitations: List[Limitation], exercises, base_dir: Path) -> AnalysisReport:
    series = compute_joint_series(sequence)
    matched_reference, joint_scores, reference_score = match_reference(series, base_dir)
    symmetry = symmetry_score(series)
    smoothness = smoothness_score(series)
    joint_component = sum(joint_scores.values()) / len(joint_scores) if joint_scores else 0.0
    limitation_penalty = 5.0 * len(limitations)
    overall = max(0.0, min(100.0, 0.55 * reference_score + 0.25 * symmetry + 0.20 * smoothness - limitation_penalty))
    feedback = infer_feedback(matched_reference, reference_score, symmetry, smoothness, joint_scores)
    return AnalysisReport(
        label=sequence.label,
        inferred_action=matched_reference,
        overall_score=overall,
        reference_score=reference_score,
        symmetry_score=symmetry,
        smoothness_score=smoothness,
        joint_scores=joint_scores,
        joint_series=series,
        limitations=limitations,
        exercises=exercises,
        feedback=feedback,
        matched_reference=matched_reference,
        metadata=sequence.metadata,
    )
