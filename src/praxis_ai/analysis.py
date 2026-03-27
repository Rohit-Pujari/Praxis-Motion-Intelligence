from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .calibration import get_target_rom
from .models import AnalysisReport, JointSeries, Landmark, Limitation, PoseFrame, PoseSequence


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


def smooth_signal(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window or window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


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


def smoothness_score(series: Dict[str, JointSeries], active_joints: List[str] | None = None) -> float:
    scores: List[float] = []
    relevant = active_joints or list(series.keys())
    for joint_name in relevant:
        if joint_name not in series:
            continue
        joint = series[joint_name]
        values = smooth_signal(resample(joint.values, 48), window=7)
        velocity = np.diff(values) / 1.0
        acceleration = np.diff(velocity)
        jerk = np.diff(acceleration)
        rom_scale = max(joint.rom, 8.0)
        normalized_jerk = float(np.mean(np.abs(jerk)) / rom_scale) if len(jerk) else 0.0
        scores.append(max(0.0, 100.0 - normalized_jerk * 280.0))
    return sum(scores) / len(scores) if scores else 0.0


def mobility_scores(series: Dict[str, JointSeries]) -> Dict[str, float]:
    target_rom = get_target_rom()
    scores: Dict[str, float] = {}
    for joint_name, joint in series.items():
        target = target_rom.get(joint_name, 30.0)
        ratio = min(joint.rom / target, 1.0)
        scores[joint_name] = max(0.0, min(100.0, ratio * 100.0))
    return scores


def active_joint_names(series: Dict[str, JointSeries]) -> List[str]:
    target_rom = get_target_rom()
    ranked = sorted(series.items(), key=lambda item: item[1].rom, reverse=True)
    if not ranked:
        return []

    active: List[str] = []
    for joint_name, joint in ranked:
        target = target_rom.get(joint_name, 30.0)
        if joint.rom >= max(12.0, target * 0.4):
            active.append(joint_name)

    if len(active) < 2:
        active = [joint_name for joint_name, _ in ranked[: min(4, len(ranked))]]

    return active


def infer_feedback(
    form_score: float,
    mobility_score: float,
    symmetry: float,
    smoothness: float,
    joint_scores: Dict[str, float],
    active_joints: List[str],
) -> List[str]:
    feedback = [
        f"Overall form quality score is {form_score:.1f}/100.",
        f"Active-joint mobility score is {mobility_score:.1f}/100.",
        f"Left-right coordination score is {symmetry:.1f}/100.",
        f"Motion smoothness score is {smoothness:.1f}/100.",
    ]
    if active_joints:
        feedback.append("Primary movement joints: " + ", ".join(active_joints[:4]).replace("_", " ") + ".")
    low_joints = [
        name for name, score in sorted(joint_scores.items(), key=lambda item: item[1]) if score < 70
    ]
    if low_joints:
        feedback.append(
            "Most limited joints: " + ", ".join(low_joints[:3]).replace("_", " ") + "."
        )
    else:
        feedback.append("No major joint-specific form deficits were detected in the measured sequence.")
    return feedback


def analyze_pose(sequence: PoseSequence, limitations: List[Limitation], exercises) -> AnalysisReport:
    series = compute_joint_series(sequence)
    all_joint_scores = mobility_scores(series)
    active_joints = active_joint_names(series)
    joint_scores = {name: score for name, score in all_joint_scores.items() if name in active_joints}
    mobility_score = sum(joint_scores.values()) / len(joint_scores) if joint_scores else 0.0
    symmetry = symmetry_score(series)
    smoothness = smoothness_score(series, active_joints=active_joints)
    limitation_penalty = min(18.0, 3.0 * len(limitations))
    overall = max(
        0.0,
        min(100.0, 0.55 * mobility_score + 0.20 * symmetry + 0.25 * smoothness - limitation_penalty),
    )
    feedback = infer_feedback(overall, mobility_score, symmetry, smoothness, joint_scores, active_joints)
    metadata = dict(sequence.metadata)
    metadata["analysis_mode"] = "form_only"
    metadata["mobility_score"] = f"{mobility_score:.1f}"
    metadata["active_joints"] = ",".join(active_joints)
    metadata["reference_source"] = "calibrated_rom"
    return AnalysisReport(
        label=sequence.label,
        inferred_action="form_analysis",
        overall_score=overall,
        reference_score=mobility_score,
        symmetry_score=symmetry,
        smoothness_score=smoothness,
        joint_scores=joint_scores,
        joint_series=series,
        limitations=limitations,
        exercises=exercises,
        feedback=feedback,
        matched_reference=None,
        metadata=metadata,
    )
