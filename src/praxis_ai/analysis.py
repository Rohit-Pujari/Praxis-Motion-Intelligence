from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .calibration import get_target_rom
from .models import (
    AnalysisReport,
    JointSeries,
    Landmark,
    Limitation,
    MotionAnnotation,
    PoseSequence,
    RepSummary,
)


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


def estimate_rep_summary(
    series: Dict[str, JointSeries],
    active_joints: List[str],
) -> List[RepSummary]:
    summaries: List[RepSummary] = []
    for joint_name in active_joints:
        joint = series.get(joint_name)
        if joint is None or len(joint.values) < 6:
            continue
        values = smooth_signal(np.asarray(joint.values, dtype=float), window=5)
        threshold = float(np.percentile(values, 65))
        peaks: List[int] = []
        for index in range(1, len(values) - 1):
            if values[index] > values[index - 1] and values[index] >= values[index + 1] and values[index] >= threshold:
                if peaks and index - peaks[-1] < 3:
                    if values[index] > values[peaks[-1]]:
                        peaks[-1] = index
                    continue
                peaks.append(index)

        reps = max(0, len(peaks))
        if reps == 0:
            continue

        if len(peaks) > 1:
            peak_gaps = np.diff(peaks).astype(float)
            gap_ratio = float(np.std(peak_gaps) / max(np.mean(peak_gaps), 1.0))
            rhythm_score = max(0.0, 100.0 - gap_ratio * 100.0)
        else:
            rhythm_score = 100.0

        summaries.append(
            RepSummary(
                joint=joint_name,
                repetitions=reps,
                average_rom=round(joint.rom / reps, 1),
                rhythm_score=round(rhythm_score, 1),
            )
        )
    return summaries


def build_motion_annotations(
    sequence: PoseSequence,
    series: Dict[str, JointSeries],
    active_joints: List[str],
    limitations: List[Limitation],
) -> List[MotionAnnotation]:
    annotations: List[MotionAnnotation] = []
    duration = sequence.frames[-1].timestamp if sequence.frames else 0.0
    annotation_index = 1

    for limitation in limitations[:6]:
        joint_name = limitation.joint.split("/")[0]
        timestamp = duration * min(0.9, 0.12 * annotation_index)
        if joint_name in series and series[joint_name].values:
            values = np.asarray(series[joint_name].values, dtype=float)
            min_index = int(np.argmin(values))
            timestamp = duration * (min_index / max(len(values) - 1, 1))
        annotations.append(
            MotionAnnotation(
                id=f"event-{annotation_index}",
                timestamp=round(timestamp, 2),
                title=f"{joint_name.replace('_', ' ')} limitation",
                detail=f"{limitation.description} {limitation.evidence}",
                severity=limitation.severity,
                joint=limitation.joint,
            )
        )
        annotation_index += 1

    left_right_pairs = [
        ("left_elbow_flexion", "right_elbow_flexion"),
        ("left_shoulder_abduction", "right_shoulder_abduction"),
        ("left_hip_flexion", "right_hip_flexion"),
        ("left_knee_flexion", "right_knee_flexion"),
    ]
    for left_name, right_name in left_right_pairs:
        left = series.get(left_name)
        right = series.get(right_name)
        if left is None or right is None:
            continue
        left_values = resample(left.values, 48)
        right_values = resample(right.values, 48)
        delta = np.abs(left_values - right_values)
        peak_index = int(np.argmax(delta))
        peak_delta = float(delta[peak_index])
        if peak_delta < 12.0:
            continue
        annotations.append(
            MotionAnnotation(
                id=f"event-{annotation_index}",
                timestamp=round(duration * (peak_index / max(len(delta) - 1, 1)), 2),
                title="Asymmetry spike",
                detail=(
                    f"{left_name.replace('_', ' ')} vs {right_name.replace('_', ' ')} "
                    f"diverged by {peak_delta:.1f} deg."
                ),
                severity="moderate" if peak_delta < 25 else "high",
                joint=f"{left_name}/{right_name}",
            )
        )
        annotation_index += 1

    for joint_name in active_joints[:4]:
        joint = series.get(joint_name)
        if joint is None or len(joint.values) < 8:
            continue
        values = smooth_signal(resample(joint.values, 48), window=5)
        velocity = np.abs(np.diff(values))
        quiet_index = int(np.argmin(velocity)) if len(velocity) else 0
        annotations.append(
            MotionAnnotation(
                id=f"event-{annotation_index}",
                timestamp=round(duration * (quiet_index / max(len(values) - 1, 1)), 2),
                title="Pacing check",
                detail=f"{joint_name.replace('_', ' ')} slowed noticeably here; check for hesitation or guarded motion.",
                severity="watch",
                joint=joint_name,
            )
        )
        annotation_index += 1

    annotations.sort(key=lambda item: item.timestamp)
    return annotations[:10]


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
    annotations = build_motion_annotations(sequence, series, active_joints, limitations)
    rep_summary = estimate_rep_summary(series, active_joints)
    metadata = dict(sequence.metadata)
    metadata["analysis_mode"] = "form_only"
    metadata["mobility_score"] = f"{mobility_score:.1f}"
    metadata["active_joints"] = ",".join(active_joints)
    metadata["calibration_source"] = "calibrated_rom"
    metadata["duration_seconds"] = f"{(sequence.frames[-1].timestamp if sequence.frames else 0.0):.2f}"
    return AnalysisReport(
        label=sequence.label,
        inferred_action="form_analysis",
        overall_score=overall,
        mobility_score=mobility_score,
        symmetry_score=symmetry,
        smoothness_score=smoothness,
        joint_scores=joint_scores,
        joint_series=series,
        limitations=limitations,
        exercises=exercises,
        feedback=feedback,
        annotations=annotations,
        rep_summary=rep_summary,
        metadata=metadata,
    )
