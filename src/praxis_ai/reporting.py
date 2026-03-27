from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .models import AnalysisReport, JointSeries


def score_band(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "watch"
    return "limited"


def joint_summary(series: Dict[str, JointSeries]) -> List[dict]:
    rows: List[dict] = []
    for name, joint in sorted(series.items()):
        rows.append(
            {
                "joint": name.replace("_", " "),
                "minimum": round(joint.minimum, 1),
                "maximum": round(joint.maximum, 1),
                "rom": round(joint.rom, 1),
                "mean": round(joint.mean, 1),
            }
        )
    return rows


def report_context(report: AnalysisReport) -> dict:
    return {
        "report": report,
        "score_band": score_band(report.overall_score),
        "joint_summary": joint_summary(report.joint_series),
    }


def serialize_report(report: AnalysisReport) -> dict:
    return {
        "label": report.label,
        "inferred_action": report.inferred_action,
        "overall_score": round(report.overall_score, 1),
        "mobility_score": round(report.reference_score, 1),
        "symmetry_score": round(report.symmetry_score, 1),
        "smoothness_score": round(report.smoothness_score, 1),
        "joint_scores": {name: round(score, 1) for name, score in report.joint_scores.items()},
        "joint_summary": joint_summary(report.joint_series),
        "limitations": [asdict(item) for item in report.limitations],
        "exercises": [asdict(item) for item in report.exercises],
        "feedback": report.feedback,
        "metadata": report.metadata,
        "score_band": score_band(report.overall_score),
    }
