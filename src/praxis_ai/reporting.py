from __future__ import annotations

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
