from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseFrame:
    timestamp: float
    landmarks: Dict[str, Landmark]


@dataclass
class PoseSequence:
    label: str
    fps: float
    frames: List[PoseFrame]
    source_type: str = "landmarks"
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class JointSeries:
    name: str
    values: List[float]

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def rom(self) -> float:
        return self.maximum - self.minimum

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0


@dataclass
class Limitation:
    joint: str
    severity: str
    description: str
    evidence: str


@dataclass
class ExerciseGuide:
    name: str
    target: str
    dosage: str
    rationale: str
    visual_cues: List[str]


@dataclass
class MotionAnnotation:
    id: str
    timestamp: float
    title: str
    detail: str
    severity: str
    joint: str


@dataclass
class RepSummary:
    joint: str
    repetitions: int
    average_rom: float
    rhythm_score: float


@dataclass
class AnalysisReport:
    label: str
    inferred_action: str
    overall_score: float
    mobility_score: float
    symmetry_score: float
    smoothness_score: float
    joint_scores: Dict[str, float]
    joint_series: Dict[str, JointSeries]
    limitations: List[Limitation]
    exercises: List[ExerciseGuide]
    feedback: List[str]
    annotations: List[MotionAnnotation] = field(default_factory=list)
    rep_summary: List[RepSummary] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
