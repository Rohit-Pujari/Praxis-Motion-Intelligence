from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
class AnalysisReport:
    label: str
    inferred_action: str
    overall_score: float
    reference_score: float
    symmetry_score: float
    smoothness_score: float
    joint_scores: Dict[str, float]
    joint_series: Dict[str, JointSeries]
    limitations: List[Limitation]
    exercises: List[ExerciseGuide]
    feedback: List[str]
    metadata: Dict[str, str] = field(default_factory=dict)
    matched_reference: Optional[str] = None
