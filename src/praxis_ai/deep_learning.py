from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import JointSeries, PoseSequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    F = None


JOINT_ORDER: List[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

GRAPH_EDGES: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

LABELS = ["Normal", "Injury Recovery", "Severe Limitation"]


def torch_available() -> bool:
    return torch is not None


def best_device():
    if not torch_available():
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_path(base_dir: Path) -> Path:
    return base_dir / "models" / "stgcn_transformer_demo.pt"


def graph_adjacency() -> np.ndarray:
    size = len(JOINT_ORDER)
    index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    adjacency = np.eye(size, dtype=np.float32)
    for start_name, end_name in GRAPH_EDGES:
        start_idx = index[start_name]
        end_idx = index[end_name]
        adjacency[start_idx, end_idx] = 1.0
        adjacency[end_idx, start_idx] = 1.0
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.clip(degree, 1.0, None)


def pose_sequence_to_array(sequence: PoseSequence, target_frames: int = 48) -> np.ndarray:
    frames = np.zeros((len(sequence.frames), len(JOINT_ORDER), 4), dtype=np.float32)
    if not sequence.frames:
        return np.zeros((target_frames, len(JOINT_ORDER), 4), dtype=np.float32)

    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    for frame_idx, frame in enumerate(sequence.frames):
        for landmark_name, landmark in frame.landmarks.items():
            if landmark_name not in joint_index:
                continue
            target_idx = joint_index[landmark_name]
            frames[frame_idx, target_idx, :] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

    if len(sequence.frames) == target_frames:
        return frames

    original_positions = np.linspace(0.0, 1.0, len(sequence.frames))
    target_positions = np.linspace(0.0, 1.0, target_frames)
    resampled = np.zeros((target_frames, len(JOINT_ORDER), 4), dtype=np.float32)
    for joint_idx in range(len(JOINT_ORDER)):
        for feature_idx in range(4):
            resampled[:, joint_idx, feature_idx] = np.interp(
                target_positions,
                original_positions,
                frames[:, joint_idx, feature_idx],
            )
    return resampled


def build_joint_importance_from_series(series: Dict[str, JointSeries]) -> Dict[str, float]:
    joint_importance: Dict[str, float] = {}
    if not series:
        return joint_importance
    max_rom = max((joint.rom for joint in series.values()), default=1.0)
    landmark_projection = {
        "left_elbow_flexion": ["left_elbow", "left_wrist"],
        "right_elbow_flexion": ["right_elbow", "right_wrist"],
        "left_shoulder_abduction": ["left_shoulder"],
        "right_shoulder_abduction": ["right_shoulder"],
        "left_hip_flexion": ["left_hip"],
        "right_hip_flexion": ["right_hip"],
        "left_knee_flexion": ["left_knee", "left_ankle"],
        "right_knee_flexion": ["right_knee", "right_ankle"],
    }
    for joint_name, joint in series.items():
        importance = float(joint.rom / max(max_rom, 1.0))
        for landmark_name in landmark_projection.get(joint_name, []):
            joint_importance[landmark_name] = round(max(joint_importance.get(landmark_name, 0.0), importance), 3)
    return joint_importance


if torch_available():

    class SpatialGraphConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray) -> None:
            super().__init__()
            self.register_buffer("adjacency", torch.tensor(adjacency, dtype=torch.float32))
            self.projection = nn.Linear(in_channels, out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            aggregated = torch.einsum("vw,btwc->btvc", self.adjacency, x)
            return self.projection(aggregated)


    class STGCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, adjacency: np.ndarray) -> None:
            super().__init__()
            self.graph = SpatialGraphConv(in_channels, out_channels, adjacency)
            self.temporal = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm = nn.LayerNorm(out_channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.graph(x)
            batch_size, frames, joints, channels = x.shape
            x = self.norm(x)
            temporal_in = x.permute(0, 2, 3, 1).reshape(batch_size * joints, channels, frames)
            temporal_out = self.temporal(temporal_in)
            temporal_out = temporal_out.reshape(batch_size, joints, channels, frames).permute(0, 3, 1, 2)
            return F.gelu(temporal_out + x)


    class TemporalEncoder(nn.Module):
        def __init__(self, hidden_dim: int, num_layers: int = 2, num_heads: int = 4, use_transformer: bool = True) -> None:
            super().__init__()
            self.use_transformer = use_transformer
            if use_transformer:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            else:
                self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_transformer:
                return self.encoder(x)
            encoded, _ = self.encoder(x)
            return encoded


    class STGCNTransformerModel(nn.Module):
        def __init__(self, num_features: int = 4, hidden_dim: int = 64, num_classes: int = 3, use_transformer: bool = False) -> None:
            super().__init__()
            adjacency = graph_adjacency()
            self.stgcn_1 = STGCNBlock(num_features, hidden_dim, adjacency)
            self.stgcn_2 = STGCNBlock(hidden_dim, hidden_dim, adjacency)
            self.temporal_encoder = TemporalEncoder(hidden_dim, use_transformer=use_transformer)
            self.classifier = nn.Linear(hidden_dim, num_classes)
            self.deviation_head = nn.Linear(hidden_dim, 1)
            self.importance_head = nn.Linear(hidden_dim, len(JOINT_ORDER))

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            x = self.stgcn_1(x)
            x = self.stgcn_2(x)
            x = x.mean(dim=2)
            temporal_features = self.temporal_encoder(x)
            pooled = temporal_features.mean(dim=1)
            logits = self.classifier(pooled)
            deviation_score = torch.sigmoid(self.deviation_head(pooled)).squeeze(-1) * 100.0
            joint_importance = torch.softmax(self.importance_head(pooled), dim=-1)
            return {
                "logits": logits,
                "deviation_score": deviation_score,
                "joint_importance": joint_importance,
            }


def deep_model_available(base_dir: Path) -> bool:
    return torch_available() and model_path(base_dir).exists()


def load_deep_model(base_dir: Path, use_transformer: bool = False):
    if not torch_available():
        raise RuntimeError("PyTorch is not installed. Install torch to use the ST-GCN + Transformer model.")
    checkpoint_path = model_path(base_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Deep model checkpoint not found at {checkpoint_path}")
    device = best_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = STGCNTransformerModel(use_transformer=checkpoint.get("use_transformer", use_transformer))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_with_deep_model(
    sequence: PoseSequence,
    joint_series: Dict[str, JointSeries],
    base_dir: Path,
) -> Tuple[Optional[str], Optional[float], Dict[str, float]]:
    if not deep_model_available(base_dir):
        return None, None, {}

    model = load_deep_model(base_dir)
    sample = pose_sequence_to_array(sequence)
    device = best_device()
    with torch.no_grad():
        outputs = model(torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0))
        probabilities = torch.softmax(outputs["logits"], dim=-1)[0].cpu().numpy()
        label_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[label_idx]) * 100.0
        importance_values = outputs["joint_importance"][0].cpu().numpy()

    joint_importance = {
        joint_name: round(float(weight), 3)
        for joint_name, weight in zip(JOINT_ORDER, importance_values)
    }
    series_importance = build_joint_importance_from_series(joint_series)
    for joint_name, weight in series_importance.items():
        joint_importance[joint_name] = round(max(joint_importance.get(joint_name, 0.0), weight), 3)

    return LABELS[label_idx], round(confidence, 1), joint_importance


def save_demo_checkpoint(base_dir: Path) -> Path:
    if not torch_available():
        raise RuntimeError("PyTorch is not installed. Install torch to generate a demo checkpoint.")
    checkpoint_target = model_path(base_dir)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    model = STGCNTransformerModel(use_transformer=False)
    checkpoint = {
        "state_dict": model.state_dict(),
        "use_transformer": False,
        "labels": LABELS,
        "joint_order": JOINT_ORDER,
    }
    torch.save(checkpoint, checkpoint_target)
    return checkpoint_target
