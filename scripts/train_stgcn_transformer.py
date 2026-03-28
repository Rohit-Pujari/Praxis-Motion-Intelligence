from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.praxis_ai import deep_learning
from src.praxis_ai.models import Landmark, PoseFrame, PoseSequence


def jitter_sequence(sequence: PoseSequence, magnitude: float) -> PoseSequence:
    jittered_frames = []
    for frame in sequence.frames:
        landmarks = {}
        for name, landmark in frame.landmarks.items():
            landmarks[name] = Landmark(
                x=float(np.clip(landmark.x + np.random.normal(0.0, magnitude), 0.0, 1.0)),
                y=float(np.clip(landmark.y + np.random.normal(0.0, magnitude), 0.0, 1.0)),
                z=float(landmark.z + np.random.normal(0.0, magnitude * 0.5)),
                visibility=landmark.visibility,
            )
        jittered_frames.append(PoseFrame(timestamp=frame.timestamp, landmarks=landmarks))
    return PoseSequence(
        label=sequence.label,
        fps=sequence.fps,
        frames=jittered_frames,
        source_type=sequence.source_type,
        metadata=sequence.metadata,
    )


def synthetic_pose_sequence(label: str, severity: str) -> PoseSequence:
    frames = []
    for index in range(48):
        t = index / 47.0
        shoulder_wave = 0.12 * np.sin(t * np.pi * 2.0)
        knee_wave = 0.10 * np.sin(t * np.pi * 2.0 + 0.6)
        severity_scale = {"normal": 1.0, "injury": 0.68, "stroke": 0.38}[severity]
        asymmetry = {"normal": 0.01, "injury": 0.03, "stroke": 0.07}[severity]
        landmarks = {
            "nose": Landmark(0.50, 0.10),
            "left_shoulder": Landmark(0.42, 0.22),
            "right_shoulder": Landmark(0.58, 0.22 + asymmetry),
            "left_elbow": Landmark(0.36, 0.33 - shoulder_wave * severity_scale),
            "right_elbow": Landmark(0.64, 0.33 + shoulder_wave * (severity_scale - asymmetry)),
            "left_wrist": Landmark(0.31, 0.47 - shoulder_wave * severity_scale),
            "right_wrist": Landmark(0.69, 0.47 + shoulder_wave * (severity_scale - asymmetry)),
            "left_hip": Landmark(0.45, 0.50),
            "right_hip": Landmark(0.55, 0.50 + asymmetry),
            "left_knee": Landmark(0.44, 0.66 + knee_wave * severity_scale),
            "right_knee": Landmark(0.56, 0.66 - knee_wave * (severity_scale - asymmetry)),
            "left_ankle": Landmark(0.43, 0.86 + knee_wave * severity_scale * 0.5),
            "right_ankle": Landmark(0.57, 0.86 - knee_wave * (severity_scale - asymmetry) * 0.5),
        }
        frames.append(PoseFrame(timestamp=index / 24.0, landmarks=landmarks))
    return PoseSequence(label=label, fps=24.0, frames=frames, source_type="synthetic")


def build_demo_dataset():
    samples = []
    label_map = {"normal": 0, "injury": 1, "stroke": 2}
    for severity in ["normal", "injury", "stroke"]:
        for seed in range(36):
            random.seed(seed)
            np.random.seed(seed)
            sequence = synthetic_pose_sequence(f"{severity}_{seed}", severity)
            samples.append((deep_learning.pose_sequence_to_array(jitter_sequence(sequence, 0.005 + seed * 0.0001)), label_map[severity]))
    random.shuffle(samples)
    return samples


def train_demo_model(epochs: int = 8, learning_rate: float = 1e-3) -> Path:
    if not deep_learning.torch_available():
        raise RuntimeError("PyTorch is not installed. Install torch to train the ST-GCN + Transformer demo model.")

    import torch
    import torch.nn.functional as F

    samples = build_demo_dataset()
    device = deep_learning.best_device()
    model = deep_learning.STGCNTransformerModel(use_transformer=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for sample, label in samples:
            inputs = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
            target = torch.tensor([label], dtype=torch.long, device=device)
            outputs = model(inputs)
            classification_loss = F.cross_entropy(outputs["logits"], target)
            deviation_target = torch.tensor([label * 40.0], dtype=torch.float32, device=device)
            deviation_loss = F.l1_loss(outputs["deviation_score"], deviation_target)
            loss = classification_loss + 0.05 * deviation_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

    checkpoint_path = PROJECT_ROOT / "models" / "stgcn_transformer_demo.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "use_transformer": False,
            "labels": deep_learning.LABELS,
        },
        checkpoint_path,
    )
    metrics_path = PROJECT_ROOT / "data" / "deep_learning_training_summary.json"
    metrics_path.write_text(
        json.dumps({"epochs": epochs, "samples": len(samples), "checkpoint": str(checkpoint_path)}, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path


def main() -> None:
    if not deep_learning.torch_available():
        summary_path = PROJECT_ROOT / "data" / "deep_learning_training_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "PyTorch is not installed in the current environment",
                    "recommended_checkpoint": str(PROJECT_ROOT / "models" / "stgcn_transformer_demo.pt"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"PyTorch not installed. Wrote training summary to {summary_path}")
        return
    checkpoint = train_demo_model()
    print(f"Wrote trained checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
