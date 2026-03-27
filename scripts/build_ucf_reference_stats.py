from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.praxis_ai.analysis import compute_joint_series
from src.praxis_ai.pose_estimation import available_pose_estimator


def main() -> None:
    dataset_dir = PROJECT_ROOT / "dataset" / "UCF101"
    output_path = PROJECT_ROOT / "data" / "ucf_reference_stats.json"
    estimator = available_pose_estimator()
    if estimator is None:
        raise RuntimeError("No pose estimator is available for UCF reference extraction.")

    rom_values: Dict[str, list[float]] = defaultdict(list)
    mean_values: Dict[str, list[float]] = defaultdict(list)
    per_class_counts: Dict[str, int] = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        success = 0
        for video_path in sorted(class_dir.glob("*.avi")):
            try:
                sequence = estimator.estimate(video_path)
            except Exception:
                continue
            if sequence is None or not sequence.frames:
                continue
            joint_series = compute_joint_series(sequence)
            for joint_name, joint in joint_series.items():
                rom_values[joint_name].append(joint.rom)
                mean_values[joint_name].append(joint.mean)
            success += 1
        per_class_counts[class_dir.name] = success

    payload = {
        "source": str(dataset_dir),
        "per_class_counts": per_class_counts,
        "joint_stats": {},
    }
    for joint_name in sorted(rom_values):
        rom_array = np.asarray(rom_values[joint_name], dtype=float)
        mean_array = np.asarray(mean_values[joint_name], dtype=float)
        payload["joint_stats"][joint_name] = {
            "rom_mean": round(float(np.mean(rom_array)), 2),
            "rom_std": round(float(np.std(rom_array)), 2),
            "mean_angle_mean": round(float(np.mean(mean_array)), 2),
            "mean_angle_std": round(float(np.std(mean_array)), 2),
        }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
