from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.praxis_ai.analysis import compute_joint_series
from src.praxis_ai.calibration import CALIBRATION_PATH, DEFAULT_TARGET_ROM
from src.praxis_ai.pose_estimation import available_pose_estimator


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return 0.0
    return float(np.percentile(array, q))


def calibrate(dataset_dir: Path, max_per_class: int | None = None) -> dict:
    estimator = available_pose_estimator()
    if estimator is None:
        raise RuntimeError("No pose estimator is available for dataset calibration.")

    rom_by_joint: Dict[str, List[float]] = defaultdict(list)
    processed_files: List[str] = []
    skipped_files: List[str] = []
    class_counts: Dict[str, int] = {}

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        videos = sorted(class_dir.glob("*.avi"))
        if max_per_class is not None:
            videos = videos[:max_per_class]
        class_success = 0
        for video_path in videos:
            try:
                sequence = estimator.estimate(video_path)
            except Exception:
                skipped_files.append(str(video_path))
                continue
            if sequence is None or not sequence.frames:
                skipped_files.append(str(video_path))
                continue
            series = compute_joint_series(sequence)
            if not series:
                skipped_files.append(str(video_path))
                continue
            for joint_name, joint in series.items():
                rom_by_joint[joint_name].append(joint.rom)
            processed_files.append(str(video_path))
            class_success += 1
        class_counts[class_dir.name] = class_success

    target_rom: Dict[str, float] = {}
    minimum_rom: Dict[str, float] = {}
    summary: Dict[str, dict] = {}
    for joint_name, fallback in DEFAULT_TARGET_ROM.items():
        values = rom_by_joint.get(joint_name, [])
        if not values:
            target_rom[joint_name] = fallback
            minimum_rom[joint_name] = fallback * 0.55
            summary[joint_name] = {
                "count": 0,
                "median_rom": fallback,
                "p25_rom": fallback * 0.55,
                "p75_rom": fallback,
            }
            continue
        target_rom[joint_name] = round(percentile(values, 70), 1)
        minimum_rom[joint_name] = round(percentile(values, 30), 1)
        summary[joint_name] = {
            "count": len(values),
            "median_rom": round(median(values), 1),
            "p25_rom": round(percentile(values, 25), 1),
            "p75_rom": round(percentile(values, 75), 1),
        }

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_dir": str(dataset_dir),
        "processed_videos": len(processed_files),
        "skipped_videos": len(skipped_files),
        "max_per_class": max_per_class,
        "class_counts": class_counts,
        "target_rom": target_rom,
        "minimum_rom": minimum_rom,
        "joint_summary": summary,
        "processed_files": processed_files,
        "skipped_files": skipped_files[:50],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate form-scoring targets from the local dataset.")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to the dataset directory.")
    parser.add_argument("--max-per-class", type=int, default=3, help="Maximum videos to process per class.")
    parser.add_argument("--output", default=str(CALIBRATION_PATH), help="Where to write the calibration JSON.")
    args = parser.parse_args()

    payload = calibrate(Path(args.dataset_dir), max_per_class=args.max_per_class)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote calibration to {output_path}")
    print(f"Processed videos: {payload['processed_videos']}, skipped: {payload['skipped_videos']}")


if __name__ == "__main__":
    main()
