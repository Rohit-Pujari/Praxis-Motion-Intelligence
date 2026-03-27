from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MAPPING = {
    "left_hip_flexion": ("HipAngles", "x"),
    "right_hip_flexion": ("HipAngles", "x"),
    "left_knee_flexion": ("KneeAngles", "x"),
    "right_knee_flexion": ("KneeAngles", "x"),
    "left_shoulder_abduction": ("ShoulderAngles", "x"),
    "right_shoulder_abduction": ("ShoulderAngles", "x"),
    "left_elbow_flexion": ("ElbowAngles", "x"),
    "right_elbow_flexion": ("ElbowAngles", "x"),
}


def collect_stats(path: Path, trial_keys: list[str], joint: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
    roms = []
    means = []
    with h5py.File(path, "r") as handle:
        for trial_key in trial_keys:
            dataset = handle["Sub"][trial_key]
            for index in range(dataset.shape[0]):
                arr = np.asarray(handle[dataset[index, 0]][joint][axis][()]).reshape(-1, 1001)
                rom = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
                mean = np.nanmean(arr, axis=1)
                roms.extend(float(x) for x in rom if np.isfinite(x))
                means.extend(float(x) for x in mean if np.isfinite(x))
    return np.asarray(roms, dtype=float), np.asarray(means, dtype=float)


def main() -> None:
    able_path = PROJECT_ROOT / "dataset" / "impaired" / "MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat"
    stroke_path = PROJECT_ROOT / "dataset" / "impaired" / "MAT_normalizedData_PostStrokeAdults_v27-02-23.mat"
    able_keys = ["LsideSegm_LsideData", "RsideSegm_RsideData"]
    stroke_keys = ["NsideSegm_NsideData", "PsideSegm_PsideData"]

    payload = {
        "source": "dataset/impaired MATLAB clinical dataset",
        "mapping": {
            metric: {"joint": joint, "axis": axis}
            for metric, (joint, axis) in MAPPING.items()
        },
        "joint_thresholds": {},
    }

    for metric, (joint, axis) in MAPPING.items():
        able_rom, able_mean = collect_stats(able_path, able_keys, joint, axis)
        stroke_rom, _ = collect_stats(stroke_path, stroke_keys, joint, axis)
        minimum_rom = float((np.percentile(able_rom, 25) + np.percentile(stroke_rom, 75)) / 2.0)
        mean_angle = float(np.mean(able_mean))
        std_angle = float(np.std(able_mean))
        payload["joint_thresholds"][metric] = {
            "minimum_rom": round(minimum_rom, 2),
            "able_mean_angle": round(mean_angle, 2),
            "able_std_angle": round(std_angle, 2),
            "normal_angle_min": round(mean_angle - 2.0 * std_angle, 2),
            "normal_angle_max": round(mean_angle + 2.0 * std_angle, 2),
        }

    output_path = PROJECT_ROOT / "data" / "stroke_thresholds.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
