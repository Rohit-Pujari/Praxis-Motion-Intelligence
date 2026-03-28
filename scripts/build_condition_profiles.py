from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INJURY_CSV_PATH = PROJECT_ROOT / "multimodal_sports_injury_dataset.csv"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_injury_csv_stats(path: Path) -> dict:
    if not path.exists():
        return {
            "source": "not_found",
            "rows": 0,
            "weighted_prevalence": 0.25,
            "joint_angle_mean": 112.0,
            "range_of_motion_mean": 125.0,
        }

    rows = 0
    injury_weight = 0.0
    joint_angle_total = 0.0
    range_of_motion_total = 0.0
    valid_angle_rows = 0
    valid_rom_rows = 0

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            injury_weight += float(row.get("injury_occurred") or 0.0)
            if row.get("joint_angles"):
                joint_angle_total += float(row["joint_angles"])
                valid_angle_rows += 1
            if row.get("range_of_motion"):
                range_of_motion_total += float(row["range_of_motion"])
                valid_rom_rows += 1

    weighted_prevalence = injury_weight / max(rows * 2.0, 1.0)
    return {
        "source": str(path),
        "rows": rows,
        "weighted_prevalence": weighted_prevalence,
        "joint_angle_mean": joint_angle_total / max(valid_angle_rows, 1),
        "range_of_motion_mean": range_of_motion_total / max(valid_rom_rows, 1),
    }


def main() -> None:
    ucf_stats = _read_json(DATA_DIR / "ucf_reference_stats.json").get("joint_stats", {})
    stroke_stats = _read_json(DATA_DIR / "stroke_thresholds.json").get("joint_thresholds", {})
    injury_csv_stats = _read_injury_csv_stats(INJURY_CSV_PATH)
    severity_blend = min(0.6, max(0.3, 0.3 + 0.4 * injury_csv_stats["weighted_prevalence"]))
    dispersion_scale = min(1.3, max(0.85, injury_csv_stats["range_of_motion_mean"] / 125.0))

    normal = {"source": "UCF101-derived normal movement profile", "joint_profiles": {}}
    injury = {
        "source": "Moderate injury-recovery profile calibrated with multimodal_sports_injury_dataset.csv and anchored between UCF101 normal and severe stroke reference anchors",
        "csv_source": injury_csv_stats["source"],
        "csv_rows": injury_csv_stats["rows"],
        "severity_blend": round(severity_blend, 3),
        "joint_profiles": {},
    }
    stroke = {"source": "Severe limitation profile anchored to stroke .mat threshold statistics", "joint_profiles": {}}

    for joint_name, joint_stats in sorted(ucf_stats.items()):
        normal_mean = float(joint_stats.get("mean_angle_mean", 0.0))
        normal_rom = float(joint_stats.get("rom_mean", 0.0))
        stroke_joint = stroke_stats.get(joint_name, {})
        stroke_mean = float(stroke_joint.get("able_mean_angle", normal_mean * 0.4))
        stroke_rom = min(float(stroke_joint.get("minimum_rom", normal_rom * 0.45)), normal_rom * 0.45 or 1.0)

        normal["joint_profiles"][joint_name] = {
            "mean_angle": round(normal_mean, 2),
            "angle_std": round(max(float(joint_stats.get("mean_angle_std", 0.0)), 8.0), 2),
            "rom_mean": round(normal_rom, 2),
            "rom_std": round(max(float(joint_stats.get("rom_std", 0.0)), 8.0), 2),
        }
        stroke["joint_profiles"][joint_name] = {
            "mean_angle": round(stroke_mean, 2),
            "angle_std": round(max(float(stroke_joint.get("able_std_angle", 0.0)), 5.0), 2),
            "rom_mean": round(stroke_rom, 2),
            "rom_std": 6.0,
        }
        injury_mean = normal_mean - (normal_mean - stroke_mean) * severity_blend
        injury_rom = normal_rom - (normal_rom - stroke_rom) * severity_blend
        injury["joint_profiles"][joint_name] = {
            "mean_angle": round(injury_mean, 2),
            "angle_std": round(9.0 * dispersion_scale, 2),
            "rom_mean": round(injury_rom, 2),
            "rom_std": round(8.0 * dispersion_scale, 2),
        }

    for name, payload in [("normal.json", normal), ("injury.json", injury), ("stroke.json", stroke)]:
        target = DATA_DIR / name
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
