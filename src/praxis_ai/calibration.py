from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


CALIBRATION_PATH = Path(__file__).resolve().parents[2] / "data" / "form_calibration.json"

DEFAULT_TARGET_ROM: Dict[str, float] = {
    "left_elbow_flexion": 45.0,
    "right_elbow_flexion": 45.0,
    "left_shoulder_abduction": 55.0,
    "right_shoulder_abduction": 55.0,
    "left_hip_flexion": 30.0,
    "right_hip_flexion": 30.0,
    "left_knee_flexion": 40.0,
    "right_knee_flexion": 40.0,
}


def load_form_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    with CALIBRATION_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_target_rom() -> Dict[str, float]:
    calibration = load_form_calibration()
    target_rom = dict(DEFAULT_TARGET_ROM)
    target_rom.update({key: float(value) for key, value in calibration.get("target_rom", {}).items()})
    return target_rom


def get_minimum_rom_overrides() -> Dict[str, float]:
    calibration = load_form_calibration()
    return {key: float(value) for key, value in calibration.get("minimum_rom", {}).items()}
