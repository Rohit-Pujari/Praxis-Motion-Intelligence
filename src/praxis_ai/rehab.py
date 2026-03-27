from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

from .calibration import get_minimum_rom_overrides
from .models import ExerciseGuide, JointSeries, Limitation
from .reference_data import load_stroke_rules, load_stroke_thresholds


def detect_limitations(
    series: Dict[str, JointSeries],
    base_dir: Path,
    relevant_joints: Optional[Set[str]] = None,
) -> List[Limitation]:
    rules = load_stroke_rules(base_dir)
    overrides = get_minimum_rom_overrides()
    threshold_stats = load_stroke_thresholds(base_dir).get("joint_thresholds", {})
    limitations: List[Limitation] = []

    for joint_name, rule in rules["joint_rom_thresholds"].items():
        if relevant_joints and joint_name not in relevant_joints:
            continue
        if joint_name not in series:
            continue
        rom = series[joint_name].rom
        stat_rule = threshold_stats.get(joint_name, {})
        minimum_rom = float(overrides.get(joint_name, stat_rule.get("minimum_rom", rule["minimum_rom"])))
        if rom < minimum_rom:
            limitations.append(
                Limitation(
                    joint=joint_name,
                    severity="moderate" if rom > minimum_rom * 0.7 else "high",
                    description=rule["description"],
                    evidence=f"Observed ROM {rom:.1f} deg, expected at least {minimum_rom:.1f} deg.",
                )
            )
    asymmetry_pairs = rules["asymmetry_pairs"]
    for pair in asymmetry_pairs:
        left = pair["left"]
        right = pair["right"]
        if relevant_joints and (left not in relevant_joints or right not in relevant_joints):
            continue
        if left not in series or right not in series:
            continue
        diff = abs(series[left].rom - series[right].rom)
        if diff > pair["max_difference"]:
            limitations.append(
                Limitation(
                    joint=f"{left}/{right}",
                    severity="moderate",
                    description=pair["description"],
                    evidence=f"ROM asymmetry is {diff:.1f} deg, above the {pair['max_difference']:.1f} deg threshold.",
                )
            )

    return limitations


def recommend_exercises(limitations: List[Limitation], inferred_action: str) -> List[ExerciseGuide]:
    exercise_bank = {
        "shoulder_abduction": ExerciseGuide(
            name="Supported Shoulder Abduction Reaches",
            target="shoulder mobility",
            dosage="3 sets of 8 slow repetitions",
            rationale="Improves frontal-plane shoulder control often reduced after stroke.",
            visual_cues=[
                "Stand or sit tall with the elbow softly bent.",
                "Raise the arm sideways to shoulder height with assistance if needed.",
                "Pause for one second, then lower with control.",
            ],
        ),
        "elbow_flexion": ExerciseGuide(
            name="Table-Supported Elbow Flexion Slides",
            target="elbow control",
            dosage="2 sets of 10 repetitions",
            rationale="Encourages controlled elbow flexion and extension without excessive trunk compensation.",
            visual_cues=[
                "Rest the forearm on a towel over a table.",
                "Slide the hand toward the shoulder, then extend back out slowly.",
                "Keep the shoulder relaxed and trunk centered.",
            ],
        ),
        "hip_flexion": ExerciseGuide(
            name="Marching Weight-Shift Drill",
            target="hip advancement",
            dosage="3 rounds of 20 seconds",
            rationale="Builds limb clearance and stance control relevant to gait recovery.",
            visual_cues=[
                "Shift weight to the stable leg before lifting the other knee.",
                "Lift to a comfortable height without leaning backward.",
                "Return the foot quietly and repeat rhythmically.",
            ],
        ),
        "knee_flexion": ExerciseGuide(
            name="Step-and-Bend Patterning",
            target="knee flexion during swing",
            dosage="3 sets of 6 passes each side",
            rationale="Targets reduced knee excursion and foot clearance during walking.",
            visual_cues=[
                "Step over a low marker with the affected limb.",
                "Let the knee bend naturally rather than hiking the hip.",
                "Use a wall or rail for balance if needed.",
            ],
        ),
        "symmetry": ExerciseGuide(
            name="Mirror-Assisted Bilateral Repetitions",
            target="left-right symmetry",
            dosage="2 minutes of continuous guided movement",
            rationale="Visual feedback helps reduce asymmetrical compensation patterns.",
            visual_cues=[
                "Perform the action in front of a mirror.",
                "Match the height and timing of both sides.",
                "Pause if one side starts moving faster than the other.",
            ],
        ),
    }

    recommended: List[ExerciseGuide] = []
    limitation_text = " ".join(item.joint for item in limitations)
    if "shoulder" in limitation_text:
        recommended.append(exercise_bank["shoulder_abduction"])
    if "elbow" in limitation_text:
        recommended.append(exercise_bank["elbow_flexion"])
    if "hip" in limitation_text:
        recommended.append(exercise_bank["hip_flexion"])
    if "knee" in limitation_text:
        recommended.append(exercise_bank["knee_flexion"])
    if "/" in limitation_text or len(limitations) > 1:
        recommended.append(exercise_bank["symmetry"])
    return recommended[:4]
