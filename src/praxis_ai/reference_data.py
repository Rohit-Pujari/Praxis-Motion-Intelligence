from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def load_reference_patterns(base_dir: Path) -> Dict[str, dict]:
    path = base_dir / "data" / "reference_patterns.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_ucf_reference_stats(base_dir: Path) -> Dict[str, dict]:
    path = base_dir / "data" / "ucf_reference_stats.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stroke_rules(base_dir: Path) -> Dict[str, dict]:
    path = base_dir / "data" / "stroke_rules.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stroke_thresholds(base_dir: Path) -> Dict[str, dict]:
    path = base_dir / "data" / "stroke_thresholds.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
