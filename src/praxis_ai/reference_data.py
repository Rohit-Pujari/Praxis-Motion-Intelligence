from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict


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


@lru_cache(maxsize=None)
def _load_json(path_str: str) -> Dict[str, dict]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_normal_profile(base_dir: Path) -> Dict[str, dict]:
    return _load_json(str(base_dir / "data" / "normal.json"))


def load_injury_profile(base_dir: Path) -> Dict[str, dict]:
    return _load_json(str(base_dir / "data" / "injury.json"))


def load_stroke_profile(base_dir: Path) -> Dict[str, dict]:
    return _load_json(str(base_dir / "data" / "stroke.json"))
