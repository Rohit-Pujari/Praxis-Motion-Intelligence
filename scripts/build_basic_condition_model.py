from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.praxis_ai.basic_model import build_basic_condition_model


def main() -> None:
    model = build_basic_condition_model(PROJECT_ROOT)
    print(f"Wrote {PROJECT_ROOT / 'data' / 'basic_condition_model.json'}")
    print(f"Model type: {model['model_type']}")


if __name__ == "__main__":
    main()
