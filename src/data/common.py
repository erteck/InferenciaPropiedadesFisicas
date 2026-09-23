from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT / "Datos"
RAW = DATOS / "raw"
INTERIM = DATOS / "interim"
PROCESSED = DATOS / "processed"
SPLITS = DATOS / "splits"


def load_params(section: str) -> dict[str, Any]:
    with open(ROOT / "params.yaml") as fh:
        return yaml.safe_load(fh)[section]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
