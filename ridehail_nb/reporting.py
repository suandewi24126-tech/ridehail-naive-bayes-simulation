"""Helpers for saving trained model artifacts and experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def save_model(model: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def save_metrics(metrics: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Tipe tidak dapat diserialisasi: {type(value).__name__}")

    destination.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )
    return destination
