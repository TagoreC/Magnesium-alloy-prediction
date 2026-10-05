"""Utility functions for file I/O, serialization, and reporting."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types, pathlib paths, and inf/nan."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON with support for numpy and pathlib types."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metrics_json(filepath: Path, section: str, metrics: Dict[str, Any]) -> None:
    """Safely merge or update a section in metrics.json."""
    data = {}
    if filepath.exists():
        try:
            data = load_json(filepath)
        except Exception:
            data = {}
    data[section] = metrics
    save_json(data, filepath)


_MODEL_CACHE: Dict[str, Any] = {}


def save_model(model: Any, filepath: Path) -> None:
    """Serialize model object using joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    _MODEL_CACHE[str(filepath)] = model


def load_model(filepath: Path) -> Any:
    """Load serialized model object using joblib (with in-memory caching)."""
    key = str(filepath)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = joblib.load(filepath)
    _MODEL_CACHE[key] = model
    return model
