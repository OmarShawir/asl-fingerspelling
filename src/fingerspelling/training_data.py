"""Verified dimensions for cached Kaggle landmarks."""

import json
from pathlib import Path

from .features import validate_dimensions


def load_dimensions(path="data/cache/kaggle_dimensions.json"):
    return json.loads(Path(path).read_text())["images"]


def dimensions_for_row(row, dimensions):
    entry = dimensions.get(row[0])
    if entry is None or entry["pixel_sha256"] != row[2]:
        raise ValueError(
            f"Missing/stale verified image dimensions for {row[0]}; run derive_aspect_v2.py"
        )
    return validate_dimensions(entry["frame_width"], entry["frame_height"])
