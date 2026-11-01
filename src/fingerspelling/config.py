"""Paths in the configuration are relative to the project root."""

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("configs/default.yaml")


def read_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"Cannot find {path}. Run from the project root or pass --config.")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # The committed configuration lives in <root>/configs/.
    root = path.parent.parent
    config["tracker"]["model_path"] = root / config["tracker"]["model_path"]
    config["data"]["directory"] = root / config["data"]["directory"]
    return config
