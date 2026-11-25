"""Paired CLI comparison fixes weights, clips and class set; legacy clips fail v2 explicitly."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from fingerspelling.dataset import ClipLabel, save_clip
from fingerspelling.features import FEATURE_V1, FEATURE_V2, LABELS
from fingerspelling.models.mlp import StaticMLP


def test_paired_cli_and_legacy_rejection(tmp_path):
    state = StaticMLP().state_dict()
    for name, version in [("v1", FEATURE_V1), ("v2", FEATURE_V2)]:
        torch.save(
            dict(state_dict=state, feature_version=version, labels=LABELS),
            tmp_path / (name + ".pt"),
        )
    data = tmp_path / "data"
    xyz = np.random.default_rng(3).random((3, 21, 3), dtype=np.float32)
    for label, width, height in [("M", 640, 480), ("N", 200, 200)]:
        save_clip(
            data,
            ClipLabel(label, "letter", "synthetic", "test", "left"),
            xyz,
            [0, 50, 100],
            frame_width=width,
            frame_height=height,
        )
    script = Path(__file__).parents[1] / "scripts/compare_aspect.py"
    prefix = tmp_path / "paired"
    command = [
        sys.executable,
        str(script),
        "--person",
        "synthetic",
        "--session",
        "test",
        "--data-dir",
        str(data),
        "--v1",
        str(tmp_path / "v1.pt"),
        "--v2",
        str(tmp_path / "v2.pt"),
        "--output-prefix",
        str(prefix),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(prefix.with_suffix(".json").read_text())
    assert report["identical_weights"] and len(report["labels"]) == 24
    assert len(report["clip_sha256"]) == 2
    assert report["checkpoints"]["v1"]["total"] == report["checkpoints"]["v2"]["total"] == 2
    legacy = save_clip(
        data, ClipLabel("M", "letter", "synthetic", "test", "left"), xyz, [0, 50, 100]
    )
    command[-1] = str(tmp_path / "rejected")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and legacy.stem in result.stderr
    assert "documented" in result.stderr
    assert not (tmp_path / "rejected_v2.json").exists()
