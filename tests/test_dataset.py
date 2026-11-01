"""Protect the clip time axis, missing-hand representation and index recovery."""

import csv

import numpy as np
import pytest

from fingerspelling.dataset import ClipLabel, rebuild_index, save_clip


def test_clip_round_trip_with_tracking_gap(tmp_path):
    label = ClipLabel("HELLO", "word", "p01", "s01", "right")
    xyz = np.zeros((3, 21, 3), dtype=np.float32)
    xyz[0, 0] = [0.25, 0.5, 0.0]
    xyz[1] = np.nan
    xyz[2, 0] = [0.75, 0.5, 0.0]
    dest = save_clip(tmp_path, label, xyz, [1000, 1031, 1080])
    with np.load(dest, allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["xyz"], xyz, equal_nan=True)
        assert saved["t_ms"].tolist() == [0, 31, 80]
        assert saved["valid"].tolist() == [True, False, True]
    index = tmp_path / "metadata.csv"
    index.unlink()
    assert rebuild_index(tmp_path) == 1
    with index.open(encoding="utf-8", newline="") as stream:
        (row,) = list(csv.DictReader(stream))
    assert (row["clip_id"], row["text"], row["person"], row["session"]) == (
        dest.stem,
        "HELLO",
        "p01",
        "s01",
    )
    assert float(row["fps"]) == 25.0
    assert row["source"] == "webcam"


def test_empty_hand_take_creates_no_example(tmp_path):
    label = ClipLabel("L", "letter", "p01", "s02", "left")
    with pytest.raises(ValueError, match="No hand"):
        save_clip(tmp_path, label, np.full((3, 21, 3), np.nan), [100, 140, 181])
    assert not (tmp_path / "metadata.csv").exists()
    assert not list(tmp_path.rglob("*.npz"))
