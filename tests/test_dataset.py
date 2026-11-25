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


def test_dimensions_png_and_legacy_index_compatibility(tmp_path):
    import json
    import cv2

    label = ClipLabel("M", "letter", "p01", "s03", "left")
    xyz = np.random.default_rng(2).random((3, 21, 3), dtype=np.float32)
    legacy = save_clip(tmp_path, label, xyz, [100, 150, 200])
    image = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    path = save_clip(
        tmp_path,
        label,
        xyz,
        [1000, 1050, 1100],
        frame_width=18,
        frame_height=12,
        diagnostic_frames={0: image.copy(), 2: image.copy()},
    )
    with np.load(path, allow_pickle=False) as clip:
        meta = json.loads(clip["meta_json"].item())
        assert (meta["frame_width"], meta["frame_height"]) == (18, 12)
        assert meta["dimensions_source"] == "mediapipe_input_frame.shape"
        np.testing.assert_array_equal(clip["t_ms"], [0, 50, 100])
    manifest_path = tmp_path / meta["diagnostic_manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["clip_id"] == path.stem
    assert [r["frame_index"] for r in manifest["images"]] == [0, 2]
    assert [r["t_ms"] for r in manifest["images"]] == [0, 100]
    assert [r["capture_timestamp_ms"] for r in manifest["images"]] == [1000, 1100]
    for entry in manifest["images"]:
        np.testing.assert_array_equal(cv2.imread(str(manifest_path.parent / entry["file"])), image)
    assert rebuild_index(tmp_path) == 2
    with (tmp_path / "metadata.csv").open() as f:
        rows = {r["clip_id"]: r for r in csv.DictReader(f)}
    assert rows[legacy.stem]["frame_width"] == ""
    assert rows[path.stem]["frame_height"] == "12"


def test_diagnostic_dimensions_must_match_original_image(tmp_path):
    label = ClipLabel("M", "letter", "p01", "s03", "left")
    xyz = np.ones((2, 21, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions"):
        save_clip(
            tmp_path,
            label,
            xyz,
            [0, 50],
            frame_width=30,
            frame_height=20,
            diagnostic_frames={0: np.zeros((10, 10, 3), dtype=np.uint8)},
        )
    assert not list(tmp_path.rglob("*.npz"))
