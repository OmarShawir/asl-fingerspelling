"""Exercise live-loop wiring and recording shortcuts without opening a camera."""

from contextlib import nullcontext
import json

import numpy as np

from fingerspelling import app
from fingerspelling.config import read_config
from fingerspelling.dataset import ClipLabel
from fingerspelling.features import LABELS


def fake_camera(monkeypatch, keys):
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in keys]
    points = np.random.default_rng(3).random((21, 3), dtype=np.float32)
    calls = {"closed": False, "display": [], "flips": 0}
    frame_iter, key_iter, times = iter(frames), iter(keys), iter(range(0, len(keys) * 100, 100))

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            return True, next(frame_iter)

        def release(self):
            calls["closed"] = True

    def flip(frame, axis):
        calls["flips"] += 1
        return np.flip(frame, axis=axis).copy()

    monkeypatch.setattr(app.cv2, "VideoCapture", lambda _: Camera())
    monkeypatch.setattr(app.cv2, "waitKey", lambda _: next(key_iter))
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app.cv2, "flip", flip)
    monkeypatch.setattr(
        app.cv2, "putText", lambda frame, text, *args: calls["display"].append(text)
    )
    monkeypatch.setattr(app, "open_tracker", lambda _: nullcontext(None))
    monkeypatch.setattr(app, "extract_points", lambda *args: points.copy())
    monkeypatch.setattr(app, "draw_hand", lambda *args: None)
    monkeypatch.setattr(app, "next_timestamp_ms", lambda _: next(times))
    return points, calls


def test_recognition_keys_preview_and_physical_hand(monkeypatch, tmp_path):
    points, calls = fake_camera(monkeypatch, [-1] * 8 + [ord(" "), 8, ord("C"), ord("q")])
    inputs = []

    class Predictor:
        labels = LABELS

        def probabilities(self, xyz, hand, *, frame_width=None, frame_height=None):
            assert (frame_width, frame_height) == (30, 20)
            inputs.append((xyz.copy(), hand))
            p = np.zeros((1, 24))
            p[0, LABELS.index("L")] = 1
            return p

    log = tmp_path / "session.json"
    app.capture(read_config(), predictor=Predictor(), physical_hand="left", session_log=log)
    report = json.loads(log.read_text())
    assert report["text"] == "" and report["accepted_letters"] == 1
    assert report["mean_acceptance_ms"] == 700
    assert "Text: L" in calls["display"] and "Text: L " in calls["display"]
    assert calls["flips"] == 12 and calls["closed"]
    assert all(hand == "left" and np.array_equal(xyz, points) for xyz, hand in inputs)


def test_recording_space_still_starts_and_saves(monkeypatch, tmp_path):
    _, calls = fake_camera(monkeypatch, [ord(" "), ord("C"), -1, ord(" "), ord("q")])
    saved = []

    def save_clip(directory, label, frames, times, **kwargs):
        assert (kwargs["frame_width"], kwargs["frame_height"]) == (30, 20)
        saved.append((label, list(frames), list(times)))
        return tmp_path / "clip.npz"

    monkeypatch.setattr(app, "save_clip", save_clip)
    label = ClipLabel("L", "letter", "p01", "s03", "left")
    app.capture(read_config(), label=label)
    assert len(saved) == 1 and saved[0][0] == label
    assert saved[0][2] == [100, 200, 300]
    assert calls["closed"]
    assert not list(tmp_path.glob("*.json"))


def test_diagnostic_png_is_unmirrored_and_before_drawing(monkeypatch, tmp_path):
    import cv2

    points, calls = fake_camera(monkeypatch, [ord(" "), -1, -1, ord(" "), ord("q")])
    original = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            return True, original.copy()

        def release(self):
            pass

    monkeypatch.setattr(app.cv2, "VideoCapture", lambda _: Camera())
    monkeypatch.setattr(app, "draw_hand", lambda frame, points: frame.fill(255))
    config = read_config()
    config["data"]["directory"] = tmp_path
    app.capture(
        config,
        label=ClipLabel("M", "letter", "p01", "s03", "left"),
        save_diagnostic_frames=True,
        diagnostic_every=2,
    )
    (path,) = list((tmp_path / "clips").glob("*.npz"))
    with np.load(path, allow_pickle=False) as c:
        meta = json.loads(c["meta_json"].item())
        assert (meta["frame_width"], meta["frame_height"]) == (30, 20)
        np.testing.assert_array_equal(c["xyz"][0], points)
    manifest_path = tmp_path / meta["diagnostic_manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert [r["frame_index"] for r in manifest["images"]] == [0, 2]
    for r in manifest["images"]:
        np.testing.assert_array_equal(cv2.imread(str(manifest_path.parent / r["file"])), original)
    assert calls["flips"] == 5


def test_resolution_change_discards_mixed_size_take(monkeypatch, tmp_path):
    fake_camera(monkeypatch, [ord(" "), -1, -1, ord("q")])
    shapes = iter([(20, 30), (20, 30), (30, 40), (30, 40)])

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            h, w = next(shapes)
            return True, np.zeros((h, w, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(app.cv2, "VideoCapture", lambda _: Camera())
    config = read_config()
    config["data"]["directory"] = tmp_path
    app.capture(
        config, label=ClipLabel("M", "letter", "p01", "s03", "left"), save_diagnostic_frames=True
    )
    assert not list(tmp_path.rglob("*.npz"))
    assert not list(tmp_path.rglob("*.png"))
