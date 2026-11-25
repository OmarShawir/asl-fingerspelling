import json
from types import SimpleNamespace

import numpy as np

from fingerspelling.sequence_live import SequenceSession


class FakePredictor:
    checkpoint = "fake.pt"
    checkpoint_sha256 = "test"
    model = SimpleNamespace(config={"test": True})
    decoder_settings = {"name": "greedy"}

    def reset(self):
        self.state = 0

    def step(self, points):
        self.state += 1
        return "a" * self.state, np.zeros(28, dtype=np.float32)


def test_manual_boundaries_missing_frames_and_state_reset(tmp_path):
    predictor = FakePredictor()
    session = SequenceSession(predictor, tmp_path)
    session.feed([], 0)  # idle input cannot change states
    assert predictor.state == 0
    session.start(10)
    session.feed([], 20)
    session.feed([], 30)
    path = session.finish(40)
    record = json.loads(path.read_text())
    assert record["raw_text"] == "aa" and record["final_text"] == "aa"
    assert record["accuracy"] is None and record["target"] is None
    assert record["segment_duration_ms"] == 30
    assert len(record["frames"]) == 2 and all(x["hand_count"] == 0 for x in record["frames"])
    assert predictor.state == 0
    session.start(50)
    session.feed([], 60)
    session.start(70)  # restart logs abandoned attempt and resets both recognizers
    assert predictor.state == 0 and session.text == ""
    session.feed([], 80)
    session.clear(90)
    assert predictor.state == 0 and session.text == "" and not session.active
    assert {json.loads(p.read_text())["status"] for p in tmp_path.glob("*.json")} == {
        "completed",
        "restarted",
        "cleared",
    }


def test_consecutive_words_do_not_inherit_text(tmp_path):
    session = SequenceSession(FakePredictor(), tmp_path)
    texts = []
    for start in [0, 100]:
        session.start(start)
        session.feed([], start + 1)
        path = session.finish(start + 2)
        texts.append(json.loads(path.read_text())["final_text"])
    assert texts == ["a", "a"]


def test_camera_keys_use_manual_boundaries_and_release(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from fingerspelling import sequence_live as live

    class Camera:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((240, 640, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    keys = iter([ord("r"), 0, 13, ord("r"), ord("c"), ord("q")])

    @contextmanager
    def tracker(settings, num_hands):
        assert num_hands == 2
        yield object()

    monkeypatch.setattr(live, "SequencePredictor", lambda *args: FakePredictor())
    monkeypatch.setattr(live, "open_tracker", tracker)
    monkeypatch.setattr(live, "extract_hands", lambda *args: [])
    monkeypatch.setattr(live.cv2, "VideoCapture", lambda index: camera)
    monkeypatch.setattr(live.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(live.cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(live.cv2, "destroyAllWindows", lambda: None)
    config = {"camera": {"index": 0, "mirror_preview": True}, "tracker": {}}
    live.capture_sequence(config, "unused.pt", "unused.json", log_dir=tmp_path)
    records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert camera.released
    assert len(records) == 2
    assert next(r for r in records if r["status"] == "completed")["final_text"] == "aa"
    assert next(r for r in records if r["status"] == "cleared")["raw_text"] == "a"
