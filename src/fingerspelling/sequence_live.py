"""Experimental user-segmented camera sequence recognition, separate from static mode."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np
import torch

from .hand_tracker import open_tracker, extract_hands, draw_hand, next_timestamp_ms
from .models.gru_ctc import SequenceGRUCTC
from .sequence_data import FEATURE_VERSION, VOCABULARY, hand_features, sha256
from .sequence_decode import make_decoder, log_probabilities


class SequencePredictor:
    def __init__(self, checkpoint, decoder_config):
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = sha256(checkpoint)
        self.decoder_settings = json.loads(Path(decoder_config).read_text())
        if self.decoder_settings["checkpoint_sha256"] != self.checkpoint_sha256:
            raise ValueError("Selected decoder belongs to a different checkpoint")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            bundle["feature_version"] != FEATURE_VERSION
            or bundle["vocabulary"] != VOCABULARY
            or bundle["blank_index"] != 0
        ):
            raise ValueError("Incompatible sequence checkpoint")
        if (
            self.decoder_settings["vocabulary"] != VOCABULARY
            or self.decoder_settings["blank_index"] != 0
            or self.decoder_settings["feature_version"] != FEATURE_VERSION
        ):
            raise ValueError("Incompatible decoder metadata")
        self.model = SequenceGRUCTC(**bundle["architecture"]).eval()
        self.model.load_state_dict(bundle["state_dict"])
        self.decoder = make_decoder(self.decoder_settings)
        self.reset()

    def reset(self):
        self.hidden = None
        self.decoder.reset()

    @torch.inference_mode()
    def step(self, points):
        # [] is a real missing timestep, not padding and not a forced CTC blank.
        x, mask = hand_features([points])
        embedding = self.model.frame_embeddings(x[None], mask[None])
        output, self.hidden = self.model.gru(embedding, self.hidden)
        logits = self.model.head(output)[0, 0].numpy().copy()
        text = self.decoder.step(log_probabilities(logits))
        return text, logits


class SequenceSession:
    def __init__(self, predictor, log_dir, settings=None):
        self.predictor = predictor
        self.log_dir = Path(log_dir)
        self.settings = settings or {}
        self.active = False
        self.text = ""
        self.frames = []
        self.predictor.reset()

    def start(self, timestamp_ms):
        if self.active:
            self.finish(timestamp_ms, "restarted")
        self.predictor.reset()
        self.text = ""
        self.frames = []
        self.start_ms = timestamp_ms
        self.start_utc = datetime.now(timezone.utc).isoformat()
        self.active = True

    def feed(self, points, timestamp_ms, width=None, height=None):
        if not self.active:
            return
        if timestamp_ms < self.start_ms or (
            self.frames and timestamp_ms <= self.frames[-1]["timestamp_ms"]
        ):
            raise ValueError("Non-increasing segment timestamps")
        begin = time.perf_counter()
        self.text, logits = self.predictor.step(points)
        elapsed = (time.perf_counter() - begin) * 1000
        self.frames.append(
            dict(
                timestamp_ms=timestamp_ms,
                hand_count=len(points) // 21,
                width=width,
                height=height,
                recognition_compute_ms=elapsed,
                logits=logits.tolist(),
            )
        )

    def finish(self, timestamp_ms, status="completed"):
        if not self.active:
            return None
        if timestamp_ms < (self.frames[-1]["timestamp_ms"] if self.frames else self.start_ms):
            raise ValueError("End precedes captured input")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex + ".json"
        )
        event = dict(
            status=status,
            raw_text=self.text,
            final_text=self.text if status == "completed" else None,
            target=None,
            accuracy=None,
            start_utc=self.start_utc,
            end_utc=datetime.now(timezone.utc).isoformat(),
            start_monotonic_ms=self.start_ms,
            end_monotonic_ms=timestamp_ms,
            segment_duration_ms=timestamp_ms - self.start_ms,
            checkpoint=self.predictor.checkpoint,
            checkpoint_sha256=self.predictor.checkpoint_sha256,
            architecture=self.predictor.model.config,
            feature_version=FEATURE_VERSION,
            decoder=self.predictor.decoder_settings,
            settings=self.settings,
            frames=self.frames,
            timing_note="segment duration is user-controlled, not inference latency; recognition_compute_ms excludes camera/tracker/display",
            limitations="User-segmented experimental recognition. No target supplied; no accuracy claim.",
        )
        with path.open("x") as stream:
            json.dump(event, stream, indent=2)
            stream.write("\n")
        self.active = False
        self.predictor.reset()
        return path

    def clear(self, timestamp_ms):
        path = self.finish(timestamp_ms, "cleared")
        self.predictor.reset()
        self.text = ""
        self.frames = []
        return path


def capture_sequence(
    config,
    checkpoint,
    decoder_config,
    camera_id=None,
    log_dir=Path("results/sequence_live_sessions"),
):
    torch.set_num_threads(4)
    predictor = SequencePredictor(checkpoint, decoder_config)
    camera_id = config["camera"]["index"] if camera_id is None else camera_id
    settings = dict(
        camera_index=camera_id,
        mirror_preview=config["camera"]["mirror_preview"],
        tracker={k: str(v) if isinstance(v, Path) else v for k, v in config["tracker"].items()},
        tracker_hands=2,
        coordinate_policy="unmirrored normalized image xyz; no aspect correction or hand flip",
        input_compatibility="Source converter hand order, resizing, mirroring, tracker version and FPS unknown",
    )
    session = SequenceSession(predictor, log_dir, settings)
    camera = cv2.VideoCapture(camera_id)
    previous = -1
    notice = "R: begin a word"
    try:
        if not camera.isOpened():
            raise SystemExit(f"Cannot open camera {camera_id}. Check access or use --camera 1.")
        with open_tracker(config["tracker"], num_hands=2) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise SystemExit("Camera stopped providing frames")
                previous = next_timestamp_ms(previous)
                hands = extract_hands(tracker, frame, previous)
                points = np.concatenate(hands).tolist() if hands else []
                height, width = frame.shape[:2]
                session.feed(points, previous, width, height)
                for hand in hands:
                    draw_hand(frame, hand)
                preview = cv2.flip(frame, 1) if config["camera"]["mirror_preview"] else frame
                lines = [
                    "EXPERIMENTAL SEQUENCE | user marks word boundaries",
                    "ACTIVE" if session.active else "IDLE",
                    f"Text: [{session.text[-65:]}]",
                    "R: new word | Enter: finish/save | C: clear/reset | Q: quit",
                    notice,
                ]
                for i, text in enumerate(lines):
                    cv2.putText(
                        preview,
                        text,
                        (10, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 230, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow("ASL experimental sequence", preview)
                key = cv2.waitKey(1) & 0xFF
                now = max(previous, time.monotonic_ns() // 1_000_000)
                if key in (ord("q"), ord("Q"), 27):
                    session.finish(now, "quit_unfinished")
                    break
                if key in (ord("r"), ord("R")):
                    session.start(now)
                    notice = "Recognizing: text may change while moving"
                elif key in (10, 13):
                    path = session.finish(now)
                    if path:
                        notice = "Final saved | R: next word"
                        print(f"Sequence raw final: {session.text!r} | {path}", flush=True)
                elif key in (ord("c"), ord("C")):
                    session.clear(now)
                    notice = "Cleared | R: begin a word"
    finally:
        try:
            if session.active:
                session.finish(max(previous, time.monotonic_ns() // 1_000_000), "interrupted")
        finally:
            camera.release()
            cv2.destroyAllWindows()
