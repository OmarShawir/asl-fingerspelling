"""Preview the tracker, or record labeled clips using scripts/collect.py."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, read_config
from .dataset import save_clip
from .hand_tracker import draw_hand, extract_points, next_timestamp_ms, open_tracker
from .word_builder import WordBuilder


WINDOW = "ASL fingerspelling"


def capture(
    config, camera_id=None, label=None, predictor=None, physical_hand=None, session_log=None
):
    if label is not None and predictor is not None:
        raise ValueError("Recording and recognition must run separately")
    builder = WordBuilder.from_config(config["static"]) if predictor is not None else None
    camera_id = config["camera"]["index"] if camera_id is None else camera_id
    camera = cv2.VideoCapture(camera_id)
    recording = False
    frames, times = [], []
    notice = "Space: start a take" if label else "Show one hand"
    previous_ms = -1
    try:
        if not camera.isOpened():
            raise SystemExit(
                f"Cannot open camera {camera_id}. Close other camera apps, "
                "check camera access, or try --camera 1."
            )
        with open_tracker(config["tracker"]) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise SystemExit("The webcam stopped sending frames. Reconnect it and retry.")
                previous_ms = next_timestamp_ms(previous_ms)
                points = extract_points(tracker, frame, previous_ms)
                if recording:
                    frames.append(points)
                    times.append(previous_ms)

                draw_hand(frame, points)
                preview = cv2.flip(frame, 1) if config["camera"]["mirror_preview"] else frame
                status = "21 landmarks" if np.isfinite(points).all() else "No hand"
                title = f"{label.text}  |  {status}" if label else status
                if builder is not None:
                    visible = bool(np.isfinite(points).all())
                    letter, confidence = None, 0.0
                    if visible:
                        probabilities = predictor.probabilities(points, physical_hand)[0]
                        best = int(probabilities.argmax())
                        letter, confidence = predictor.labels[best], float(probabilities[best])
                    event = builder.update(previous_ms, letter, confidence, visible)
                    title = f"Current: {letter or '-'}  {confidence:.0%}  |  {status}"
                    if event:
                        print(
                            f"Accepted {letter}: {event['acceptance_ms']} ms | {builder.text}",
                            flush=True,
                        )
                hint = "Recording | Space: save" if recording else notice
                lines = [(28, title), (55, hint), (82, "Q / Esc: quit")]
                if builder is not None:
                    # Keep the full text in the session log; show its tail in the camera window.
                    lines = [
                        (28, title),
                        (55, f"Text: {builder.text[-40:]}"),
                        (
                            82,
                            f"Candidate: {builder.candidate or '-'}  {builder.evidence_ms}/{builder.settings['hold_ms']} ms",
                        ),
                        (109, "Space: space | Backspace: delete | C: clear | Q: quit"),
                        (136, f"Accepted letters: {len(builder.events)}"),
                    ]
                for y, text in lines:
                    cv2.putText(
                        preview,
                        text,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 230, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    if recording:
                        print("Discarded unfinished take.")
                    break
                if label and key == ord(" "):
                    if not recording:
                        frames, times = [], []
                        recording = True
                    else:
                        recording = False
                        try:
                            dest = save_clip(config["data"]["directory"], label, frames, times)
                            print(f"Saved {label.text}: {dest}", flush=True)
                            notice = "Saved | Space: new take"
                        except ValueError as error:
                            print(error)
                            notice = "Empty take | Space: retry"
                elif builder is not None:
                    builder.handle_key(key)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if builder is not None:
            if session_log is None:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                session_log = Path("results/live_sessions") / f"{stamp}.json"
            builder.save_session(session_log)
            print(f"Letter acceptance time: {builder.summary()} | session: {session_log}")


def main():
    parser = argparse.ArgumentParser(description="Show a live hand skeleton from the webcam.")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Static MLP checkpoint for live letter predictions")
    parser.add_argument("--hand", choices=["left", "right"], help="Physical signing hand")
    parser.add_argument(
        "--session-log",
        type=Path,
        help="Recognition event log (default: unique file in results/live_sessions/)",
    )
    args = parser.parse_args()
    predictor = None
    if args.model:
        if not args.hand:
            parser.error("--model requires --hand for the physical signing hand")
        from .models.mlp import StaticPredictor

        predictor = StaticPredictor(args.model)
    capture(
        read_config(args.config),
        args.camera,
        predictor=predictor,
        physical_hand=args.hand,
        session_log=args.session_log,
    )


if __name__ == "__main__":
    main()
