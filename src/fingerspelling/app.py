"""Preview the tracker, or record labeled clips using scripts/collect.py."""

import argparse

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, read_config
from .dataset import save_clip
from .hand_tracker import draw_hand, extract_points, next_timestamp_ms, open_tracker


WINDOW = "ASL fingerspelling"


def capture(config, camera_id=None, label=None):
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
                hint = "Recording | Space: save" if recording else notice
                for y, text in [(28, title), (55, hint), (82, "Q / Esc: quit")]:
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
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Show a live hand skeleton from the webcam.")
    parser.add_argument("--camera", type=int, help="Override the camera index")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    capture(read_config(args.config), args.camera)


if __name__ == "__main__":
    main()
