"""Store self-contained landmark clips and build the local CSV index."""

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np

from .config import DEFAULT_CONFIG, read_config
from .features import validate_dimensions


FIELDS = [
    "clip_id",
    "text",
    "kind",
    "person",
    "session",
    "hand",
    "fps",
    "frames",
    "created_at",
    "source",
    "frame_width",
    "frame_height",
    "dimensions_source",
    "diagnostic_manifest",
]


@dataclass(frozen=True)
class ClipLabel:
    text: str
    kind: str
    person: str
    session: str
    hand: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Z]+", self.text):
            raise ValueError("Use a transcript made of letters A-Z.")
        if self.kind not in ("letter", "transition", "word"):
            raise ValueError("Choose letter, transition or word.")
        required_length = {"letter": 1, "transition": 2}.get(self.kind)
        if required_length is not None and len(self.text) != required_length:
            raise ValueError(f"A {self.kind} clip needs {required_length} letter(s).")
        if self.hand not in ("left", "right"):
            raise ValueError("Specify the physical signing hand as left or right.")
        for code in (self.person, self.session):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", code):
                raise ValueError("Use letters, numbers, underscores or hyphens for person/session.")


def rebuild_index(data_dir):
    """Recreate metadata.csv from complete clips. Each NPZ retains its own label."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    clips = data_dir / "clips"
    while True:
        paths = sorted(clips.glob("*.npz"))
        temp = NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=data_dir,
            prefix=".metadata-",
            suffix=".tmp",
            delete=False,
        )
        temp_path = Path(temp.name)
        try:
            with temp:
                writer = csv.DictWriter(temp, fieldnames=FIELDS)
                writer.writeheader()
                for path in paths:
                    with np.load(path, allow_pickle=False) as clip:
                        writer.writerow(json.loads(clip["meta_json"].item()))
            temp_path.replace(data_dir / "metadata.csv")
        finally:
            temp_path.unlink(missing_ok=True)
        # If another collector finished a clip while we indexed, include it too.
        if paths == sorted(clips.glob("*.npz")):
            return len(paths)


def save_clip(
    data_dir,
    label,
    point_frames,
    timestamps,
    *,
    frame_width=None,
    frame_height=None,
    diagnostic_frames=None,
):
    """Save observed frames, including gaps, with elapsed capture times."""
    xyz = np.array(point_frames, dtype=np.float32, copy=True)
    t_ms = np.asarray(timestamps, dtype=np.int64)
    if len(t_ms) < 2 or xyz.shape != (len(t_ms), 21, 3):
        raise ValueError("Hold the pose longer and save at least two frames.")
    if np.any(np.diff(t_ms) <= 0):
        raise ValueError("Capture timestamps must be strictly increasing.")
    valid = np.isfinite(xyz).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("No hand was visible. Try this take again.")
    xyz[~valid] = np.nan
    t_ms = t_ms - t_ms[0]

    if frame_width is not None or frame_height is not None:
        validate_dimensions(frame_width, frame_height)
    diagnostic_frames = diagnostic_frames or {}
    if diagnostic_frames:
        validate_dimensions(frame_width, frame_height)
        for index, image in diagnostic_frames.items():
            if not isinstance(index, int) or not 0 <= index < len(t_ms):
                raise ValueError("Diagnostic frame index outside clip")
            if image.shape != (frame_height, frame_width, 3) or image.dtype != np.uint8:
                raise ValueError("Diagnostic image must match documented BGR frame dimensions")
    meta = asdict(label)
    meta.update(
        clip_id=uuid4().hex,
        fps=round(1000 * (len(t_ms) - 1) / int(t_ms[-1]), 2),
        frames=len(t_ms),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source="webcam",
    )
    if frame_width is not None:
        meta.update(
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            dimensions_source="mediapipe_input_frame.shape",
        )
    diagnostic_dir = None
    if diagnostic_frames:
        import cv2

        diagnostic_dir = Path(data_dir) / "diagnostic_frames" / meta["clip_id"]
        diagnostic_dir.mkdir(parents=True, exist_ok=False)
        try:
            images = []
            for index, image in sorted(diagnostic_frames.items()):
                filename = f"frame_{index:06d}_t_{int(t_ms[index]):010d}ms.png"
                if not cv2.imwrite(str(diagnostic_dir / filename), image):
                    raise OSError("Failed to write diagnostic PNG")
                images.append(
                    dict(
                        file=filename,
                        frame_index=index,
                        t_ms=int(t_ms[index]),
                        capture_timestamp_ms=int(timestamps[index]),
                    )
                )
            manifest = dict(
                clip_id=meta["clip_id"],
                frame_width=frame_width,
                frame_height=frame_height,
                images=images,
                source="Original BGR camera frame before drawing or preview mirroring",
                frame_index_base=0,
                time_origin="first recorded frame",
            )
            (diagnostic_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            meta["diagnostic_manifest"] = str(
                (diagnostic_dir / "manifest.json").relative_to(data_dir)
            )
        except BaseException:
            shutil.rmtree(diagnostic_dir)
            raise
    clips = Path(data_dir) / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    output = clips / (meta["clip_id"] + ".npz")
    temp = NamedTemporaryFile(dir=clips, suffix=".tmp", delete=False)
    temp_path = Path(temp.name)
    try:
        with temp:
            np.savez_compressed(temp, xyz=xyz, t_ms=t_ms, valid=valid, meta_json=json.dumps(meta))
        temp_path.replace(output)
    except BaseException:
        if diagnostic_dir is not None:
            shutil.rmtree(diagnostic_dir)
        raise
    finally:
        temp_path.unlink(missing_ok=True)
    try:
        rebuild_index(data_dir)
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(
            f"Clip saved to {output}; CSV update failed ({error}). "
            "After fixing the cause, run python -m fingerspelling.dataset to rebuild it."
        ) from error
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the local metadata.csv from its NPZ clips."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    directory = read_config(parser.parse_args().config)["data"]["directory"]
    print(f"Indexed {rebuild_index(directory)} clip(s) in {directory / 'metadata.csv'}")
