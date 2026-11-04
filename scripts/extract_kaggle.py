"""Cache IMAGE-mode landmarks; pilot first, then all 24 static letters."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fingerspelling.config import read_config
from fingerspelling.hand_tracker import ensure_tracker
from fingerspelling.features import LABELS

LOCAL = threading.local()
TRACKERS = []


def init_worker(options):
    LOCAL.tracker = vision.HandLandmarker.create_from_options(options)
    TRACKERS.append(LOCAL.tracker)


def detect(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Unreadable image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    digest = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
    result = LOCAL.tracker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    points, hand, score = None, "", 0.0
    if result.hand_landmarks:
        xyz = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float32)
        if np.isfinite(xyz).all():
            points = xyz.tobytes()
            hand = result.handedness[0][0].category_name.lower()
            score = float(result.handedness[0][0].score)
    return digest, points, hand, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    results = Path("results")
    results.mkdir(exist_ok=True)
    settings = read_config()["tracker"]
    model = ensure_tracker(settings["model_path"])
    signature = json.dumps(
        {
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "mediapipe": mp.__version__,
            "mode": "IMAGE",
            "detection": settings["detection_confidence"],
            "presence": settings["presence_confidence"],
            "root": str(args.root.resolve()),
        },
        sort_keys=True,
    )
    db = sqlite3.connect(cache / "kaggle_landmarks.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS settings (signature TEXT)")
    prior = db.execute("SELECT signature FROM settings").fetchone()
    if prior and prior[0] != signature:
        raise SystemExit("Cache settings differ; use a fresh cache directory.")
    if not prior:
        db.execute("INSERT INTO settings VALUES (?)", (signature,))
    db.execute(
        "CREATE TABLE IF NOT EXISTS images "
        "(path TEXT PRIMARY KEY, label TEXT, digest TEXT, points BLOB, hand TEXT, score REAL)"
    )
    db.commit()
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=settings["detection_confidence"],
        min_hand_presence_confidence=settings["presence_confidence"],
    )
    rng = np.random.default_rng(args.seed)
    report = {}
    manifest = []
    with ThreadPoolExecutor(
        max_workers=args.workers, initializer=init_worker, initargs=(options,)
    ) as pool:
        for label in LABELS:
            paths = sorted((args.root / label).glob("*.jpg"))
            if len(paths) < 50:
                raise ValueError(f"{label}: only {len(paths)} images")
            if args.pilot:
                paths = [paths[i] for i in rng.choice(len(paths), 50, replace=False)]
            success = 0
            hands = Counter()
            cached = dict(
                db.execute("SELECT path, points IS NOT NULL FROM images WHERE label=?", (label,))
            )
            pending = [p for p in paths if str(p.relative_to(args.root)) not in cached]
            for i, (path, result) in enumerate(zip(pending, pool.map(detect, pending))):
                relative = str(path.relative_to(args.root))
                digest, points, hand, score = result
                db.execute(
                    "INSERT INTO images VALUES (?,?,?,?,?,?)",
                    (relative, label, digest, points, hand, score),
                )
                if (i + 1) % 250 == 0:
                    db.commit()
                    print(f"{label}: {i + 1}/{len(pending)} new images", flush=True)
            for path in paths:
                relative = str(path.relative_to(args.root))
                manifest.append(relative)
                row = db.execute(
                    "SELECT points,hand FROM images WHERE path=?", (relative,)
                ).fetchone()
                if row[0] is not None:
                    success += 1
                    hands[row[1]] += 1
            db.commit()
            report[label] = {
                "attempted": len(paths),
                "detected": success,
                "rate": success / len(paths),
                "hands": dict(hands),
            }
            print(
                f"{label}: {success}/{len(paths)} ({100 * success / len(paths):.1f}%) {dict(hands)}",
                flush=True,
            )
    for tracker in TRACKERS:
        tracker.close()
    name = "pilot" if args.pilot else "full"
    (results / f"kaggle_{name}_tracking.json").write_text(
        json.dumps(
            {"seed": args.seed, "settings": json.loads(signature), "letters": report}, indent=2
        )
        + "\n"
    )
    (cache / f"{name}_manifest.json").write_text(json.dumps(manifest))
    db.close()


if __name__ == "__main__":
    main()
