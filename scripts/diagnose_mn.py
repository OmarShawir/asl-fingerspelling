"""Read-only M/N baseline diagnostics; exploratory variants never change model/config."""

import csv
import hashlib
import importlib.util
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fingerspelling.config import read_config
from fingerspelling.evaluation import rank_clip
from fingerspelling.features import prepare_features
from fingerspelling.models.mlp import StaticPredictor

OUT = Path("results/mn_diagnostic_2026-11-07")
ROOT = Path("data/external/asl-alphabet")
EDGES = [(e.start, e.end) for e in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def draw(ax, xyz, image=None, title=""):
    if image is not None:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), extent=(0, 1, 1, 0))
    if xyz is not None:
        for a, b in EDGES:
            ax.plot(xyz[[a, b], 0], xyz[[a, b], 1], color="cyan", linewidth=1.1)
        for i, p in enumerate(xyz):
            color = (
                "red"
                if i == 0
                else "#ffb000"
                if i in range(1, 5)
                else "magenta"
                if i in range(13, 17)
                else "white"
            )
            ax.scatter(p[0], p[1], c=color, s=18, edgecolors="black", linewidths=0.3)
            ax.annotate(
                str(i),
                (p[0], p[1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, pad=0.3, edgecolor="none"),
            )
    else:
        ax.text(
            0.5,
            0.08,
            "NO LANDMARKS: no MLP prediction",
            transform=ax.transAxes,
            ha="center",
            color="red",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.9),
        )
    ax.set_title(title, fontsize=10)
    if image is not None:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def metrics(xyz, hand):
    f = prepare_features(xyz, hand).reshape(-1, 21, 3)
    ring_path = sum(
        np.linalg.norm(f[:, b] - f[:, a], axis=1) for a, b in [(13, 14), (14, 15), (15, 16)]
    )
    return {
        "ring_extension_3d": np.linalg.norm(f[:, 16] - f[:, 13], axis=1)
        / np.maximum(ring_path, 1e-8),
        "thumb_to_ring_mcp": np.linalg.norm(f[:, 4] - f[:, 13], axis=1),
        "thumb_to_ring_tip": np.linalg.norm(f[:, 4] - f[:, 16], axis=1),
        "ring_tip_to_wrist": np.linalg.norm(f[:, 16], axis=1),
        "ring_tip_relative_mcp_y": f[:, 16, 1] - f[:, 13, 1],
        "middle_mcp_relative_wrist_y": f[:, 9, 1],
    }


def describe(values):
    a = np.asarray(values)
    return {
        "n": len(a),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.median(a)),
        "q75": float(np.quantile(a, 0.75)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [
        Path("models/mlp_static.pt"),
        Path("configs/default.yaml"),
        Path("src/fingerspelling/features.py"),
        *Path("data/clips").glob("*.npz"),
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    predictor = StaticPredictor("models/mlp_static.pt")
    spec = importlib.util.spec_from_file_location(
        "kaggle_extractor", Path(__file__).with_name("extract_kaggle.py")
    )
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)
    settings = read_config()["tracker"]
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(settings["model_path"])),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=settings["detection_confidence"],
        min_hand_presence_confidence=settings["presence_confidence"],
    )
    extraction.init_worker(options)
    image_rows = []
    test_panels = []
    training_panels = []
    rng = np.random.default_rng(20261107)
    with sqlite3.connect("file:data/cache/kaggle_landmarks.sqlite?mode=ro", uri=True) as db:
        cached = db.execute(
            "SELECT path,label,points,hand FROM images WHERE label IN ('M','N') AND points IS NOT NULL ORDER BY path"
        ).fetchall()
    selections = []
    for label in ("M", "N"):
        group = [r for r in cached if r[1] == label]
        selections.extend(group[i] for i in rng.choice(len(group), 6, replace=False))
    paths = [("test", x, next(ROOT.rglob(x + "_test.jpg")), None) for x in ("M", "N")]
    paths += [
        ("train_sample", r[1], ROOT / "asl_alphabet_train/asl_alphabet_train" / r[0], r)
        for r in selections
    ]
    for kind, label, path, cached_row in paths:
        _, blob, hand, hand_score = extraction.detect(path)
        xyz = None if blob is None else np.frombuffer(blob, dtype=np.float32).reshape(21, 3)
        rgb = cv2.imread(str(path))
        top = [] if xyz is None else rank_clip(predictor.probabilities(xyz, hand), predictor.labels)
        row = {
            "kind": kind,
            "path": str(path),
            "target": label,
            "prediction": top[0]["letter"] if top else None,
            "top3": top,
            "handedness_used": hand or None,
            "handedness_score": hand_score if hand else None,
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "raw_landmarks": None if xyz is None else xyz.tolist(),
        }
        if cached_row is not None:
            row["cache_max_abs_difference"] = (
                None
                if xyz is None
                else float(
                    np.abs(
                        xyz - np.frombuffer(cached_row[2], dtype=np.float32).reshape(21, 3)
                    ).max()
                )
            )
        image_rows.append(row)
        title = f"{path.name} | true {label} -> {row['prediction'] or 'NO HAND'} | hand {hand or 'unknown'}"
        fig, ax = plt.subplots(figsize=(7, 7))
        draw(ax, xyz, rgb, title)
        fig.tight_layout()
        fig.savefig(OUT / (path.stem + "_landmarks.png"), dpi=160)
        plt.close(fig)
        (test_panels if kind == "test" else training_panels).append((xyz, rgb, title))
    # Separate, explicit sensitivity trials. They do not replace the baseline's failures.
    variants = []
    for threshold in (0.5, 0.25, 0.1):
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(settings["model_path"])),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=threshold,
            min_hand_presence_confidence=threshold,
        )
        tracker = vision.HandLandmarker.create_from_options(opts)
        extraction.LOCAL.tracker = tracker
        for label in ("M", "N"):
            source = next(ROOT.rglob(label + "_test.jpg"))
            for scale in (1, 2, 4):
                # Call the very same extraction function on a temporary, lossless resized image.
                image = cv2.resize(cv2.imread(str(source)), None, fx=scale, fy=scale)
                temp = OUT / "resized_probe.png"
                cv2.imwrite(str(temp), image)
                _, blob, hand, hand_score = extraction.detect(temp)
                xyz = None if blob is None else np.frombuffer(blob, dtype=np.float32).reshape(21, 3)
                top = (
                    []
                    if xyz is None
                    else rank_clip(predictor.probabilities(xyz, hand), predictor.labels)
                )
                variants.append(
                    {
                        "target": label,
                        "thresholds": threshold,
                        "scale": scale,
                        "handedness_used": hand or None,
                        "handedness_score": hand_score,
                        "top3": top,
                        "detected": xyz is not None,
                    }
                )
                if threshold == 0.25 and scale == 1 and xyz is not None:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    draw(
                        ax,
                        xyz,
                        image,
                        f"{label}_test | DIAGNOSTIC thresholds 0.25 | {hand} | {top[0]['letter']} {top[0]['score']:.2%}",
                    )
                    fig.tight_layout()
                    fig.savefig(OUT / (label + "_test_relaxed_landmarks.png"), dpi=160)
                    plt.close(fig)
        tracker.close()
    (OUT / "resized_probe.png").unlink()
    for tracker in extraction.TRACKERS:
        tracker.close()
    for name, panels, cols in [
        ("test_examples", test_panels, 2),
        ("training_examples", training_panels, 3),
    ]:
        fig, axes = plt.subplots(
            (len(panels) + cols - 1) // cols,
            cols,
            figsize=(cols * 5, ((len(panels) + cols - 1) // cols) * 5),
            squeeze=False,
        )
        for ax, (xyz, img, title) in zip(axes.flat, panels):
            draw(ax, xyz, img, title)
        fig.tight_layout()
        fig.savefig(OUT / (name + ".png"), dpi=150)
        plt.close(fig)
    clip_rows = []
    clip_features = {"M": [], "N": []}
    clip_panels = []
    for path in sorted(Path("data/clips").glob("*.npz")):
        with np.load(path, allow_pickle=False) as c:
            meta = json.loads(c["meta_json"].item())
            local = datetime.fromisoformat(meta["created_at"]).astimezone(
                ZoneInfo("Asia/Kuala_Lumpur")
            )
            if (meta["person"], meta["session"], meta["kind"], meta["source"]) != (
                "p01",
                "s02",
                "letter",
                "webcam",
            ) or local.date().isoformat() != "2026-11-07":
                continue
            mask = c["valid"].astype(bool) & np.isfinite(c["xyz"]).all(axis=(1, 2))
            xyz = c["xyz"][mask]
            indices = np.flatnonzero(mask)
            times = c["t_ms"][mask]
        top = (
            rank_clip(predictor.probabilities(xyz, meta["hand"]), predictor.labels)
            if len(xyz)
            else []
        )
        row = {
            "clip_id": path.stem,
            "target": meta["text"],
            "prediction": top[0]["letter"] if top else None,
            "handedness_used": meta["hand"],
            "top3": top,
            "frames": meta["frames"],
            "valid_frames": len(xyz),
            "created_at_local": local.isoformat(),
        }
        if meta["text"] in ("M", "N") and len(xyz):
            features = prepare_features(xyz, meta["hand"]).reshape(-1, 21, 3)
            clip_features[meta["text"]].append(features)
            row["geometry_medians"] = {
                k: float(np.median(v)) for k, v in metrics(xyz, meta["hand"]).items()
            }
            # Choose a real frame nearest this clip's median feature vector, never synthesize a hand.
            representative = int(
                np.argmin(
                    np.linalg.norm(
                        features.reshape(-1, 63) - np.median(features.reshape(-1, 63), axis=0),
                        axis=1,
                    )
                )
            )
            row["representative_frame"] = int(indices[representative])
            row["representative_t_ms"] = int(times[representative])
            row["representative_raw_landmarks"] = xyz[representative].tolist()
            row["aspect_sensitivity_assumed_height_over_width"] = {}
            for ratio in (1.0, 0.75, 0.5625):
                alternative = xyz.copy()
                alternative[:, :, 1] *= ratio
                row["aspect_sensitivity_assumed_height_over_width"][str(ratio)] = rank_clip(
                    predictor.probabilities(alternative, meta["hand"]), predictor.labels
                )
            opposite = "right" if meta["hand"] == "left" else "left"
            row["wrong_hand_counterfactual"] = rank_clip(
                predictor.probabilities(xyz, opposite), predictor.labels
            )
            clip_panels.append(
                (
                    features[representative],
                    None,
                    f"{path.stem[:8]} | {meta['text']} -> {row['prediction']}\nframe {indices[representative]}, t={times[representative]}ms | canonical x",
                )
            )
        clip_rows.append(row)
    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    for ax, (xyz, img, title) in zip(axes.flat, clip_panels):
        draw(ax, xyz, img, title)
    for ax in list(axes.flat)[len(clip_panels) :]:
        ax.axis("off")
    fig.suptitle(
        "p01/s02 2026-11-07: actual representative landmarks, no RGB frames saved", fontsize=16
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "camera_landmarks.png", dpi=150)
    plt.close(fig)
    geometry = {}
    train_features = {}
    for label in ("M", "N"):
        rows = [r for r in cached if r[1] == label]
        features = np.stack(
            [
                prepare_features(
                    np.frombuffer(r[2], dtype=np.float32).reshape(21, 3), r[3]
                ).reshape(21, 3)
                for r in rows
            ]
        )
        train_features[label] = features
        # Features are already centered/scaled/canonical; right preserves them under normalization.
        geometry["kaggle_" + label] = {
            k: describe(v) for k, v in metrics(features, "right").items()
        }
        target_clips = [r for r in clip_rows if r["target"] == label]
        geometry["camera_" + label] = {
            k: describe([r["geometry_medians"][k] for r in target_clips])
            for k in target_clips[0]["geometry_medians"]
        }
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, label in zip(axes[0], ("M", "N")):
        draw(
            ax,
            np.median(train_features[label], axis=0),
            title=f"Kaggle {label}: coordinate median (summary, not a real hand)",
        )
    for ax, label in zip(axes[1], ("M", "N")):
        features = np.stack([np.median(f, axis=0) for f in clip_features[label]])
        draw(
            ax,
            np.median(features, axis=0),
            title=f"Camera {label}: median of clip medians (summary)",
        )
    fig.tight_layout()
    fig.savefig(OUT / "geometry_summary.png", dpi=160)
    plt.close(fig)
    # Isolate aspect dependence using one planar geometry encoded on two canvas shapes.
    pixel = np.column_stack([np.arange(21) * 3, np.arange(21) * 3, np.zeros(21)]).astype(np.float32)
    square = pixel / np.array([200, 200, 200], dtype=np.float32)
    wide = pixel / np.array([400, 200, 400], dtype=np.float32)
    aspect_demo = {
        "same_pixel_geometry_feature_max_abs_difference": float(
            np.abs(prepare_features(square, "right") - prepare_features(wide, "right")).max()
        ),
        "canvas_sizes": [[200, 200], [400, 200]],
        "z_zero": True,
        "interpretation": "Changing canvas aspect changes the features even with identical pixel geometry.",
    }
    report = {
        "date": "2026-11-07",
        "timezone": "Asia/Kuala_Lumpur",
        "checkpoint_sha256": before["models/mlp_static.pt"],
        "sample_seed": 20261107,
        "sample_policy": "six uniformly sampled successful cached detections per class; conditional sample, not coverage estimate",
        "baseline_images": image_rows,
        "test_sensitivity": variants,
        "clips_on_date": clip_rows,
        "geometry": geometry,
        "aspect_demo": aspect_demo,
        "camera_dimensions": "Not saved in NPZ metadata or CSV; historical resolution unknown.",
        "camera_raw_images": "Not saved; wrist/finger visual correctness cannot be verified against RGB.",
        "class_labels": predictor.labels,
    }
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
    report["baseline_config_features_recordings_unchanged"] = True
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "clip_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip_id",
                "target",
                "prediction",
                "hand",
                "valid_frames",
                "top1",
                "score1",
                "top2",
                "score2",
                "top3",
                "score3",
            ]
        )
        for r in clip_rows:
            writer.writerow(
                [
                    r["clip_id"],
                    r["target"],
                    r["prediction"],
                    r["handedness_used"],
                    r["valid_frames"],
                    *[v for p in r["top3"] for v in (p["letter"], p["score"])],
                ]
            )
    print(
        "Images:",
        [(r["path"].split("/")[-1], r["prediction"], r["handedness_used"]) for r in image_rows],
    )
    print(
        "Dated clips:", len(clip_rows), Counter((r["target"], r["prediction"]) for r in clip_rows)
    )
    print("Geometry:", json.dumps(geometry, indent=2))
    print("Aspect:", aspect_demo)


if __name__ == "__main__":
    main()
