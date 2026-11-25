"""Audit local s05 reports and render saved-frame evidence without re-extraction or fitting."""

import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mediapipe.tasks.python import vision

OUT = Path("results/s05_diagnostic")
FOCUS = list("KVPQWODTN")
EDGES = [(e.start, e.end) for e in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_rows(rows):
    result = {}
    for row in rows:
        if row["clip_id"] in result:
            raise ValueError("Duplicate clip_id: " + row["clip_id"])
        result[row["clip_id"]] = row
    return result


def join_clips(before, after):
    a = unique_rows(before)
    b = unique_rows(after)
    if a.keys() != b.keys():
        raise ValueError("Clip sets differ")
    joined = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["target"] != y["target"]:
            raise ValueError("Targets differ: " + key)
        good_a = x["prediction"] == x["target"]
        good_b = y["prediction"] == y["target"]
        category = (
            "both_correct"
            if good_a and good_b
            else "improved"
            if good_b
            else "regressed"
            if good_a
            else "both_wrong"
        )
        joined.append(
            dict(
                clip_id=key,
                target=x["target"],
                v2=x["prediction"],
                adapted=y["prediction"],
                v2_correct=good_a,
                adapted_correct=good_b,
                category=category,
                valid_frames=x["valid_frames"],
                v2_top3=x["top3"],
                adapted_top3=y["top3"],
            )
        )
    return joined


def validate_frame(entry, times):
    index = entry["frame_index"]
    if not isinstance(index, int) or not 0 <= index < len(times):
        raise ValueError("Invalid frame index")
    if entry["t_ms"] != int(times[index]):
        raise ValueError("Frame timestamp mismatch")
    return index


def read_report(stem):
    report = json.loads(Path("results", stem + ".json").read_text())
    rows = unique_rows(report["clips"])
    with Path("results", stem + ".csv").open(newline="") as stream:
        table = unique_rows(list(csv.DictReader(stream)))
    if rows.keys() != table.keys():
        raise ValueError("CSV/JSON clip set mismatch")
    for key, row in rows.items():
        tab = table[key]
        for field in ("target", "prediction"):
            if row[field] != tab[field]:
                raise ValueError(f"CSV/JSON {field} mismatch")
        if int(tab["valid_frames"]) != row["valid_frames"]:
            raise ValueError("Valid-frame count mismatch")
        if row["correct"] != (row["target"] == row["prediction"]):
            raise ValueError("Incorrect stored correctness flag")
        for i, rank in enumerate(row["top3"], 1):
            if rank["letter"] != tab[f"top{i}_letter"] or not np.isclose(
                rank["score"], float(tab[f"top{i}_score"]), atol=1e-9, rtol=0
            ):
                raise ValueError("CSV/JSON ranking mismatch")
    if report["total"] != len(rows) or report["correct"] != sum(
        r["correct"] for r in rows.values()
    ):
        raise ValueError("Report totals disagree with rows")
    if report["valid_frames"] != sum(r["valid_frames"] for r in rows.values()) or report[
        "frames"
    ] != sum(r["frames"] for r in rows.values()):
        raise ValueError("Report frame totals disagree")
    return report, rows


def panel(ax, image, points, title):
    h, w = image.shape[:2]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if points is not None:
        xy = points[:, :2] * [w, h]
        for a, b in EDGES:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="cyan", linewidth=1.3)
        for i, (x, y) in enumerate(xy):
            color = "red" if i == 0 else "gold" if i < 5 else "magenta" if 13 <= i <= 16 else "lime"
            ax.scatter(x, y, c=color, s=18, edgecolors="black", linewidths=0.4)
            ax.annotate(
                str(i),
                (x, y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3),
            )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def camera_evidence(row, kind, limit, protected):
    clip_path = Path("data/clips") / (row["clip_id"] + ".npz")
    with np.load(clip_path, allow_pickle=False) as c:
        meta = json.loads(c["meta_json"].item())
        times = c["t_ms"].copy()
        xyz = c["xyz"].copy()
        valid = c["valid"].copy()
    record = dict(
        clip_id=row["clip_id"],
        target=row["target"],
        v2=row["v2"],
        adapted=row["adapted"],
        kind=kind,
        missing_files=[],
        selected=[],
        issues=[],
        available_images=0,
        manifest_entries=0,
    )
    relative = meta.get("diagnostic_manifest")
    if not relative:
        record["issues"].append("No diagnostic_manifest in clip metadata")
        return record
    path = Path("data") / relative
    if not path.is_file():
        record["missing_files"].append(str(path))
        return record
    protected[str(path)] = sha(path)
    manifest = json.loads(path.read_text())
    record["manifest"] = str(path)
    if manifest["clip_id"] != row["clip_id"]:
        raise ValueError("Manifest clip_id mismatch")
    if (manifest["frame_width"], manifest["frame_height"]) != (
        meta["frame_width"],
        meta["frame_height"],
    ):
        raise ValueError("Manifest dimensions mismatch")
    candidates = []
    seen = set()
    offsets = set()
    record["manifest_entries"] = len(manifest["images"])
    for entry in manifest["images"]:
        index = validate_frame(entry, times)
        if index in seen:
            raise ValueError("Duplicate diagnostic frame index")
        seen.add(index)
        offsets.add(entry["capture_timestamp_ms"] - entry["t_ms"])
        image_path = path.parent / entry["file"]
        if not image_path.is_file():
            record["missing_files"].append(str(image_path))
            continue
        candidates.append((index, image_path, entry))
    if len(offsets) > 1:
        raise ValueError("Capture/relative timestamp origin inconsistent")
    candidates.sort()
    record["available_images"] = len(candidates)
    if not candidates:
        record["issues"].append("No available diagnostic images")
        return record
    selected = (
        [len(candidates) // 2]
        if limit == 1
        else np.unique(
            np.linspace(0, len(candidates) - 1, min(limit, len(candidates))).round().astype(int)
        )
    )
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 4.9 * len(selected)), squeeze=False)
    fig.suptitle(
        f"{row['clip_id']} | true {row['target']} | v2 {row['v2']} | adapted {row['adapted']}\nPredictions are clip-level mean probabilities; images are original, unmirrored",
        fontsize=11,
    )
    for pair, position in zip(axes, selected):
        index, image_path, entry = candidates[position]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Unreadable image " + str(image_path))
        if image.shape[:2] != (meta["frame_height"], meta["frame_width"]):
            raise ValueError("PNG dimensions mismatch")
        protected[str(image_path)] = sha(image_path)
        original_copy = OUT / "originals" / row["clip_id"] / image_path.name
        original_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, original_copy)
        points = xyz[index] if valid[index] and np.isfinite(xyz[index]).all() else None
        caption = f"frame {index}, t={int(times[index])}ms, {meta['hand']} hand, {image.shape[1]}x{image.shape[0]}"
        panel(pair[0], image, None, "Original | " + caption)
        panel(pair[1], image, points, "Saved landmarks (no new extraction) | " + caption)
        record["selected"].append(
            dict(
                frame_index=index,
                t_ms=int(times[index]),
                source=str(image_path),
                source_sha256=protected[str(image_path)],
                valid=bool(valid[index]),
                original_copy=str(original_copy.relative_to(OUT)),
            )
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    target = OUT / kind / (row["clip_id"] + ".png")
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    record["comparison_image"] = str(target.relative_to(OUT))
    return record


def kaggle_examples(protected):
    rng = np.random.default_rng(20261115)
    records = []
    with sqlite3.connect("file:data/cache/kaggle_landmarks.sqlite?mode=ro", uri=True) as db:
        root = Path(json.loads(db.execute("SELECT signature FROM settings").fetchone()[0])["root"])
        for label in FOCUS:
            rows = db.execute(
                "SELECT path,digest,points,hand FROM images WHERE label=? AND points IS NOT NULL ORDER BY path",
                (label,),
            ).fetchall()
            selected = [rows[i] for i in rng.choice(len(rows), min(2, len(rows)), replace=False)]
            fig, axes = plt.subplots(
                len(selected), 2, figsize=(11, 5.5 * len(selected)), squeeze=False
            )
            for pair, (relative, digest, blob, hand) in zip(axes, selected):
                path = root / relative
                image = cv2.imread(str(path))
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest() != digest:
                    raise ValueError("Kaggle image does not match cached landmarks")
                protected[str(path)] = sha(path)
                points = np.frombuffer(blob, dtype=np.float32).reshape(21, 3)
                title = f"{relative} | dataset label {label} | predicted hand {hand}"
                panel(pair[0], image, None, "Original | " + title)
                panel(pair[1], image, points, "Cached landmarks | " + title)
                records.append(
                    dict(
                        label=label,
                        path=str(path),
                        handedness=hand,
                        pixel_sha256=digest,
                        gallery="kaggle/" + label + ".png",
                    )
                )
            target = OUT / "kaggle" / (label + ".png")
            target.parent.mkdir(exist_ok=True)
            fig.tight_layout()
            fig.savefig(target, dpi=140)
            plt.close(fig)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, a = read_report("s05_v2")
    adapted, b = read_report("s05_adapted")
    if baseline["labels"] != adapted["labels"] or len(baseline["labels"]) != 24:
        raise ValueError("24-label order mismatch")
    if (baseline["person"], baseline["session"]) != ("p01", "s05") or (
        adapted["person"],
        adapted["session"],
    ) != ("p01", "s05"):
        raise ValueError("Unexpected session")
    protected = {
        str(p): sha(p)
        for p in [
            Path("results/s05_v2.csv"),
            Path("results/s05_v2.json"),
            Path("results/s05_adapted.csv"),
            Path("results/s05_adapted.json"),
            Path("models/mlp_static_aspect_v2.pt"),
            Path("models/mlp_static_adapted.pt"),
            Path("data/metadata.csv"),
        ]
    }
    if (
        protected["models/mlp_static_aspect_v2.pt"] != baseline["checkpoint_sha256"]
        or protected["models/mlp_static_adapted.pt"] != adapted["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint hash mismatch")
    joined = join_clips(baseline["clips"], adapted["clips"])
    for row in joined:
        path = Path("data/clips") / (row["clip_id"] + ".npz")
        protected[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as c:
            meta = json.loads(c["meta_json"].item())
            count = int((c["valid"].astype(bool) & np.isfinite(c["xyz"]).all(axis=(1, 2))).sum())
            if (
                meta["clip_id"] != row["clip_id"]
                or meta["text"] != row["target"]
                or (meta["person"], meta["session"], meta["kind"]) != ("p01", "s05", "letter")
            ):
                raise ValueError("NPZ metadata does not match reports")
            for report_row in (a[row["clip_id"]], b[row["clip_id"]]):
                if count != report_row["valid_frames"] or len(c["xyz"]) != report_row["frames"]:
                    raise ValueError("NPZ/report frame mismatch")
                if (meta["frame_width"], meta["frame_height"], meta["hand"]) != (
                    report_row["frame_width"],
                    report_row["frame_height"],
                    report_row["hand"],
                ):
                    raise ValueError("NPZ/report dimension or hand mismatch")
    categories = {
        key: [r["clip_id"] for r in joined if r["category"] == key]
        for key in ("both_correct", "improved", "regressed", "both_wrong")
    }
    counts = {k: len(v) for k, v in categories.items()}
    per_letter = {}
    for label in baseline["labels"]:
        rows = [r for r in joined if r["target"] == label]
        per_letter[label] = {
            "total": len(rows),
            "v2_correct": sum(r["v2_correct"] for r in rows),
            "adapted_correct": sum(r["adapted_correct"] for r in rows),
        }
    fields = [
        "clip_id",
        "target",
        "v2",
        "adapted",
        "v2_correct",
        "adapted_correct",
        "category",
        "valid_frames",
    ]
    for name, rows in [
        ("comparison", joined),
        *[(k, [r for r in joined if r["category"] == k]) for k in categories],
    ]:
        with (OUT / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with (OUT / "per_letter.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["letter", "total", "v2_correct", "adapted_correct"])
        writer.writerows(
            [label, r["total"], r["v2_correct"], r["adapted_correct"]]
            for label, r in per_letter.items()
        )
    errors = [r for r in joined if not r["adapted_correct"]]
    print("Counts:", counts, "adapted errors", len(errors), flush=True)
    error_evidence = [camera_evidence(r, "errors", 3, protected) for r in errors]
    controls = []
    missing_controls = []
    for label in FOCUS:
        candidates = [r for r in joined if r["target"] == label and r["category"] == "both_correct"]
        if candidates:
            controls.append(camera_evidence(candidates[0], "controls", 1, protected))
        else:
            missing_controls.append(label)
    kaggle = kaggle_examples(protected)
    manifest = json.loads(Path("results/adaptation/manifest.json").read_text())
    prior_ids = {c["clip_id"] for k in ("camera_train", "camera_validation") for c in manifest[k]}
    if set(a) & prior_ids:
        raise ValueError("s05 overlaps fine-tuning train/selection IDs")
    report = {
        "source": "local reports, saved NPZ/PNG and local Kaggle cache only; no internet, extraction rerun or model changes",
        "total": len(joined),
        "v2_correct": sum(r["v2_correct"] for r in joined),
        "adapted_correct": sum(r["adapted_correct"] for r in joined),
        "valid_frames": baseline["valid_frames"],
        "counts": counts,
        "categories": categories,
        "per_letter": per_letter,
        "comparisons": joined,
        "errors": error_evidence,
        "correct_controls": controls,
        "missing_correct_controls": missing_controls,
        "kaggle_examples": kaggle,
        "kaggle_seed": 20261115,
        "selection_policy": "errors: up to 3 evenly spaced available manifest images; controls: first both-correct clip per label sorted by ID, middle available image; Kaggle: 2 random successful-cache images per focus class",
        "selection_overlap_with_s03_s04": [],
        "baseline_feature_version": baseline["feature_version"],
        "adapted_feature_version": adapted["feature_version"],
        "inputs_sha256": protected,
    }
    if not all(sha(path) == digest for path, digest in protected.items()):
        raise ValueError("Protected input changed during diagnosis")
    report["inputs_unchanged"] = True
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Errors: saved images",
        sum(len(e["selected"]) for e in error_evidence),
        "missing",
        sum(len(e["missing_files"]) for e in error_evidence),
        flush=True,
    )


if __name__ == "__main__":
    main()
