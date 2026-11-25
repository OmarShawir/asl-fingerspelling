"""Adapt the fixed v2 MLP using original Kaggle train + p01/s04; select only on s03."""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fingerspelling.adaptation import assert_disjoint, balanced_batch, camera_frame_weights
from fingerspelling.features import FEATURE_V2, LABELS, prepare_features, validate_dimensions
from fingerspelling.models.mlp import StaticMLP
from fingerspelling.training_data import dimensions_for_row, load_dimensions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_camera(session, labels):
    clips = []
    for path in sorted(Path("data/clips").glob("*.npz")):
        with np.load(path, allow_pickle=False) as c:
            meta = json.loads(c["meta_json"].item())
            if (meta["person"], meta["session"], meta["kind"], meta["source"]) != (
                "p01",
                session,
                "letter",
                "webcam",
            ):
                continue
            width, height = validate_dimensions(meta.get("frame_width"), meta.get("frame_height"))
            if meta["clip_id"] != path.stem:
                raise ValueError("Clip ID differs from filename")
            mask = c["valid"].astype(bool) & np.isfinite(c["xyz"]).all(axis=(1, 2))
            xyz = c["xyz"][mask]
            if not len(xyz):
                raise ValueError(f"No valid frames: {path}")
            x = prepare_features(
                xyz,
                meta["hand"],
                feature_version=FEATURE_V2,
                frame_width=width,
                frame_height=height,
            )
        clips.append(
            {
                "clip_id": path.stem,
                "target": meta["text"],
                "target_index": labels.index(meta["text"]),
                "valid_frames": len(x),
                "frames": meta["frames"],
                "frame_width": width,
                "frame_height": height,
                "hand": meta["hand"],
                "path": str(path),
                "sha256": sha(path),
                "landmark_sequence_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
                "x": torch.from_numpy(x),
            }
        )
    if not clips:
        raise ValueError(f"No {session} clips")
    return clips


def kaggle_tensors(entries, rows, dimensions, labels):
    xs = []
    ys = []
    for entry in entries:
        row = rows[entry["path"]]
        if row[2] != entry["sha256_pixels"] or row[1] != entry["label"]:
            raise ValueError("Original Kaggle split does not match cache")
        width, height = dimensions_for_row(row, dimensions)
        xs.append(
            prepare_features(
                np.frombuffer(row[3], dtype=np.float32).reshape(21, 3),
                row[4],
                feature_version=FEATURE_V2,
                frame_width=width,
                frame_height=height,
            )
        )
        ys.append(labels.index(row[1]))
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys, dtype=torch.long)


@torch.inference_mode()
def camera_score(model, clips, labels):
    model.eval()
    predictions = []
    for clip in clips:
        probs = torch.cat([model(x).softmax(-1) for x in clip["x"].split(2048)]).mean(0)
        order = torch.argsort(probs, descending=True, stable=True)[:3].tolist()
        predictions.append(
            {
                "clip_id": clip["clip_id"],
                "target": clip["target"],
                "prediction": labels[order[0]],
                "correct": order[0] == clip["target_index"],
                "valid_frames": clip["valid_frames"],
                "nll": -math.log(max(float(probs[clip["target_index"]]), 1e-12)),
                "top3": [{"letter": labels[i], "score": float(probs[i])} for i in order],
            }
        )
    per_letter = {}
    for label in labels:
        rows = [r for r in predictions if r["target"] == label]
        per_letter[label] = {
            "correct": sum(r["correct"] for r in rows),
            "total": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None,
        }
    observed = [label for label in labels if per_letter[label]["total"]]
    return {
        "correct": sum(r["correct"] for r in predictions),
        "total": len(predictions),
        "macro_accuracy": float(np.mean([per_letter[c]["accuracy"] for c in observed])),
        "macro_nll": float(
            np.mean(
                [np.mean([r["nll"] for r in predictions if r["target"] == c]) for c in observed]
            )
        ),
        "per_letter": per_letter,
        "clips": predictions,
    }


@torch.inference_mode()
def kaggle_score(model, x, y, labels):
    model.eval()
    logits = torch.cat([model(b) for b in x.split(2048)])
    predicted = logits.argmax(1)
    matrix = torch.bincount(y * len(labels) + predicted, minlength=len(labels) ** 2).reshape(
        len(labels), len(labels)
    )
    per_letter = {
        label: {"correct": int(matrix[i, i]), "total": int(matrix[i].sum())}
        for i, label in enumerate(labels)
    }
    return {
        "correct": int((predicted == y).sum()),
        "total": len(y),
        "accuracy": float((predicted == y).float().mean()),
        "loss": float(nn.functional.cross_entropy(logits, y)),
        "macro_accuracy": float(np.mean([v["correct"] / v["total"] for v in per_letter.values()])),
        "per_letter": per_letter,
        "confusion": matrix.tolist(),
    }


def selection_key(score):
    return (-score["macro_accuracy"], score["macro_nll"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/mlp_static_aspect_v2.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mlp_static_adapted.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--result-dir", type=Path, default=Path("results/adaptation"))
    args = parser.parse_args()
    if (args.result_dir / "report.json").exists():
        raise SystemExit("Refusing to overwrite experiment results; choose a new --result-dir")
    if args.output.exists():
        raise SystemExit("Refusing to overwrite existing adapted checkpoint")
    if min(args.epochs, args.steps_per_epoch, args.patience) < 1:
        raise ValueError("Training limits must be positive")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(42)
    bundle = torch.load(args.source, map_location="cpu", weights_only=True)
    if bundle["feature_version"] != FEATURE_V2:
        raise ValueError("Source must use features v2")
    labels = bundle["labels"]
    if len(labels) != 24 or set(labels) != set(LABELS):
        raise ValueError("Expected 24 static labels")
    source_sha = sha(args.source)
    split_path = Path("data/cache/static_split.json")
    split_sha = sha(split_path)
    if split_sha != bundle["split_sha256"]:
        raise ValueError("Original Kaggle split hash mismatch")
    split = json.loads(split_path.read_text())
    assert_disjoint([r["path"] for r in split["train"]], [r["path"] for r in split["validation"]])
    assert_disjoint(
        [r["sha256_pixels"] for r in split["train"]],
        [r["sha256_pixels"] for r in split["validation"]],
    )
    train_clips = load_camera("s04", labels)
    val_clips = load_camera("s03", labels)
    assert_disjoint([r["clip_id"] for r in train_clips], [r["clip_id"] for r in val_clips])
    assert_disjoint(
        [r["landmark_sequence_sha256"] for r in train_clips],
        [r["landmark_sequence_sha256"] for r in val_clips],
    )
    with sqlite3.connect("file:data/cache/kaggle_landmarks.sqlite?mode=ro", uri=True) as db:
        rows = {
            r[0]: r
            for r in db.execute(
                "SELECT path,label,digest,points,hand FROM images WHERE points IS NOT NULL"
            )
        }
    dimensions = load_dimensions()
    kx, ky = kaggle_tensors(split["train"], rows, dimensions, labels)
    vx, vy = kaggle_tensors(split["validation"], rows, dimensions, labels)
    cx = torch.cat([r["x"] for r in train_clips])
    cy = torch.tensor([r["target_index"] for r in train_clips for _ in range(r["valid_frames"])])
    frame_ids = [r["clip_id"] for r in train_clips for _ in range(r["valid_frames"])]
    cw = camera_frame_weights(cy.tolist(), frame_ids)
    counts = Counter(ky.tolist())
    kw = torch.tensor([1 / (len(labels) * counts[y]) for y in ky.tolist()], dtype=torch.float64)
    out = args.result_dir
    out.mkdir(parents=True, exist_ok=True)
    configuration = {
        "seed": 42,
        "feature_version": FEATURE_V2,
        "labels": labels,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "source_fraction_camera": 0.5,
        "max_epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "patience": args.patience,
        "augmentation": "none; train-time dropout 0.2 retained",
        "selection": "s03 macro clip accuracy over observed classes, tie-break minimum macro clip NLL; baseline epoch 0 eligible",
        "sampling": "128 Kaggle + 128 camera each batch, with replacement; Kaggle class uniform then image uniform; camera class uniform then clip uniform then valid frame uniform",
        "source_checkpoint": str(args.source),
        "source_checkpoint_sha256": source_sha,
        "kaggle_split_sha256": split_sha,
    }
    manifest = {
        "configuration": configuration,
        "kaggle_train_count": len(ky),
        "kaggle_validation_count": len(vy),
        "camera_train": [{k: v for k, v in c.items() if k != "x"} for c in train_clips],
        "camera_validation": [{k: v for k, v in c.items() if k != "x"} for c in val_clips],
        "clip_id_overlap": [],
        "landmark_sequence_overlap": [],
        "kaggle_pixel_hash_overlap": [],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    model = StaticMLP(len(labels))
    model.load_state_dict(bundle["state_dict"])
    baseline_camera = camera_score(model, val_clips, labels)
    baseline_kaggle = kaggle_score(model, vx, vy, labels)
    best_key = selection_key(baseline_camera)
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    history = []
    draws = Counter()
    kdraws = Counter()
    print(
        f"Data: Kaggle train {len(ky)}, val {len(vy)}; s04 {len(train_clips)} clips / {len(cy)} valid frames; s03 {len(val_clips)} clips",
        flush=True,
    )
    print(
        f"Baseline s03 {baseline_camera['correct']}/{baseline_camera['total']}, Kaggle {baseline_kaggle['accuracy']:.4%}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for _ in range(args.steps_per_epoch):
            ki, ci = balanced_batch(kw, cw, 128, generator)
            x = torch.cat((kx[ki], cx[ci]))
            y = torch.cat((ky[ki], cy[ci]))
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
            draws.update(frame_ids[i] for i in ci.tolist())
            kdraws.update(labels[i] for i in ky[ki].tolist())
        score = camera_score(model, val_clips, labels)
        history.append(
            {
                "epoch": epoch,
                "train_sampled_loss": loss_total / args.steps_per_epoch,
                "s03_correct": score["correct"],
                "s03_total": score["total"],
                "s03_macro_accuracy": score["macro_accuracy"],
                "s03_macro_nll": score["macro_nll"],
            }
        )
        print(
            f"Epoch {epoch}: s03 {score['correct']}/{score['total']}, macro {score['macro_accuracy']:.4%}, NLL {score['macro_nll']:.5f}",
            flush=True,
        )
        if selection_key(score) < best_key:
            best_key = selection_key(score)
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    adapted_camera = camera_score(model, val_clips, labels)
    adapted_kaggle = kaggle_score(model, vx, vy, labels)
    training_camera = camera_score(model, train_clips, labels)
    improvements = []
    regressions = []
    changes = []
    for before, after in zip(baseline_camera["clips"], adapted_camera["clips"]):
        if before["clip_id"] != after["clip_id"]:
            raise ValueError("Validation clip set changed")
        entry = {
            "clip_id": before["clip_id"],
            "target": before["target"],
            "before": before["prediction"],
            "after": after["prediction"],
            "before_top3": before["top3"],
            "after_top3": after["top3"],
        }
        if not before["correct"] and after["correct"]:
            improvements.append(entry)
        if before["correct"] and not after["correct"]:
            regressions.append(entry)
        if before["prediction"] != after["prediction"]:
            changes.append(entry)
    for c in train_clips + val_clips:
        if sha(c["path"]) != c["sha256"]:
            raise ValueError("Camera data changed during experiment")
    if sha(args.source) != source_sha:
        raise ValueError("Source checkpoint changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "feature_version": FEATURE_V2,
            "epoch": best_epoch,
            "seed": 42,
            "split_sha256": split_sha,
            "adaptation": configuration,
            "adaptation_manifest_sha256": sha(out / "manifest.json"),
        },
        args.output,
    )
    report = {
        "configuration": configuration,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_seconds": time.monotonic() - started,
        "torch_version": str(torch.__version__),
        "baseline_s03": baseline_camera,
        "adapted_s03": adapted_camera,
        "adapted_s04_training": training_camera,
        "baseline_kaggle_validation": baseline_kaggle,
        "adapted_kaggle_validation": adapted_kaggle,
        "improved_clips": improvements,
        "regressed_clips": regressions,
        "changed_predictions": changes,
        "history": history,
        "actual_camera_draws_by_clip": dict(draws),
        "actual_kaggle_draws_by_class": dict(kdraws),
        "output_checkpoint": str(args.output),
        "output_sha256": sha(args.output),
        "interpretation": "s03 is validation used for selection, not an untouched test; only one signer. Kaggle validation monitored before/after, not used for selection.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "s03_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "clip_id",
                "target",
                "valid_frames",
                "baseline",
                "adapted",
                "baseline_correct",
                "adapted_correct",
                "change",
            ]
        )
        for before, after in zip(baseline_camera["clips"], adapted_camera["clips"]):
            change = (
                "improved"
                if not before["correct"] and after["correct"]
                else "regressed"
                if before["correct"] and not after["correct"]
                else "unchanged_outcome"
            )
            writer.writerow(
                [
                    before["clip_id"],
                    before["target"],
                    before["valid_frames"],
                    before["prediction"],
                    after["prediction"],
                    before["correct"],
                    after["correct"],
                    change,
                ]
            )
    with (out / "per_letter.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "letter",
                "s03_baseline_correct",
                "s03_adapted_correct",
                "s03_total",
                "kaggle_baseline_correct",
                "kaggle_adapted_correct",
                "kaggle_total",
            ]
        )
        for label in labels:
            a = baseline_camera["per_letter"][label]
            b = adapted_camera["per_letter"][label]
            c = baseline_kaggle["per_letter"][label]
            d = adapted_kaggle["per_letter"][label]
            writer.writerow(
                [
                    label,
                    a["correct"],
                    b["correct"],
                    a["total"],
                    c["correct"],
                    d["correct"],
                    c["total"],
                ]
            )
    print(
        f"Selected epoch {best_epoch}; s03 {adapted_camera['correct']}/{adapted_camera['total']}; Kaggle {adapted_kaggle['accuracy']:.4%}; improved {len(improvements)}, regressed {len(regressions)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
