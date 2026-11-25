"""Train the static MLP from cached Kaggle landmarks, never webcam recordings."""

import argparse
import hashlib
import json
import sqlite3
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fingerspelling.features import FEATURE_V1, FEATURE_V2, LABELS, prepare_features
from fingerspelling.training_data import load_dimensions, dimensions_for_row
from fingerspelling.models.mlp import StaticMLP


def build_split(rows, seed=42):
    """Deduplicate decoded image pixels before a stratified split; drop label conflicts."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)
    rng = np.random.default_rng(seed)
    train, val, conflicts = [], [], []
    for digest, group in groups.items():
        if len({r[1] for r in group}) != 1:
            conflicts.append(digest)
    conflict_set = set(conflicts)
    for label in LABELS:
        unique = [
            group[0]
            for digest, group in sorted(groups.items())
            if digest not in conflict_set and group[0][1] == label
        ]
        if len(unique) < 5:
            raise ValueError(f"Insufficient detected unique images for {label}")
        order = rng.permutation(len(unique))
        nval = max(1, round(len(unique) * 0.2))
        val.extend(unique[i] for i in order[:nval])
        train.extend(unique[i] for i in order[nval:])
    assert not {r[2] for r in train} & {r[2] for r in val}
    return (
        train,
        val,
        {
            "detected_rows": len(rows),
            "unique_hashes": len(groups),
            "conflicting_hashes_dropped": len(conflicts),
            "duplicate_rows": len(rows) - len(groups),
        },
    )


def tensors(rows, feature_version=FEATURE_V1, dimensions=None):
    features = []
    for row in rows:
        kwargs = {}
        if feature_version == FEATURE_V2:
            width, height = dimensions_for_row(row, dimensions or {})
            kwargs = dict(frame_width=width, frame_height=height)
        features.append(
            prepare_features(
                np.frombuffer(row[3], dtype=np.float32).reshape(21, 3),
                row[4],
                feature_version=feature_version,
                **kwargs,
            )
        )
    return torch.from_numpy(np.stack(features)), torch.from_numpy(
        np.array([LABELS.index(r[1]) for r in rows])
    )


def augment(x):
    """Small angular jitter and landmark noise on training batches only."""
    xyz = x.reshape(-1, 21, 3).clone()
    angle = (torch.rand(len(x)) * 2 - 1) * np.deg2rad(10)
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    px, py = xyz[:, :, 0].clone(), xyz[:, :, 1].clone()
    xyz[:, :, 0] = cos * px - sin * py
    xyz[:, :, 1] = sin * px + cos * py
    xyz += torch.randn_like(xyz) * 0.005
    return xyz.reshape(-1, 63)


@torch.inference_mode()
def measure(model, x, y):
    model.eval()
    logits = torch.cat([model(b) for b in x.split(2048)])
    prediction = logits.argmax(1)
    confusion = torch.bincount(y * len(LABELS) + prediction, minlength=len(LABELS) ** 2)
    return {
        "loss": float(nn.functional.cross_entropy(logits, y)),
        "accuracy": float((prediction == y).float().mean()),
        "count": len(y),
        "confusion": confusion.reshape(len(LABELS), -1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New checkpoint path; existing files are never overwritten",
    )
    parser.add_argument("--result-prefix", type=Path, default=Path("results/static_training_new"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; choose a new checkpoint path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result_prefix.parent.mkdir(parents=True, exist_ok=True)
    feature_version = FEATURE_V1 if args.feature_version == "v1" else FEATURE_V2
    dimensions = load_dimensions() if feature_version == FEATURE_V2 else None
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    coverage = json.loads(Path("results/kaggle_full_tracking.json").read_text())
    if set(coverage["letters"]) != set(LABELS):
        raise ValueError("Complete extraction of all 24 labels before training")
    with sqlite3.connect("data/cache/kaggle_landmarks.sqlite") as db:
        count = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if count != sum(r["attempted"] for r in coverage["letters"].values()):
            raise ValueError("Cache does not match completed extraction report")
        rows = db.execute(
            "SELECT path,label,digest,points,hand FROM images "
            "WHERE points IS NOT NULL ORDER BY path"
        ).fetchall()
    train, val, stats = build_split(rows, args.seed)
    manifest = {
        "seed": args.seed,
        "train": [{"path": r[0], "sha256_pixels": r[2], "label": r[1]} for r in train],
        "validation": [{"path": r[0], "sha256_pixels": r[2], "label": r[1]} for r in val],
    }
    manifest_path = Path("data/cache") / (args.output.stem + "_split.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Training on {len(train)} unique images; validation {len(val)}", flush=True)
    train_x, train_y = tensors(train, feature_version, dimensions)
    val_x, val_y = tensors(val, feature_version, dimensions)
    model = StaticMLP(len(LABELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best = float("inf")
    best_epoch = 0
    history = []
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_y))
        for ids in order.split(256):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(augment(train_x[ids])), train_y[ids])
            loss.backward()
            optimizer.step()
        score = measure(model, val_x, val_y)
        history.append(
            {
                "epoch": epoch,
                "validation_loss": score["loss"],
                "validation_accuracy": score["accuracy"],
            }
        )
        print(
            f"Epoch {epoch}: validation loss={score['loss']:.4f} accuracy={score['accuracy']:.4%}",
            flush=True,
        )
        if score["loss"] < best:
            best = score["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": LABELS,
                    "feature_version": feature_version,
                    "seed": args.seed,
                    "epoch": epoch,
                    "split_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                },
                args.output,
            )
        if epoch - best_epoch >= args.patience:
            break
    bundle = torch.load(args.output, weights_only=True)
    model.load_state_dict(bundle["state_dict"])
    result = {
        "seed": args.seed,
        "elapsed_seconds": time.monotonic() - started,
        "device": "cpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "labels": LABELS,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "features": feature_version,
        "deduplication": stats,
        "train": measure(model, train_x, train_y),
        "validation": measure(model, val_x, val_y),
        "history": history,
        "split_sha256": bundle["split_sha256"],
        "torch_version": str(torch.__version__),
        "augmentation": "train only: xy rotation +/-10 degrees, Gaussian noise std 0.005",
        "limitation": "Random image split; near duplicates and signer overlap may remain. "
        "Not a signer-independent evaluation. Metrics condition on detected hands.",
    }
    args.result_prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".labels.json").write_text(json.dumps(LABELS) + "\n")
    print(json.dumps({k: result[k] for k in ("best_epoch", "deduplication")}), flush=True)
    print(
        "Train:",
        result["train"]["accuracy"],
        "Validation:",
        result["validation"]["accuracy"],
        flush=True,
    )


if __name__ == "__main__":
    main()
