"""Audit actual image dimensions and derive a same-weights v2 checkpoint only for square data."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from fingerspelling.features import FEATURE_V1, FEATURE_V2, prepare_features


def weights_hash(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        digest.update(
            key.encode() + b"\0" + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()
        )
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("models/mlp_static.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mlp_static_aspect_v2.pt"))
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        raise SystemExit("Refusing to overwrite an existing checkpoint; choose a new --output")
    source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
    bundle = torch.load(args.source, map_location="cpu", weights_only=True)
    if bundle["feature_version"] != FEATURE_V1:
        raise ValueError("Source must be v1")
    with sqlite3.connect("file:data/cache/kaggle_landmarks.sqlite?mode=ro", uri=True) as db:
        settings = json.loads(db.execute("SELECT signature FROM settings").fetchone()[0])
        rows = db.execute(
            "SELECT path,label,digest,points,hand FROM images ORDER BY path"
        ).fetchall()
    root = Path(settings["root"])
    sizes = Counter()
    entries = {}
    maximum = 0.0
    compared = 0
    for i, (path, label, digest, blob, hand) in enumerate(rows, 1):
        image = cv2.imread(str(root / path))
        if image is None:
            raise ValueError(f"Unreadable source image {path}")
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        actual = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Image changed since extraction: {path}")
        sizes[f"{width}x{height}"] += 1
        entries[path] = dict(frame_width=width, frame_height=height, pixel_sha256=digest)
        if blob is not None:
            xyz = np.frombuffer(blob, dtype=np.float32).reshape(21, 3)
            a = prepare_features(xyz, hand)
            b = prepare_features(
                xyz, hand, feature_version=FEATURE_V2, frame_width=width, frame_height=height
            )
            maximum = max(maximum, float(np.abs(a - b).max()))
            compared += 1
        if i % 10000 == 0:
            print(f"Audited {i}/{len(rows)} actual images", flush=True)
    manifest_path = Path("data/cache/static_split.json")
    manifest = json.loads(manifest_path.read_text())
    split_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if bundle["split_sha256"] != split_sha:
        raise ValueError("Source checkpoint split manifest mismatch")
    for subset in ("train", "validation"):
        for row in manifest[subset]:
            if entries[row["path"]]["pixel_sha256"] != row["sha256_pixels"]:
                raise ValueError("Split source mismatch")
    dimension_path = Path("data/cache/kaggle_dimensions.json")
    dimension_path.write_text(
        json.dumps(
            {
                "source": "cv2 decoded image.shape; checked against cached RGB hashes",
                "images": entries,
            }
        )
        + "\n"
    )
    audit = dict(
        image_count=len(rows),
        image_sizes=dict(sizes),
        detected_features_compared=compared,
        train_count=len(manifest["train"]),
        validation_count=len(manifest["validation"]),
        max_abs_feature_difference=maximum,
        dimensions_manifest_sha256=hashlib.sha256(dimension_path.read_bytes()).hexdigest(),
        split_sha256=split_sha,
        source_checkpoint=str(args.source),
        source_checkpoint_sha256=source_sha,
        feature_version=FEATURE_V2,
        weights_sha256=weights_hash(bundle["state_dict"]),
        weights_hash_format="sorted keys + NUL + numpy dtype + shape + contiguous tensor bytes",
        training_performed=False,
    )
    if any(e["frame_width"] != e["frame_height"] for e in entries.values()) or maximum != 0:
        raise ValueError(
            "Same-weights derivation not justified: images/features differ; no model written"
        )
    derived = {**bundle, "feature_version": FEATURE_V2, "derivation": audit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, args.output)
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    assert weights_hash(loaded["state_dict"]) == audit["weights_sha256"]
    assert all(torch.equal(v, loaded["state_dict"][k]) for k, v in bundle["state_dict"].items())
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_sha
    audit.update(
        derived_checkpoint=str(args.output),
        derived_checkpoint_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        tensors_bit_identical=True,
    )
    Path("results/aspect_v2_derivation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
