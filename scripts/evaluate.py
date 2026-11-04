"""One prediction per held-out webcam clip, averaging valid-frame probabilities."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from fingerspelling.models.mlp import StaticPredictor
from fingerspelling.evaluation import rank_clip
from fingerspelling.features import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/mlp_static.pt")
    parser.add_argument("--person", default="p01")
    parser.add_argument("--session", default="s01")
    parser.add_argument("--output-prefix", type=Path, default=Path("results/webcam_evaluation"))
    args = parser.parse_args()
    predictor = StaticPredictor(args.checkpoint)
    if set(predictor.labels) != set(LABELS) or len(predictor.labels) != 24:
        raise ValueError("Evaluation requires all 24 static labels")
    results = []
    for path in sorted(Path("data/clips").glob("*.npz")):
        with np.load(path, allow_pickle=False) as c:
            meta = json.loads(c["meta_json"].item())
            if (meta["person"], meta["session"], meta["kind"], meta["source"]) != (
                args.person,
                args.session,
                "letter",
                "webcam",
            ):
                continue
            valid = c["valid"].astype(bool) & np.isfinite(c["xyz"]).all(axis=(1, 2))
            xyz = c["xyz"][valid]
        prediction = None
        confidence = 0.0
        top3 = []
        if len(xyz):
            top3 = rank_clip(predictor.probabilities(xyz, meta["hand"]), predictor.labels)
            prediction = top3[0]["letter"]
            confidence = top3[0]["score"]
        results.append(
            {
                "clip_id": meta["clip_id"],
                "target": meta["text"],
                "hand": meta["hand"],
                "prediction": prediction,
                "confidence": confidence,
                "top3": top3,
                "frames": meta["frames"],
                "valid_frames": len(xyz),
                "correct": prediction == meta["text"],
            }
        )
    if not results:
        raise SystemExit("No matching clips")
    report = {
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "labels": predictor.labels,
        "person": args.person,
        "session": args.session,
        "correct": sum(r["correct"] for r in results),
        "total": len(results),
        "frames": sum(r["frames"] for r in results),
        "valid_frames": sum(r["valid_frames"] for r in results),
        "clips": results,
        "aggregation": "mean softmax probabilities over valid frames; one argmax per clip",
        "training_use": "none; no model or hyperparameter selection on these clips",
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["clip_id", "target", "prediction", "valid_frames"]
    fields += [f"top{i}_{kind}" for i in range(1, 4) for kind in ("letter", "score")]
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result[key] for key in fields[:4]}
            for i, item in enumerate(result["top3"], 1):
                row[f"top{i}_letter"] = item["letter"]
                row[f"top{i}_score"] = item["score"]
            writer.writerow(row)
    print(
        f"Correct: {report['correct']}/{report['total']}; frames: {report['frames']}; valid: {report['valid_frames']}"
    )
    for result in results:
        if not result["correct"]:
            ranking = ", ".join(f"{r['letter']}={r['score']:.4%}" for r in result["top3"])
            print(
                f"{result['clip_id']}: {result['target']} -> {result['prediction']} | {ranking} | valid={result['valid_frames']}"
            )


if __name__ == "__main__":
    main()
