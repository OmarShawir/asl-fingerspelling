"""Compare same-weight v1/v2 checkpoints on one dimension-documented clip set."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fingerspelling.features import FEATURE_V1, FEATURE_V2, LABELS, validate_dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--v1", type=Path, default=Path("models/mlp_static.pt"))
    parser.add_argument("--v2", type=Path, default=Path("models/mlp_static_aspect_v2.pt"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    a = torch.load(args.v1, map_location="cpu", weights_only=True)
    b = torch.load(args.v2, map_location="cpu", weights_only=True)
    if (a["feature_version"], b["feature_version"]) != (FEATURE_V1, FEATURE_V2):
        raise SystemExit("Expected v1 and v2 checkpoints, respectively")
    if a["labels"] != b["labels"] or set(a["labels"]) != set(LABELS) or len(a["labels"]) != 24:
        raise SystemExit("Checkpoints must score the same ordered 24 classes")
    if a["state_dict"].keys() != b["state_dict"].keys() or any(
        not torch.equal(v, b["state_dict"][k]) for k, v in a["state_dict"].items()
    ):
        raise SystemExit("This controlled comparison requires bit-identical weights")
    snapshot = {}
    for path in sorted((args.data_dir / "clips").glob("*.npz")):
        with np.load(path, allow_pickle=False) as c:
            meta = json.loads(c["meta_json"].item())
        if (meta["person"], meta["session"], meta["kind"], meta["source"]) != (
            args.person,
            args.session,
            "letter",
            "webcam",
        ):
            continue
        try:
            validate_dimensions(meta.get("frame_width"), meta.get("frame_height"))
        except ValueError as error:
            raise SystemExit(
                f"Clip {path.stem}: {error}. Record a new session; no dimensions are guessed."
            ) from error
        snapshot[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not snapshot:
        raise SystemExit("No matching clips")
    reports = {}
    for version, checkpoint in [("v1", args.v1), ("v2", args.v2)]:
        prefix = Path(str(args.output_prefix) + "_" + version)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("evaluate.py")),
                "--checkpoint",
                str(checkpoint),
                "--person",
                args.person,
                "--session",
                args.session,
                "--data-dir",
                str(args.data_dir),
                "--output-prefix",
                str(prefix),
            ],
            check=True,
        )
        reports[version] = json.loads(prefix.with_suffix(".json").read_text())
    ids = {Path(p).stem for p in snapshot}
    for report in reports.values():
        if {r["clip_id"] for r in report["clips"]} != ids:
            raise SystemExit("Clip set changed during comparison; rerun with collector closed")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in snapshot.items()):
        raise SystemExit("Clip content changed during comparison; rerun with collector closed")
    report = {
        "person": args.person,
        "session": args.session,
        "labels": a["labels"],
        "identical_weights": True,
        "clip_sha256": snapshot,
        "checkpoints": {
            v: {
                k: r[k]
                for k in (
                    "checkpoint_sha256",
                    "feature_version",
                    "correct",
                    "total",
                    "valid_frames",
                )
            }
            for v, r in reports.items()
        },
        "aggregation": "one clip prediction from mean valid-frame probabilities over all 24 classes",
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("Paired comparison:", json.dumps(report["checkpoints"], indent=2))


if __name__ == "__main__":
    main()
