"""Final development-set selection using fixed Kaggle split and s04+s06 training."""

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from finetune_static import camera_score, kaggle_score, kaggle_tensors, load_camera, sha
from fingerspelling.adaptation import assert_disjoint, balanced_batch, camera_frame_weights
from fingerspelling.features import FEATURE_V2, LABELS
from fingerspelling.models.mlp import StaticMLP
from fingerspelling.training_data import load_dimensions

OUT = Path("results/final_static")
SOURCE = Path("models/mlp_static_adapted.pt")
DEST = Path("models/mlp_static_final.pt")


def accept_candidate(candidate, best, baseline):
    """NLL only breaks ties among candidates strictly above epoch-zero accuracy."""
    return candidate["macro_accuracy"] > baseline["macro_accuracy"] and (
        candidate["macro_accuracy"] > best["macro_accuracy"]
        or (
            candidate["macro_accuracy"] == best["macro_accuracy"]
            and candidate["macro_nll"] < best["macro_nll"]
        )
    )


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(name, value):
    (OUT / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")


def paired(name, before, after, labels):
    a = {c["clip_id"]: c for c in before["clips"]}
    b = {c["clip_id"]: c for c in after["clips"]}
    if a.keys() != b.keys():
        raise ValueError("Unpaired reports")
    rows = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["target"] != y["target"]:
            raise ValueError("Targets changed")
        category = (
            "both_correct"
            if x["correct"] and y["correct"]
            else "improved"
            if y["correct"]
            else "regressed"
            if x["correct"]
            else "both_wrong"
        )
        rows.append(
            dict(
                clip_id=key,
                target=x["target"],
                before=x["prediction"],
                after=y["prediction"],
                category=category,
                valid_frames=x["valid_frames"],
                before_nll=x["nll"],
                after_nll=y["nll"],
            )
        )
    for suffix, subset in [
        ("comparison", rows),
        ("improved", [r for r in rows if r["category"] == "improved"]),
        ("regressed", [r for r in rows if r["category"] == "regressed"]),
    ]:
        with (OUT / f"{name}_{suffix}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    per_letter(name, before, after, labels)
    return dict(Counter(r["category"] for r in rows))


def per_letter(name, before, after, labels):
    with (OUT / f"{name}_per_letter.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "before_correct", "after_correct", "total"])
        for label in labels:
            a, b = before["per_letter"][label], after["per_letter"][label]
            assert a["total"] == b["total"]
            writer.writerow([label, a["correct"], b["correct"], a["total"]])


def main():
    if DEST.exists() or OUT.exists():
        raise SystemExit("Refusing to overwrite final checkpoint or experiment directory")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(42)
    bundle = torch.load(SOURCE, map_location="cpu", weights_only=True)
    labels = bundle["labels"]
    if bundle["feature_version"] != FEATURE_V2 or len(labels) != 24 or set(labels) != set(LABELS):
        raise ValueError("Expected v2 and all 24 labels")
    protected = {str(SOURCE): sha(SOURCE)}
    previous = json.loads(Path("results/adaptation/manifest.json").read_text())["configuration"]
    config = {
        k: previous[k]
        for k in (
            "seed",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "source_fraction_camera",
            "max_epochs",
            "steps_per_epoch",
            "patience",
            "augmentation",
            "sampling",
        )
    }
    config.update(
        feature_version=FEATURE_V2,
        labels=labels,
        source_checkpoint=str(SOURCE),
        source_checkpoint_sha256=sha(SOURCE),
        source_state_sha256=state_hash(bundle["state_dict"]),
        training_sessions=["p01/s04", "p01/s06"],
        development_sessions=["p01/s03", "p01/s05"],
        selection="combined 24-class macro clip accuracy; strict improvement over epoch 0 required; ties among improved candidates use lower macro clip NLL",
        aggregation="mean valid-frame probabilities over all 24 classes",
        kaggle_source="grassknoted/asl-alphabet; original saved split",
    )
    sessions = {s: load_camera(s, labels) for s in ("s03", "s04", "s05", "s06")}
    train = sessions["s04"] + sessions["s06"]
    dev = sessions["s03"] + sessions["s05"]
    for clips in (train, dev):
        if len({c["clip_id"] for c in clips}) != len(clips):
            raise ValueError("Duplicate clip ID within group")
    for field in ("clip_id", "landmark_sequence_sha256"):
        assert_disjoint([c[field] for c in train], [c[field] for c in dev])
    # Also detect identical valid feature sequences despite relabeled handedness/metadata.
    feature_digests = {}
    for c in train + dev:
        protected[c["path"]] = c["sha256"]
        feature_digests[c["clip_id"]] = hashlib.sha256(c["x"].numpy().tobytes()).hexdigest()
    assert_disjoint(
        [feature_digests[c["clip_id"]] for c in train], [feature_digests[c["clip_id"]] for c in dev]
    )
    if set(c["target"] for c in dev) != set(labels):
        raise ValueError("Combined development must cover all 24 letters")
    split_path = Path("data/cache/static_split.json")
    protected[str(split_path)] = sha(split_path)
    if sha(split_path) != bundle["split_sha256"]:
        raise ValueError("Original split changed")
    config["split_sha256"] = sha(split_path)
    split = json.loads(split_path.read_text())
    for field in ("path", "sha256_pixels"):
        assert_disjoint([r[field] for r in split["train"]], [r[field] for r in split["validation"]])
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
    cx = torch.cat([c["x"] for c in train])
    cy = torch.tensor([c["target_index"] for c in train for _ in range(c["valid_frames"])])
    frame_ids = [c["clip_id"] for c in train for _ in range(c["valid_frames"])]
    cw = camera_frame_weights(cy.tolist(), frame_ids)
    counts = Counter(ky.tolist())
    kw = torch.tensor([1 / (24 * counts[y]) for y in ky.tolist()], dtype=torch.float64)
    OUT.mkdir(parents=True)
    audit = {}
    for s, clips in sessions.items():
        audit[s] = dict(
            clips=len(clips),
            valid_frames=sum(c["valid_frames"] for c in clips),
            frames=sum(c["frames"] for c in clips),
            per_letter=dict(Counter(c["target"] for c in clips)),
            dimensions=sorted({(c["frame_width"], c["frame_height"]) for c in clips}),
            hands=sorted({c["hand"] for c in clips}),
        )
    write_json(
        "manifest",
        dict(
            configuration=config,
            audit=audit,
            kaggle_train=len(ky),
            kaggle_validation=len(vy),
            clips={
                s: [{k: v for k, v in c.items() if k != "x"} for c in cs]
                for s, cs in sessions.items()
            },
            overlap_checks=dict(
                clip_ids=[],
                valid_landmark_sequences=[],
                feature_sequences=[],
                kaggle_paths=[],
                kaggle_pixel_hashes=[],
            ),
            duplicate_check_scope="exact complete valid landmark/feature sequences, independent of metadata; not a guarantee against near-duplicate performances",
            inputs_sha256=protected,
        ),
    )
    model = StaticMLP(24)
    model.load_state_dict(bundle["state_dict"])

    def scores():
        return {
            **{s: camera_score(model, sessions[s], labels) for s in ("s03", "s05", "s06")},
            "combined": camera_score(model, dev, labels),
            "camera_training": camera_score(model, train, labels),
            "kaggle_validation": kaggle_score(model, vx, vy, labels),
            "kaggle_training": kaggle_score(model, kx, ky, labels),
        }

    before = scores()
    for name, value in before.items():
        write_json(name + "_before", value)
    baseline = before["combined"]
    print("AUDIT", json.dumps(audit), flush=True)
    print(
        f"Epoch 0: {baseline['correct']}/{baseline['total']}; macro {baseline['macro_accuracy']:.8f}; NLL {baseline['macro_nll']:.8f}; s06 {before['s06']['correct']}/{before['s06']['total']}",
        flush=True,
    )
    best = baseline
    best_epoch = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    history = [
        dict(
            epoch=0,
            train_sampled_loss=None,
            macro_accuracy=best["macro_accuracy"],
            macro_nll=best["macro_nll"],
            correct=best["correct"],
        )
    ]
    draws = Counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for _ in range(config["steps_per_epoch"]):
            ki, ci = balanced_batch(kw, cw, config["batch_size"] // 2, generator)
            x, y = torch.cat((kx[ki], cx[ci])), torch.cat((ky[ki], cy[ci]))
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            draws.update(frame_ids[i] for i in ci.tolist())
        score = camera_score(model, dev, labels)
        accepted = accept_candidate(score, best, baseline)
        history.append(
            dict(
                epoch=epoch,
                train_sampled_loss=float(np.mean(losses)),
                macro_accuracy=score["macro_accuracy"],
                macro_nll=score["macro_nll"],
                correct=score["correct"],
                selected=accepted,
            )
        )
        if accepted:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        write_json("history", history)
        print(
            f"Epoch {epoch}: dev {score['correct']}/162 macro {score['macro_accuracy']:.8f} NLL {score['macro_nll']:.8f} selected={accepted}",
            flush=True,
        )
        if epoch - best_epoch >= config["patience"]:
            break
    model.load_state_dict(best_state)
    after = scores()
    for name, value in after.items():
        write_json(name + "_after", value)
    outcomes = {
        s: paired(s, before[s], after[s], labels)
        for s in ("s03", "s05", "combined", "s06", "camera_training")
    }
    for name in ("kaggle_validation", "kaggle_training"):
        per_letter(name, before[name], after[name], labels)
    if not all(sha(path) == h for path, h in protected.items()):
        raise ValueError("Inputs changed")
    final_bundle = dict(bundle)
    final_bundle.update(
        state_dict=best_state,
        epoch=best_epoch,
        final_training=config,
        final_manifest_sha256=sha(OUT / "manifest.json"),
    )
    torch.save(final_bundle, DEST)
    report = dict(
        configuration=config,
        adopted_new_weights=best_epoch > 0,
        best_epoch=best_epoch,
        epochs_run=len(history) - 1,
        history=history,
        before=before,
        after=after,
        outcomes=outcomes,
        camera_draws_by_clip=dict(draws),
        output_checkpoint=str(DEST),
        output_checkpoint_sha256=sha(DEST),
        output_state_sha256=state_hash(best_state),
        inputs_unchanged=True,
        interpretation="s03 and s05 are development validation used for selection; s06 is training. No independent final test.",
        torch_version=str(torch.__version__),
    )
    write_json("report", report)
    print(
        "FINAL",
        best_epoch,
        best["correct"],
        best["macro_accuracy"],
        "Kaggle",
        before["kaggle_validation"]["correct"],
        after["kaggle_validation"]["correct"],
        flush=True,
    )


if __name__ == "__main__":
    main()
