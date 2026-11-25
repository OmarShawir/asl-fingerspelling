"""Audited ChicagoFSWild subset; no anatomical hand identity or static v2 assumptions."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

FEATURE_VERSION = "sequence-unordered-hand-shape-wrist-scale-v1"
VOCABULARY = "abcdefghijklmnopqrstuvwxyz "
BLANK = 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(text):
    if not text or any(c not in VOCABULARY for c in text):
        raise ValueError("Target must be unchanged alphabet-plus-space text")
    return torch.tensor([VOCABULARY.index(c) + 1 for c in text], dtype=torch.long)


def ctc_minimum(text):
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def hand_features(frames):
    """21-point blocks assumed; retain time, wrist xyz and scale, pool identities later."""
    if not frames:
        raise ValueError("Empty sequence")
    values = np.zeros((len(frames), 2, 67), dtype=np.float32)
    mask = np.zeros((len(frames), 2), dtype=bool)
    for t, frame in enumerate(frames):
        if len(frame) not in (0, 21, 42):
            raise ValueError("Expected 0, 21 or 42 points")
        if not frame:
            continue
        points = np.asarray(frame, dtype=np.float32)
        if points.shape != (len(frame), 3) or not np.isfinite(points).all():
            raise ValueError("Invalid hand coordinates")
        for k, block in enumerate(points.reshape(-1, 21, 3)):
            wrist = block[0]
            shape = block - wrist
            scale = max(float(np.linalg.norm(shape, axis=1).max()), 1e-6)
            values[t, k] = np.concatenate((shape.reshape(-1) / scale, wrist, [np.log(scale)]))
            mask[t, k] = True
    return torch.from_numpy(values), torch.from_numpy(mask)


def fingerprint(frames):
    digest = hashlib.sha256()
    digest.update(len(frames).to_bytes(8, "little"))
    for frame in frames:
        digest.update(len(frame).to_bytes(8, "little", signed=True))
        if frame:
            digest.update(np.asarray(frame, dtype="<f8").tobytes())
    return digest.hexdigest()


def verify_sources(cache, audit):
    """Hashes include test payload but never decode test hands or infer test predictions."""
    bundle = json.loads((audit / "FILES_SHA256.json").read_text())
    for name, checksum in bundle.items():
        if sha256(audit / name) != checksum:
            raise ValueError("Audit package changed: " + name)
    manifest = json.loads((audit / "provenance/download_manifest.json").read_text())
    hashes = {}
    for item in manifest:
        path = cache / "parquet" / item["name"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Downloaded source differs from pinned payload")
        hashes[str(path)] = item["sha256"]
    original = cache / "source/ChicagoFSWild.csv"
    expected = json.loads((audit / "results/summary.json").read_text())["official_metadata_sha256"]
    if sha256(original) != expected:
        raise ValueError("Official CSV differs")
    hashes[str(original)] = expected
    with (audit / "results/sequence_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source sequence IDs")
    memberships, signers, fingerprints = {}, {}, {}
    for split in ("train", "validation", "test"):
        path = audit / "results" / f"pilot_{split}_ids.txt"
        ids = path.read_text().splitlines()
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate membership IDs")
        memberships[split] = set(ids)
        hashes[str(path)] = sha256(path)
        scoped = [r for r in rows if r["id"] in memberships[split]]
        if len(scoped) != len(ids) or any(r["split"] != split for r in scoped):
            raise ValueError("Invalid fixed memberships")
        signers[split] = {r["signer"] for r in scoped}
        fingerprints[split] = {r["hands_sha256"] for r in scoped}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if memberships[a] & memberships[b] or signers[a] & signers[b]:
            raise ValueError("Signer or ID leakage")
        if fingerprints[a] & fingerprints[b]:
            raise ValueError("Exact hand sequence leakage")
    for name in ("results/sequence_manifest.csv", "results/pilot_plan.json", "FILES_SHA256.json"):
        hashes[str(audit / name)] = sha256(audit / name)
    return hashes


def load_sequences(cache, audit, split):
    if split not in ("train", "validation"):
        raise ValueError("Development loader deliberately refuses test inference")
    import pyarrow.parquet as pq

    ids = set((audit / "results" / f"pilot_{split}_ids.txt").read_text().splitlines())
    with (audit / "results/sequence_manifest.csv").open(newline="") as f:
        records = {r["id"]: r for r in csv.DictReader(f) if r["split"] == split}
    sequences = {}
    parquet = pq.ParquetFile(cache / "parquet" / f"{split}-00000-of-00001.parquet")
    seen = set()
    for batch in parquet.iter_batches(batch_size=16, columns=["id", "label", "hands"]):
        for row in batch.to_pylist():
            key, text, frames = row["id"], row["label"], row["hands"]
            if key in seen or key not in records:
                raise ValueError("Duplicate or unknown sequence ID")
            seen.add(key)
            meta = records[key]
            if (
                text != meta["label"]
                or len(frames) != int(meta["frames"])
                or fingerprint(frames) != meta["hands_sha256"]
            ):
                raise ValueError("Sequence differs from audited evidence: " + key)
            if key not in ids:
                continue
            x, mask = hand_features(frames)
            target = encode(text)
            if not mask.any():
                raise ValueError("Scoped sequence has no hand input")
            if split == "train" and len(x) < ctc_minimum(text):
                raise ValueError("Unalignable training sequence")
            sequences[key] = dict(
                id=key, text=text, signer=meta["signer"], x=x, mask=mask, target=target
            )
    if seen != set(records) or set(sequences) != ids:
        raise ValueError("Incomplete source or pilot membership")
    return [sequences[key] for key in sorted(sequences)]


def collate_sequences(rows):
    lengths = torch.tensor([len(r["x"]) for r in rows], dtype=torch.long)
    return dict(
        x=pad_sequence([r["x"] for r in rows], batch_first=True),
        mask=pad_sequence([r["mask"] for r in rows], batch_first=True),
        lengths=lengths,
        targets=torch.cat([r["target"] for r in rows]),
        target_lengths=torch.tensor([len(r["target"]) for r in rows], dtype=torch.long),
        rows=rows,
    )
