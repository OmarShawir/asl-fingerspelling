"""Measure the candidate without training, relabeling or reshuffling its splits."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from audit_core import alphabet_scope, ctc_minimum
from fetch_data import FILES, METADATA_SHA, REVISION

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
parser.add_argument("--output-dir", type=Path, default=Path("results"))
args = parser.parse_args()
ROOT = args.cache_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return None
    return {"min": float(a.min()), "p25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "p75": float(np.quantile(a, .75)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()),
            "mean": float(a.mean())}


with (ROOT / "source/ChicagoFSWild.csv").open(newline="") as f:
    original_rows = list(csv.DictReader(f))
original = {r["filename"]: r for r in original_rows}
assert len(original) == len(original_rows)
if sha256(ROOT / "source/ChicagoFSWild.csv") != METADATA_SHA:
    raise ValueError("Official metadata differs from the audited source")
manifest = json.loads((ROOT / "source/download_manifest.json").read_text())
pinned = {name: (size, checksum) for name, size, checksum in FILES}
if len(manifest) != len(pinned) or {x["name"] for x in manifest} != set(pinned):
    raise ValueError("Manifest must contain exactly the three pinned files")
for item in manifest:
    if (item["bytes"], item["sha256"]) != pinned[item["name"]] or item["revision"] != REVISION:
        raise ValueError("Manifest differs from pinned revision")
    p = ROOT / "parquet" / item["name"]
    assert p.stat().st_size == item["bytes"] and sha256(p) == item["sha256"]

summary = {"repository": "Kibalama/ChicagoFSWild-Landmarks",
           "revision": manifest[0]["revision"], "download_bytes": sum(x["bytes"] for x in manifest),
           "official_metadata_sha256": sha256(ROOT / "source/ChicagoFSWild.csv"),
           "official_metadata_rows": len(original), "splits": {}}
all_records = []
signers_by_split, ids_by_split, fingerprints_by_split = {}, {}, {}
issues = []

for split in ("train", "validation", "test"):
    f = pq.ParquetFile(ROOT / "parquet" / f"{split}-00000-of-00001.parquet")
    (OUT / f"{split}_schema.txt").write_text(str(f.schema_arrow))
    counters = {k: Counter() for k in ("chars", "hand_sizes", "pose_sizes", "face_sizes", "resolutions")}
    lengths, text_lengths, coverage, records = [], [], [], []
    ids, signers, fingerprints = set(), set(), Counter()
    issue_counts = Counter()
    frame_total = frame_any_hand = frame_two_hands = frame_no_hand = frame_malformed = 0
    finite_points = nonfinite_points = zero_points = 0
    mins = np.full(3, np.inf); maxs = np.full(3, -np.inf)
    for batch in f.iter_batches(batch_size=32, columns=["id", "label", "hands", "pose", "face"]):
        for row in batch.to_pylist():
            seq_id, text, hands = row["id"], row["label"], row["hands"]
            seq_issues = []
            if seq_id in ids: seq_issues.append("duplicate_id")
            ids.add(seq_id)
            meta = original.get(seq_id)
            if meta is None:
                seq_issues.append("unmatched_original_id")
            else:
                signers.add(meta["signer"])
                counters["resolutions"][f"{meta['width']}x{meta['height']}"] += 1
                if text != meta["label_proc"]: seq_issues.append("label_mismatch")
                expected_split = "validation" if meta["partition"] == "dev" else meta["partition"]
                if expected_split != split: seq_issues.append("split_mismatch")
                if len(hands) != int(meta["number_of_frames"]): seq_issues.append("frame_count_mismatch")
            if not text: seq_issues.append("empty_label")
            if len(hands) != len(row["pose"]) or len(hands) != len(row["face"]):
                seq_issues.append("channel_length_mismatch")
            if len(hands) < ctc_minimum(text): seq_issues.append("ctc_too_short")
            good = malformed = nonfinite = zeros = no_hand = 0
            fp = hashlib.sha256()
            fp.update(len(hands).to_bytes(8, "little"))
            for frame in hands:
                size = len(frame) if frame is not None else -1
                counters["hand_sizes"][size] += 1
                fp.update(int(size).to_bytes(8, "little", signed=True))
                if size == 0:
                    no_hand += 1
                    continue
                if size not in (21, 42):
                    malformed += 1
                    continue
                arr = np.asarray(frame, dtype=np.float64)
                if arr.shape != (size, 3):
                    malformed += 1
                    continue
                fp.update(arr.astype("<f8", copy=False).tobytes())
                finite = np.isfinite(arr).all(axis=1)
                finite_points += int(finite.sum()); nonfinite_points += int((~finite).sum())
                nonfinite += int((~finite).sum())
                zero = np.all(arr == 0, axis=1)
                zeros += int(zero.sum()); zero_points += int(zero.sum())
                if finite.any():
                    mins = np.minimum(mins, arr[finite].min(axis=0))
                    maxs = np.maximum(maxs, arr[finite].max(axis=0))
                if finite.all() and not zero.all():
                    good += 1
                else:
                    malformed += 1
            for channel in ("pose", "face"):
                counters[f"{channel}_sizes"].update(len(x) if x is not None else -1 for x in row[channel])
            if good == 0: seq_issues.append("no_usable_hand_frames")
            if malformed: seq_issues.append("malformed_or_nonfinite_hand_frames")
            fingerprint = fp.hexdigest()
            if good: fingerprints[fingerprint] += 1
            frame_total += len(hands); frame_any_hand += good; frame_no_hand += no_hand
            frame_two_hands += sum(len(x) == 42 for x in hands if x is not None)
            frame_malformed += malformed
            lengths.append(len(hands)); text_lengths.append(len(text)); counters["chars"].update(text)
            coverage.append(good / len(hands) if hands else 0)
            record = {"split": split, "id": seq_id, "label": text,
                      "signer": meta["signer"] if meta else "", "frames": len(hands),
                      "original_frames": meta["number_of_frames"] if meta else "",
                      "width": meta["width"] if meta else "", "height": meta["height"] if meta else "",
                      "usable_hand_frames": good, "empty_hand_frames": no_hand,
                      "malformed_frames": malformed, "nonfinite_points": nonfinite,
                      "zero_points": zeros, "ctc_minimum": ctc_minimum(text),
                      "has_j": "j" in text.lower(), "has_z": "z" in text.lower(),
                      "hands_sha256": fingerprint, "issues": "|".join(seq_issues)}
            records.append(record)
            for problem in seq_issues:
                issue_counts[problem] += 1
                issues.append({"split": split, "id": seq_id, "issue": problem})
    chars = counters["chars"]
    summary["splits"][split] = {
        "sequences": len(records), "unique_ids": len(ids), "signers": len(signers),
        "frames": frame_total, "usable_hand_frames": frame_any_hand, "empty_hand_frames": frame_no_hand,
        "two_hand_frames": frame_two_hands, "malformed_frames": frame_malformed,
        "finite_points": finite_points, "nonfinite_points": nonfinite_points, "zero_points": zero_points,
        "coordinate_min": mins.tolist(), "coordinate_max": maxs.tolist(),
        "character_counts": dict(sorted(chars.items())), "vocabulary": sorted(chars),
        "missing_ascii_letters": [c for c in "abcdefghijklmnopqrstuvwxyz" if not(chars[c] + chars[c.upper()])],
        "j_occurrences": chars['j']+chars['J'], "z_occurrences": chars['z']+chars['Z'],
        "j_sequences": sum(x["has_j"] for x in records), "z_sequences": sum(x["has_z"] for x in records),
        "j_sequences_with_hand_frames": sum(x["has_j"] and x["usable_hand_frames"] > 0 for x in records),
        "z_sequences_with_hand_frames": sum(x["has_z"] and x["usable_hand_frames"] > 0 for x in records),
        "frame_lengths": distribution(lengths), "text_lengths": distribution(text_lengths),
        "hand_coverage": distribution(coverage), "hand_points_per_frame": dict(counters["hand_sizes"]),
        "pose_points_per_frame": dict(counters["pose_sizes"]), "face_points_per_frame": dict(counters["face_sizes"]),
        "original_resolutions": dict(counters["resolutions"]), "issues": dict(issue_counts),
        "nonempty_duplicate_sequence_groups": sum(n > 1 for n in fingerprints.values())}
    all_records.extend(records)
    ids_by_split[split], signers_by_split[split], fingerprints_by_split[split] = ids, signers, set(fingerprints)
    print(split, {k: summary["splits"][split][k] for k in ("sequences", "signers", "frames", "usable_hand_frames", "empty_hand_frames", "j_sequences", "z_sequences", "issues")}, flush=True)

summary["overlaps"] = {}
for a, b in itertools.combinations(summary["splits"], 2):
    summary["overlaps"][f"{a}/{b}"] = {
        "ids": sorted(ids_by_split[a] & ids_by_split[b]),
        "signers": sorted(signers_by_split[a] & signers_by_split[b]),
        "nonempty_hand_sequence_fingerprints": sorted(fingerprints_by_split[a] & fingerprints_by_split[b])}
summary["unmatched_original_ids"] = sorted(set(original) - set.union(*ids_by_split.values()))
summary["unique_signers"] = len(set.union(*signers_by_split.values()))
train_vocab = set(summary["splits"]["train"]["vocabulary"])
summary["symbols_unseen_in_train"] = {s: sorted(set(summary["splits"][s]["vocabulary"]) - train_vocab) for s in ("validation", "test")}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
with (OUT / "sequence_manifest.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_records[0]));writer.writeheader();writer.writerows(all_records)
with (OUT / "issues.csv").open("w", newline="") as f:
    writer=csv.DictWriter(f,fieldnames=["split","id","issue"]);writer.writeheader();writer.writerows(issues)
print('OVERLAPS',summary['overlaps'],flush=True)

# A declared alphabet-plus-space pilot: exclude whole out-of-scope clips,
# preserve every source label and timestamp position, retain the original splits.
plan = {"status": "eligible_for_offline_pilot_with_documented_unknowns",
        "scope": "lowercase ASCII a-z plus literal space",
        "vocabulary": list("abcdefghijklmnopqrstuvwxyz "), "ctc_blank_index": 0,
        "vocabulary_index_offset": 1,
        "not_supported_by_this_dataset": "digits; no digit target occurs",
        "no_time_downsampling": True,
        "hand_channel_identity": "unknown; never assume first block is anatomical left",
        "frame_order": "stored order retained; count matches official metadata, chronology not independently verified",
        "splits": {}}
excluded = []
for split in ("train", "validation", "test"):
    rows = [r for r in all_records if r["split"] == split]
    scope_rows = [r for r in rows if alphabet_scope(r["label"])]
    selected = [r for r in scope_rows if r["usable_hand_frames"] > 0]
    for row in rows:
        reasons = []
        if not alphabet_scope(row["label"]): reasons.append("whole_sequence_outside_alphabet_space_scope")
        if row["usable_hand_frames"] == 0: reasons.append("no_hand_input")
        if reasons: excluded.append({"split": split, "id": row["id"], "label": row["label"], "reason": "|".join(reasons)})
    loss_rows = [r for r in selected if r["frames"] >= r["ctc_minimum"]]
    plan["splits"][split] = {"original_sequences": len(rows), "pilot_sequences": len(selected),
        "ctc_loss_eligible": len(loss_rows), "metric_denominator": len(selected),
        "excluded_whole_sequences": len(rows) - len(selected),
        "signers": len({r["signer"] for r in selected}),
        "j_sequences": sum(r["has_j"] for r in selected),
        "z_sequences": sum(r["has_z"] for r in selected),
        "ctc_infeasible_ids_retained_in_metrics": [r["id"] for r in selected if r["frames"] < r["ctc_minimum"]]}
    (OUT / f"pilot_{split}_ids.txt").write_text("".join(r["id"] + "\n" for r in selected))
    (OUT / f"pilot_{split}_ctc_loss_ids.txt").write_text("".join(r["id"] + "\n" for r in loss_rows))
with (OUT / "pilot_exclusions.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "id", "label", "reason"])
    writer.writeheader(); writer.writerows(excluded)
(OUT / "pilot_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
print("PILOT", plan["splits"], flush=True)
