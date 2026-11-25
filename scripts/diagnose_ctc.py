"""Recompute transcript metrics and pre-decoder argmax counts from saved logits."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fingerspelling.ctc_decode import greedy_decode
from fingerspelling.sequence_data import VOCABULARY, encode
from fingerspelling.sequence_diagnostics import detailed_metrics


def synthetic_checks():
    results = {}
    for text in [VOCABULARY, "jazz", "all", "a b", " a b ", "  "]:
        tokens = [0]
        for value in encode(text).tolist():
            tokens.extend([value, value, 0])
        logits = np.full((len(tokens), 28), -10.0)
        logits[np.arange(len(tokens)), tokens] = 10
        decoded = greedy_decode(logits.argmax(-1))
        assert decoded == text
        results[text] = decoded
    return results


def diagnose(prefix, vocabulary, output):
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(vocabulary.read_text())
    assert mapping == {"blank": 0, "characters": {c: i + 1 for i, c in enumerate(VOCABULARY)}}
    with prefix.with_suffix(".csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len({r["id"] for r in rows}) == len(rows)
    stored = json.loads(prefix.with_suffix(".json").read_text())
    assert {r["id"]: (r["target"], r["prediction"]) for r in rows} == {
        r["id"]: (r["target"], r["prediction"]) for r in stored["predictions"]
    }
    raw = np.load(str(prefix) + "_logits.npz", allow_pickle=False)
    assert raw["logits"].shape[1] == 28 and np.isfinite(raw["logits"]).all()
    assert len(raw["ids"]) == len(raw["lengths"]) == len(rows)
    counts, offset = Counter(), 0
    by_id = {r["id"]: r for r in rows}
    assert set(raw["ids"]) == set(by_id)
    for key, length in zip(raw["ids"], raw["lengths"]):
        tokens = raw["logits"][offset : offset + length].argmax(-1)
        assert len(tokens) == int(by_id[key]["frames"])
        assert greedy_decode(tokens) == by_id[key]["prediction"]
        counts.update(tokens.tolist())
        offset += int(length)
    assert offset == len(raw["logits"])
    metrics, per_clip = detailed_metrics(rows)
    metrics.update(
        frame_argmax_counts={
            "<blank>" if i == 0 else VOCABULARY[i - 1]: counts[i] for i in range(28)
        },
        synthetic_decoding=synthetic_checks(),
        blank_index=0,
        space_index=27,
        saved_logit_decodes_verified=len(rows),
        decoder_bug_found=False,
    )
    for c in "jzq ":
        metrics["focus"][c]["frame_argmax"] = counts[VOCABULARY.index(c) + 1]
    (output / "diagnostic.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output / "edit_operations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        writer.writeheader()
        writer.writerows(per_clip)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in (
                    "sequences",
                    "reference_characters",
                    "predicted_characters",
                    "character_edits",
                    "deletions",
                    "substitutions",
                    "insertions",
                    "focus",
                )
            },
            indent=2,
        )
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.prefix, args.vocabulary, args.output)
