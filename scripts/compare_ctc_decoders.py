"""One fixed-width comparison on saved validation logits, never on test."""

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from fingerspelling.ctc_decode import transcript_metrics
from fingerspelling.sequence_data import VOCABULARY, FEATURE_VERSION, sha256
from fingerspelling.sequence_decode import GreedyDecoder, PrefixBeamDecoder, log_probabilities

SOURCE = Path("results/sequence/gru_ctc_long_v1")
OUT = Path("results/sequence/ctc_decode_v1")


def main():
    if OUT.exists():
        raise SystemExit("Refusing to replace decoder results")
    report = json.loads((SOURCE / "report.json").read_text())
    checkpoint = Path("models/sequence/gru_ctc_long_v1.pt")
    assert sha256(checkpoint) == report["checkpoint_sha256"]
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert bundle["vocabulary"] == VOCABULARY and bundle["blank_index"] == 0
    assert bundle["feature_version"] == FEATURE_VERSION
    vocabulary = json.loads((SOURCE / "vocabulary.json").read_text())
    assert vocabulary == dict(blank=0, characters={c: i + 1 for i, c in enumerate(VOCABULARY)})
    with (SOURCE / "validation_predictions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    stored = json.loads((SOURCE / "validation_predictions.json").read_text())
    assert {r["id"]: (r["target"], r["prediction"]) for r in rows} == {
        r["id"]: (r["target"], r["prediction"]) for r in stored["predictions"]
    }
    ids = set(Path("asl_sequence_audit/results/pilot_validation_ids.txt").read_text().splitlines())
    assert len(rows) == 978 and len({r["id"] for r in rows}) == 978
    assert {r["id"] for r in rows} == ids
    with Path("asl_sequence_audit/results/sequence_manifest.csv").open(newline="") as f:
        audit = {r["id"]: r for r in csv.DictReader(f) if r["id"] in ids}
    for r in rows:
        assert r["target"] == audit[r["id"]]["label"]
        assert int(r["frames"]) == int(audit[r["id"]]["frames"])
    raw = np.load(SOURCE / "validation_predictions_logits.npz", allow_pickle=False)
    assert raw["ids"].tolist() == [r["id"] for r in rows]
    assert raw["lengths"].tolist() == [int(r["frames"]) for r in rows]
    assert raw["logits"].shape == (sum(raw["lengths"]), 28)
    assert np.isfinite(raw["logits"]).all()
    OUT.mkdir(parents=True)
    outputs = {"greedy": [], "prefix_beam": []}
    timings = {k: [] for k in outputs}
    started = time.perf_counter()
    offset = 0
    for row, length in zip(rows, raw["lengths"]):
        logits = raw["logits"][offset : offset + length]
        offset += int(length)
        for name, decoder in [("greedy", GreedyDecoder()), ("prefix_beam", PrefixBeamDecoder(10))]:
            begin = time.perf_counter()
            for frame in logits:
                if time.perf_counter() - started > 900:
                    raise TimeoutError(
                        "Comparison exceeded 900 seconds; no decoder selection written"
                    )
                decoder.step(log_probabilities(frame))
            elapsed = (time.perf_counter() - begin) * 1000
            timings[name].append(elapsed)
            outputs[name].append(
                dict(
                    id=row["id"],
                    target=row["target"],
                    prediction=decoder.text,
                    frames=int(length),
                    decoding_ms=elapsed,
                )
            )
            if name == "greedy":
                assert decoder.text == row["prediction"]
    metrics = {name: transcript_metrics(pred) for name, pred in outputs.items()}
    assert [
        metrics["greedy"][k]
        for k in ("sequences", "reference_characters", "character_edits", "exact", "empty")
    ] == [978, 4967, 2121, 170, 17]
    for name in metrics:
        metrics[name].update(
            mean_decoding_ms=float(np.mean(timings[name])),
            p95_decoding_ms=float(np.quantile(timings[name], 0.95)),
        )
        with (OUT / f"{name}_predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(outputs[name][0]))
            w.writeheader()
            w.writerows(outputs[name])
    selected = (
        "prefix_beam" if metrics["prefix_beam"]["cer"] < metrics["greedy"]["cer"] else "greedy"
    )
    settings = dict(
        name=selected,
        beam_width=10 if selected == "prefix_beam" else None,
        blank_index=0,
        vocabulary=VOCABULARY,
        checkpoint_sha256=sha256(checkpoint),
        feature_version=FEATURE_VERSION,
        language_model=None,
        token_pruning=False,
    )
    (OUT / "selected_decoder.json").write_text(json.dumps(settings, indent=2) + "\n")
    result = dict(
        metrics=metrics,
        selected=selected,
        elapsed_seconds=time.perf_counter() - started,
        budget_seconds=900,
        threads_note="single-thread Python decoder; timings include log-softmax and incremental decoding, exclude loading and model inference",
        tie_rule="greedy on equal validation CER; beam prefix score ties lexicographic token IDs",
        provenance={
            str(p): sha256(p)
            for p in [
                checkpoint,
                SOURCE / "report.json",
                SOURCE / "vocabulary.json",
                SOURCE / "validation_predictions.csv",
                SOURCE / "validation_predictions.json",
                SOURCE / "validation_predictions_logits.npz",
                Path("asl_sequence_audit/results/pilot_validation_ids.txt"),
            ]
        },
        no_training=True,
        test_inference=False,
    )
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print("SELECTED", selected, flush=True)


if __name__ == "__main__":
    main()
