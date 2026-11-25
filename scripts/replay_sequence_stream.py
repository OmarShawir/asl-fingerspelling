"""Validation-only saved sequence replay: full forward versus causal frame stepping."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from fingerspelling.sequence_data import hand_features, sha256
from fingerspelling.sequence_decode import make_decoder, log_probabilities
from fingerspelling.sequence_live import SequencePredictor

OUT = Path("results/sequence/ctc_decode_v1")


def main():
    dest = OUT / "stream_replay.json"
    if dest.exists():
        raise SystemExit("Refusing to overwrite replay evidence")
    torch.set_num_threads(4)
    predictor = SequencePredictor(
        "models/sequence/gru_ctc_long_v1.pt", Path("configs/sequence_decoder.json")
    )
    ids = sorted(
        Path("asl_sequence_audit/results/pilot_validation_ids.txt").read_text().splitlines()
    )[:2]
    found = {}
    # Only two existing validation examples; no full-dataset audit or test inference.
    path = Path("data/external/chicagofswild/parquet/validation-00000-of-00001.parquet")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["id", "label", "hands"]):
        for row in batch.to_pylist():
            if row["id"] in ids:
                found[row["id"]] = row
        if len(found) == len(ids):
            break
    results = []
    for key in ids:
        row = found[key]
        x, mask = hand_features(row["hands"])
        with torch.inference_mode():
            full = predictor.model(x[None], mask[None], torch.tensor([len(x)]))[0].numpy()
        decoder = make_decoder(predictor.decoder_settings)
        for logits in full:
            decoder.step(log_probabilities(logits))
        predictor.reset()
        streamed = []
        for frame in row["hands"]:
            text, logits = predictor.step(frame)
            streamed.append(logits)
        streamed = np.stack(streamed)
        np.testing.assert_allclose(streamed, full, atol=2e-5, rtol=1e-5)
        assert text == decoder.text
        # Replay the same word after processing another word, with reset.
        results.append(
            dict(
                id=key,
                target=row["label"],
                frames=len(x),
                missing_frames=sum(not frame for frame in row["hands"]),
                max_abs_logit_difference=float(np.max(np.abs(streamed - full))),
                full_text=decoder.text,
                stream_text=text,
            )
        )
    predictor.reset()
    repeated = np.stack([predictor.step(frame)[1] for frame in found[ids[0]]["hands"]])
    x, mask = hand_features(found[ids[0]]["hands"])
    with torch.inference_mode():
        reference = predictor.model(x[None], mask[None], torch.tensor([len(x)]))[0].numpy()
    np.testing.assert_allclose(repeated, reference, atol=2e-5, rtol=1e-5)
    # Missing input advances recurrent state exactly as a zero-mask training frame.
    artificial = [[], found[ids[0]]["hands"][1], [], found[ids[0]]["hands"][1]]
    predictor.reset()
    step = np.stack([predictor.step(frame)[1] for frame in artificial])
    x, mask = hand_features(artificial)
    with torch.inference_mode():
        whole = predictor.model(x[None], mask[None], torch.tensor([4]))[0].numpy()
    np.testing.assert_allclose(step, whole, atol=2e-5, rtol=1e-5)
    report = dict(
        checkpoint_sha256=predictor.checkpoint_sha256,
        decoder_config_sha256=sha256(Path("configs/sequence_decoder.json")),
        tolerance=dict(atol=2e-5, rtol=1e-5),
        results=results,
        reset_between_words_verified=True,
        missing_frames_advance_state_without_forced_blank=True,
        test_inference=False,
        camera_trial=False,
    )
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
