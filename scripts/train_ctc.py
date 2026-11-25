"""First offline ChicagoFSWild GRU/CTC pilot. No test inference or static weights."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fingerspelling.ctc_decode import greedy_decode, transcript_metrics
from fingerspelling.models.gru_ctc import SequenceGRUCTC
from fingerspelling.sequence_data import (
    FEATURE_VERSION,
    VOCABULARY,
    collate_sequences,
    ctc_minimum,
    load_sequences,
    sha256,
    verify_sources,
)

try:
    import resource
except ImportError:  # Windows: optional telemetry must not prevent training.
    resource = None


def peak_process_rss_bytes():
    """Return peak process RSS in bytes, or None when unavailable."""
    if resource is None:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


LOSS = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def loader(rows, batch_size, shuffle=False, generator=None):
    return DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_sequences,
        generator=generator,
    )


def ctc_loss(logits, batch):
    losses = LOSS(
        logits.log_softmax(-1).transpose(0, 1),
        batch["targets"],
        batch["lengths"],
        batch["target_lengths"],
    )
    if not torch.isfinite(losses).all():
        raise ValueError("Nonfinite CTC loss; never silently zeroed")
    return (losses / batch["target_lengths"]).mean()


@torch.inference_mode()
def evaluate(model, rows, batch_size, output=None):
    model.eval()
    predictions, raw = [], []
    loss_sum, eligible, skipped = 0.0, 0, []
    for batch in loader(rows, batch_size):
        logits = model(batch["x"], batch["mask"], batch["lengths"])
        for i, row in enumerate(batch["rows"]):
            frame_logits = logits[i, : len(row["x"])]
            prediction = greedy_decode(frame_logits.argmax(-1).tolist())
            predictions.append(
                dict(
                    id=row["id"],
                    signer=row["signer"],
                    target=row["text"],
                    prediction=prediction,
                    frames=len(row["x"]),
                    ctc_minimum=ctc_minimum(row["text"]),
                )
            )
            if output:
                raw.append(frame_logits.numpy())
        indices = [
            i for i, row in enumerate(batch["rows"]) if len(row["x"]) >= ctc_minimum(row["text"])
        ]
        skipped.extend(
            row["id"] for row in batch["rows"] if len(row["x"]) < ctc_minimum(row["text"])
        )
        if indices:
            aligned = collate_sequences([batch["rows"][i] for i in indices])
            loss = ctc_loss(logits[indices], aligned)
            loss_sum += float(loss) * len(indices)
            eligible += len(indices)
    metrics = transcript_metrics(predictions)
    metrics.update(
        ctc_loss=loss_sum / eligible if eligible else None,
        ctc_loss_denominator=eligible,
        ctc_loss_skipped=len(skipped),
        ctc_loss_skipped_ids=skipped,
        loss_definition="mean per-clip CTC NLL / target length, alignable clips only",
        j_subset=transcript_metrics([r for r in predictions if "j" in r["target"]]),
        z_subset=transcript_metrics([r for r in predictions if "z" in r["target"]]),
    )
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output.with_suffix(".json"), dict(metrics=metrics, predictions=predictions))
        with output.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        np.savez_compressed(
            str(output) + "_logits.npz",
            logits=np.concatenate(raw),
            lengths=np.array([len(x) for x in raw]),
            ids=np.array([r["id"] for r in predictions]),
        )
    return metrics


def checkpoint(path, model, config, sources, epoch):
    torch.save(
        dict(
            state_dict=model.state_dict(),
            architecture=model.config,
            feature_version=FEATURE_VERSION,
            vocabulary=VOCABULARY,
            blank_index=0,
            config=config,
            source_hashes=sources,
            epoch=epoch,
        ),
        path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/chicagofswild"))
    parser.add_argument("--audit-dir", type=Path, default=Path("asl_sequence_audit"))
    parser.add_argument("--output", type=Path, default=Path("results/sequence/gru_ctc_pilot_v1"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/sequence/gru_ctc_pilot_v1.pt")
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--debug-seconds", type=float, default=120)
    parser.add_argument("--debug-steps", type=int, default=1500)
    parser.add_argument("--debug-seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise SystemExit("Refusing to overwrite a sequence experiment")
    if min(args.epochs, args.seconds, args.debug_seconds, args.debug_steps) <= 0:
        raise ValueError("Positive compute bounds required")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    sources = verify_sources(args.cache_dir, args.audit_dir)
    static = Path("models/mlp_static_final.pt")
    static_sha = sha256(static)
    train = load_sequences(args.cache_dir, args.audit_dir, "train")
    validation = load_sequences(args.cache_dir, args.audit_dir, "validation")
    if (len(train), len(validation)) != (5433, 978):
        raise ValueError("Wrong pilot membership")
    args.output.mkdir(parents=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = dict(
        seed=42,
        threads=4,
        optimizer="Adam",
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=32,
        gradient_clip_norm=5.0,
        max_epochs=args.epochs,
        pilot_seconds_budget=args.seconds,
        patience=5,
        debug_steps=args.debug_steps,
        debug_seconds=args.debug_seconds,
        debug_learning_rate=0.003,
        debug_seed=args.debug_seed,
        debug_selection="first 16 unique short training labels sorted by frames/id, 10-35 frames, 2-5 characters, hand present in first frame; debugging only",
        feature_version=FEATURE_VERSION,
        train_count=len(train),
        validation_count=len(validation),
        selection="lowest validation micro CER across all 978 clips, including epoch 0",
        decoder="greedy CTC, no language model",
        no_time_downsampling=True,
        test_inference=False,
        static_checkpoint_sha256=static_sha,
        library_versions=dict(torch=str(torch.__version__), numpy=np.__version__),
    )
    write_json(args.output / "config.json", config)
    write_json(args.output / "source_hashes.json", sources)
    write_json(
        args.output / "vocabulary.json",
        {"blank": 0, "characters": {c: i + 1 for i, c in enumerate(VOCABULARY)}},
    )
    print(f"Loaded {len(train)} train, {len(validation)} validation; no test decoding.", flush=True)

    # Fixed training-only short subset; no dev labels used to choose debug examples.
    tiny, texts = [], set()
    for row in sorted(train, key=lambda r: (len(r["x"]), r["id"])):
        if (
            2 <= len(row["text"]) <= 5
            and 10 <= len(row["x"]) <= 35
            and row["text"] not in texts
            and bool(row["mask"][0].any())
        ):
            tiny.append(row)
            texts.add(row["text"])
        if len(tiny) == 16:
            break
    if len(tiny) != 16:
        raise ValueError("Cannot form debug subset")
    write_json(args.output / "debug_ids.json", [r["id"] for r in tiny])
    torch.manual_seed(args.debug_seed)
    debug = SequenceGRUCTC(dropout=0.0)
    debug_optimizer = torch.optim.Adam(debug.parameters(), lr=0.003)
    debug_batch = collate_sequences(tiny)
    debug_history = []
    debug_started = time.monotonic()
    debug_passed = False
    for step in range(1, args.debug_steps + 1):
        debug.train()
        debug_optimizer.zero_grad()
        logits = debug(debug_batch["x"], debug_batch["mask"], debug_batch["lengths"])
        loss = ctc_loss(logits, debug_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(debug.parameters(), 5.0)
        debug_optimizer.step()
        if step == 1 or step % 25 == 0:
            metrics = evaluate(debug, tiny, 16)
            debug_history.append(dict(step=step, loss=float(loss.detach()), metrics=metrics))
            print(
                f"Debug {step}: CER {metrics['cer']:.4f}, exact {metrics['exact']}/16", flush=True
            )
            if metrics["exact"] == 16:
                debug_passed = True
                break
        if time.monotonic() - debug_started >= args.debug_seconds:
            break
    debug_metrics = evaluate(debug, tiny, 16, args.output / "debug_predictions")
    debug_passed = debug_metrics["exact"] == 16
    write_json(
        args.output / "debug_report.json",
        dict(
            passed=debug_passed,
            metrics=debug_metrics,
            history=debug_history,
            steps=step,
            elapsed_seconds=time.monotonic() - debug_started,
            interpretation="Training-only overfit debug, not generalization",
        ),
    )
    if not debug_passed:
        raise SystemExit("Debug subset did not reach 16/16; pilot not started")
    del debug, debug_optimizer

    # Start the pilot from an independent random initialization, NOT the overfit weights.
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    model = SequenceGRUCTC()
    config["architecture"] = model.config
    config["parameters"] = sum(p.numel() for p in model.parameters())
    write_json(args.output / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    baseline = evaluate(model, validation, 32)
    history = [dict(epoch=0, validation=baseline, training_loss=None)]
    best_cer, best_epoch = baseline["cer"], 0
    checkpoint(args.checkpoint, model, config, sources, 0)
    pilot_started = time.monotonic()
    timed_out = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, samples = 0.0, 0
        epoch_started = time.monotonic()
        for batch in loader(train, 32, shuffle=True, generator=generator):
            optimizer.zero_grad()
            logits = model(batch["x"], batch["mask"], batch["lengths"])
            loss = ctc_loss(logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses += float(loss.detach()) * len(batch["rows"])
            samples += len(batch["rows"])
            if time.monotonic() - pilot_started >= args.seconds:
                timed_out = True
                break
        metrics = evaluate(model, validation, 32)
        improved = metrics["cer"] < best_cer
        history.append(
            dict(
                epoch=epoch,
                training_loss=losses / samples,
                training_samples=samples,
                complete_epoch=samples == len(train),
                validation=metrics,
                elapsed_seconds=time.monotonic() - epoch_started,
                selected=improved,
            )
        )
        if improved:
            best_cer, best_epoch = metrics["cer"], epoch
            checkpoint(args.checkpoint, model, config, sources, epoch)
        write_json(args.output / "history.json", history)
        print(
            f"Pilot {epoch}: train loss {losses / samples:.4f}; val CER {metrics['cer']:.4f}, exact {metrics['exact']}/978, empty {metrics['empty']}; selected={improved}",
            flush=True,
        )
        if timed_out or epoch - best_epoch >= config["patience"]:
            break
    selected = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    reloaded = SequenceGRUCTC(**selected["architecture"])
    reloaded.load_state_dict(selected["state_dict"])
    final = evaluate(reloaded, validation, 32, args.output / "validation_predictions")
    train_metrics = evaluate(reloaded, train, 32)
    if sha256(static) != static_sha:
        raise ValueError("Static checkpoint changed")
    if not all(sha256(p) == h for p, h in sources.items()):
        raise ValueError("Source changed during experiment")
    write_json(
        args.output / "report.json",
        dict(
            best_epoch=best_epoch,
            epochs_run=len(history) - 1,
            budget_reached=timed_out,
            validation=final,
            training=train_metrics,
            debug=debug_metrics,
            initial_validation=baseline,
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=sha256(args.checkpoint),
            elapsed_seconds=time.monotonic() - started,
            pilot_elapsed_seconds=time.monotonic() - pilot_started,
            peak_process_rss_bytes=peak_process_rss_bytes(),
            test_inference=False,
            sources_unchanged=True,
            static_unchanged=True,
            interpretation="Alphabet-plus-space offline feasibility pilot; validation used for selection. No test, streaming or live performance claim.",
        ),
    )
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
