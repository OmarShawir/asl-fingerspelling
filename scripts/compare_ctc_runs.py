"""Select between the longer run and untouched pilot; report all development edits."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from diagnose_ctc import diagnose
from train_ctc import evaluate
from fingerspelling.models.gru_ctc import SequenceGRUCTC
from fingerspelling.sequence_data import load_sequences, sha256
from fingerspelling.sequence_diagnostics import detailed_metrics


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_model(path):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    model = SequenceGRUCTC(**bundle["architecture"])
    model.load_state_dict(bundle["state_dict"])
    return model.eval()


def save_details(prefix, output):
    rows = json.loads(prefix.with_suffix(".json").read_text())["predictions"]
    metrics, clips = detailed_metrics(rows)
    dump(output.with_suffix(".json"), metrics)
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clips[0]))
        writer.writeheader()
        writer.writerows(clips)
    return metrics


def reproducibility(old, new):
    a = json.loads((old / "history.json").read_text())
    b = json.loads((new / "history.json").read_text())
    results = []
    for x, y in zip(a, b):
        assert x["epoch"] == y["epoch"]
        pairs = {
            k: (x["validation"][k], y["validation"][k])
            for k in ("cer", "character_edits", "exact", "empty", "ctc_loss")
        }
        if x["training_loss"] is not None:
            pairs["training_loss"] = (x["training_loss"], y["training_loss"])
        results.append(
            dict(
                epoch=x["epoch"],
                exact_equal=all(v == w for v, w in pairs.values()),
                within_tolerance=all(
                    np.isclose(v, w, rtol=1e-6, atol=1e-7) for v, w in pairs.values()
                ),
                deltas={k: float(w - v) for k, (v, w) in pairs.items()},
            )
        )
    ac = json.loads((old / "config.json").read_text())
    bc = json.loads((new / "config.json").read_text())
    changes = {
        k: dict(before=ac.get(k), after=bc.get(k))
        for k in ac.keys() | bc.keys()
        if ac.get(k) != bc.get(k)
    }
    assert set(changes) == {"max_epochs", "pilot_seconds_budget"}
    result = dict(
        configuration_changes=changes,
        epochs=results,
        all_exact=all(r["exact_equal"] for r in results),
        all_within_tolerance=all(r["within_tolerance"] for r in results),
        tolerance=dict(rtol=1e-6, atol=1e-7),
    )
    dump(new / "reproducibility.json", result)
    return result


def write_report(out, comparison, candidate):
    before = comparison["pilot_validation"]
    after = comparison["selected_validation"]
    lines = [
        "# GRU + CTC longer-budget experiment — 2026-11-21",
        "",
        f"Selected **{comparison['selected_origin']}**, epoch **{comparison['selected_epoch']}**. "
        f"Validation CER: **{before['cer']:.4%} → {after['cer']:.4%}**.",
        "",
        "## Pre-training diagnosis",
        "",
        "Pilot numbers independently reconstructed from CSV and raw logits: 978 clips, "
        "4,967 target characters, 3,703 emitted characters, 2,513 edits, 110 exact transcripts, "
        "35 empty outputs. Edit allocation: 1,395 deletions, 987 substitutions, 131 insertions. "
        "J/Z/Q/space had zero frame-argmax occurrences, before greedy decoding.",
        "",
        "Blank is index 0 and literal space index 27. Synthetic logits successfully decode all "
        "26 letters plus space, jazz, all, a b, leading/trailing and consecutive spaces. "
        "No mapping/decoder bug was found, so no model, feature, decoder or optimization fix was made.",
        "",
        "## Controlled experiment",
        "",
        "One longer run from random initialization, not warm-started from pilot weights. "
        "Only max_epochs 15→60 and loop budget 600→900 seconds changed. "
        "Seed 42, Adam lr=0.001, batch 32, dropout 0.1, patience 5, full temporal resolution, "
        "architecture, features, clip memberships, gradient clipping and greedy decoding are identical. "
        "No augmentation, class balancing or scheduler. The same training-only debug protocol ran "
        "before an independent seed reset; its weights were not used.",
        "",
        "Original pilot remains eligible. Final selection takes minimum validation CER across "
        "the longer run (including epoch 0) and the pilot; ties retain the pilot. "
        "Patience within the longer run retains the original run-local definition. "
        "Neither test predictions nor test-based parameter choices were made.",
        "",
        f"Reproducibility: {len(comparison['reproducibility']['epochs'])} shared epoch records "
        f"(including epoch 0); exact metric/loss agreement: "
        f"{comparison['reproducibility']['all_exact']}. See reproducibility.json.",
        "",
        f"Executed epochs: {candidate['epochs_run']}; time budget reached: "
        f"{candidate['budget_reached']}. Successful training invocation: "
        f"{candidate['elapsed_seconds']:.2f} seconds; peak process RSS "
        f"{candidate['peak_process_rss_bytes'] / 1024**2:.2f} MiB. "
        "Elapsed invocation includes setup/debug/final scoring outside the loop budget.",
        "",
        "## Text metrics",
        "",
        "| Split / model | CER | Exact | Empty | Deletions | Substitutions | Insertions | Output characters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [
        ("Validation / pilot", before),
        ("Validation / selected", after),
        ("Training / pilot", comparison["pilot_training"]),
        ("Training / selected", comparison["selected_training"]),
    ]
    for name, m in groups:
        lines.append(
            f"| {name} | {m['cer']:.4%} | {m['exact']}/{m['sequences']} | "
            f"{m['empty']}/{m['sequences']} | {m['deletions']} | {m['substitutions']} | "
            f"{m['insertions']} | {m['predicted_characters']} |"
        )
    lines += [
        "",
        "Training metrics are not held-out performance. All 978 validation clips remain in "
        "CER/exact/empty metrics. Finite target-length-normalized CTC loss uses the 976 "
        "alignable validation clips; emmett (7<8) and all (3<4) remain in text metrics. "
        "zero_infinity=False; no target-dependent stretching.",
        "",
        "## Validation J/Z/Q/space",
        "",
        "| Symbol | Target occurrences | Pilot emitted | Selected emitted | Selected aligned correct | Selected frame argmax |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for char in "jzq ":
        a, b = before["focus"][char], after["focus"][char]
        lines.append(
            f"| {'space' if char == ' ' else char.upper()} | {b['target']} | "
            f"{a['emitted']} | {b['emitted']} | {b['aligned_correct']} | {b['frame_argmax']} |"
        )
    lines += [
        "",
        "Emitting a symbol does not prove correct recognition. Aligned-correct counts "
        "refer to transcript matching under the specified tie rule, not isolated gesture recognition "
        "or verified frame alignment.",
        "",
        "## Edit alignment rule",
        "",
        after["alignment_tie_rule"],
        "",
        "The overall Levenshtein distance is invariant "
        "to ties; substitution/deletion/insertion allocations and per-symbol correct counts can vary. "
        "Both models use the same rule. The output-length difference equals insertions minus deletions.",
        "",
        "## Files and reproduction",
        "",
        "- comparison.json/csv: both training and validation metrics.",
        "- pilot_diagnostic/ and long_diagnostic/: per-clip edit counts and pre-decoder argmax counts.",
        "- training_edits.* and pilot_training_edits.*: all training clip edit counts.",
        "- validation_predictions.* and training_predictions.*: raw transcripts and local unpadded logits.",
        "- candidate_report.json: unmodified result of the longer run before pilot eligibility comparison.",
        "- history.json/config.json/source_hashes.json/vocabulary.json: original trainer outputs.",
        "- protected_inputs_sha256.json: pre-run hashes of static/pilot/recording files.",
        "",
        f"Selected checkpoint: {comparison['checkpoint']}",
        f"SHA-256: {comparison['checkpoint_sha256']}",
        f"Pilot SHA-256: {comparison['pilot_checkpoint_sha256']}",
        "",
        "Commands (fresh result/checkpoint paths required for a new run):",
        "",
        "    .venv/bin/python scripts/train_ctc.py --epochs 60 --seconds 900 --output results/sequence/gru_ctc_long_v1 --checkpoint models/sequence/gru_ctc_long_v1.pt",
        "    .venv/bin/python scripts/compare_ctc_runs.py",
        "",
        "Validation was used for selection, not an independent final test. The same "
        "conversion/hand-order/aspect/FPS unknowns apply as in the pilot. This is an "
        "offline segmented-sequence experiment, with no new live or test claim.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/sequence/gru_ctc_pilot_v1"))
    parser.add_argument("--long", type=Path, default=Path("results/sequence/gru_ctc_long_v1"))
    args = parser.parse_args()
    out = args.long
    if (out / "comparison.json").exists():
        raise SystemExit("Refusing to overwrite completed comparison")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    old = json.loads((args.pilot / "report.json").read_text())
    new = json.loads((out / "report.json").read_text())
    repro = reproducibility(args.pilot, out)
    dump(out / "candidate_report.json", new)
    selected_origin = "long" if new["validation"]["cer"] < old["validation"]["cer"] else "pilot"
    output_checkpoint = Path(new["checkpoint"])
    if selected_origin == "pilot":
        shutil.copy2(
            output_checkpoint, output_checkpoint.with_name(output_checkpoint.stem + "_candidate.pt")
        )
        for path in out.glob("validation_predictions*"):
            shutil.copy2(path, path.with_name("candidate_" + path.name))
        shutil.copy2(old["checkpoint"], output_checkpoint)
        validation = load_sequences(
            Path("data/external/chicagofswild"), Path("asl_sequence_audit"), "validation"
        )
        model = load_model(output_checkpoint)
        new["validation"] = evaluate(model, validation, 32, out / "validation_predictions")
        new["best_epoch"] = old["best_epoch"]
        new["checkpoint_sha256"] = sha256(output_checkpoint)
    new["selected_origin"] = selected_origin
    old_validation = diagnose(
        args.pilot / "validation_predictions",
        args.pilot / "vocabulary.json",
        out / "pilot_diagnostic",
    )
    new_validation = diagnose(
        out / "validation_predictions", out / "vocabulary.json", out / "long_diagnostic"
    )
    train = load_sequences(Path("data/external/chicagofswild"), Path("asl_sequence_audit"), "train")
    old_train = evaluate(
        load_model(old["checkpoint"]), train, 32, out / "pilot_training_predictions"
    )
    selected_train = evaluate(
        load_model(output_checkpoint), train, 32, out / "training_predictions"
    )
    old_train_details = save_details(
        out / "pilot_training_predictions", out / "pilot_training_edits"
    )
    new_train_details = save_details(out / "training_predictions", out / "training_edits")
    new["training"] = selected_train
    dump(out / "report.json", new)
    comparison = dict(
        selected_origin=selected_origin,
        selected_epoch=new["best_epoch"],
        selection_rule="minimum validation CER across long-run epochs, epoch 0 and existing pilot; pilot retained on a tie",
        pilot_validation=old_validation,
        selected_validation=new_validation,
        pilot_training={**old_train, **old_train_details},
        selected_training={**selected_train, **new_train_details},
        reproducibility=repro,
        checkpoint=str(output_checkpoint),
        checkpoint_sha256=sha256(output_checkpoint),
        pilot_checkpoint_sha256=sha256(old["checkpoint"]),
        test_inference=False,
        interpretation="Validation is used for selection. Symbol emissions are not necessarily correct recognitions.",
    )
    dump(out / "comparison.json", comparison)
    write_report(out, comparison, json.loads((out / "candidate_report.json").read_text()))
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "model",
                "sequences",
                "cer",
                "exact",
                "empty",
                "deletions",
                "substitutions",
                "insertions",
                "predicted_characters",
                "j_emitted",
                "z_emitted",
                "q_emitted",
                "space_emitted",
            ]
        )
        for split, model, metrics in [
            ("validation", "pilot", old_validation),
            ("validation", selected_origin, new_validation),
            ("train", "pilot", old_train_details),
            ("train", selected_origin, new_train_details),
        ]:
            writer.writerow(
                [
                    split,
                    model,
                    *[
                        metrics[k]
                        for k in (
                            "sequences",
                            "cer",
                            "exact",
                            "empty",
                            "deletions",
                            "substitutions",
                            "insertions",
                            "predicted_characters",
                        )
                    ],
                    *[metrics["focus"][c]["emitted"] for c in "jzq "],
                ]
            )
    with (out / "character_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "symbol",
                "target",
                "pilot_emitted",
                "selected_emitted",
                "pilot_aligned_correct",
                "selected_aligned_correct",
            ]
        )
        for split, before, after in [
            ("validation", old_validation, new_validation),
            ("train", old_train_details, new_train_details),
        ]:
            for symbol in "abcdefghijklmnopqrstuvwxyz ":
                writer.writerow(
                    [
                        split,
                        symbol,
                        after["target_counts"].get(symbol, 0),
                        before["output_counts"].get(symbol, 0),
                        after["output_counts"].get(symbol, 0),
                        before["aligned_correct_counts"].get(symbol, 0),
                        after["aligned_correct_counts"].get(symbol, 0),
                    ]
                )
    print(
        "SELECTED",
        selected_origin,
        "epoch",
        new["best_epoch"],
        "validation CER",
        new_validation["cer"],
        "reproducible",
        repro["all_exact"],
    )


if __name__ == "__main__":
    main()
