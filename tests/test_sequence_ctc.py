"""Sequence shape, missingness, causality, set invariance and CTC contracts."""

import numpy as np
import pytest
import torch

from fingerspelling.ctc_decode import edit_distance, greedy_decode, transcript_metrics
from fingerspelling.models.gru_ctc import SequenceGRUCTC
from fingerspelling.sequence_data import (
    collate_sequences,
    ctc_minimum,
    encode,
    hand_features,
    load_sequences,
)


def example(key, frames, text="all"):
    x, mask = hand_features(frames)
    return dict(id=key, signer="synthetic", text=text, target=encode(text), x=x, mask=mask)


def hand():
    return np.random.default_rng(42).normal(size=(21, 3)).tolist()


def test_features_keep_missing_time_and_global_translation():
    points = hand()
    x, mask = hand_features([points, [], points + points])
    assert x.shape == (3, 2, 67)
    assert mask.tolist() == [[True, False], [False, False], [True, True]]
    assert torch.all(x[1] == 0)
    moved, _ = hand_features([(np.array(points) + [0.2, 0.1, 0.0]).tolist()])
    torch.testing.assert_close(x[0, 0, :63], moved[0, 0, :63])
    torch.testing.assert_close(moved[0, 0, 63:66] - x[0, 0, 63:66], torch.tensor([0.2, 0.1, 0.0]))
    with pytest.raises(ValueError):
        hand_features([points[:20]])
    points[0][0] = float("nan")
    with pytest.raises(ValueError):
        hand_features([points])


def test_swap_and_lone_block_invariance():
    torch.manual_seed(7)
    model = SequenceGRUCTC(hidden_size=16, layers=1).eval()
    x = torch.randn(2, 5, 2, 67)
    mask = torch.tensor(
        [[[True, True], [True, False], [False, False], [False, True], [True, True]]] * 2
    )
    lengths = torch.tensor([5, 4])
    with torch.inference_mode():
        a = model(x, mask, lengths)
        b = model(x.flip(2), mask.flip(2), lengths)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    poisoned = x.clone()
    poisoned[~mask] = float("nan")
    torch.testing.assert_close(
        model.frame_embeddings(x, mask), model.frame_embeddings(poisoned, mask)
    )


def test_padding_causality_and_variable_lengths():
    model = SequenceGRUCTC(hidden_size=16, layers=1).eval()
    a = example("a", [hand()] * 4)
    b = example("b", [hand()] * 7, "hello")
    batch = collate_sequences([a, b])
    logits = model(batch["x"], batch["mask"], batch["lengths"])
    solo = collate_sequences([a])
    alone = model(solo["x"], solo["mask"], solo["lengths"])
    torch.testing.assert_close(logits[0, :4], alone[0], atol=1e-6, rtol=1e-5)
    batch["x"][1, 4:] += 100
    modified = model(batch["x"], batch["mask"], batch["lengths"])
    torch.testing.assert_close(logits[1, :4], modified[1, :4], atol=1e-6, rtol=1e-5)
    assert logits.shape == (2, 7, 28)


def test_blank_repeat_space_and_scope():
    a, ell = encode("al").tolist()
    assert greedy_decode([0, a, a, ell, ell, 0, ell, 0]) == "all"
    assert greedy_decode([0, 0, 0]) == ""
    assert greedy_decode(encode("a a")) == "a a"
    assert ctc_minimum("all") == 4
    assert ctc_minimum("emmett") == 8
    with pytest.raises(ValueError):
        encode("a@b")
    with pytest.raises(ValueError):
        encode("ABC")


def test_ctc_short_loss_is_not_silently_zeroed():
    logits = torch.zeros(3, 1, 28).log_softmax(-1)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        logits, encode("all"), torch.tensor([3]), torch.tensor([3])
    )
    assert torch.isinf(loss)
    finite = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        torch.zeros(4, 1, 28).log_softmax(-1), encode("all"), torch.tensor([4]), torch.tensor([3])
    )
    assert torch.isfinite(finite)


def test_cer_includes_empty_and_long_predictions():
    rows = [dict(target="all", prediction=""), dict(target="a", prediction="abc")]
    metrics = transcript_metrics(rows)
    assert metrics["cer"] == 5 / 4
    assert metrics["empty_rate"] == 0.5
    assert metrics["sequences"] == 2
    assert edit_distance("kitten", "sitting") == 3


def test_development_loader_refuses_test(tmp_path):
    with pytest.raises(ValueError, match="refuses"):
        load_sequences(tmp_path, tmp_path, "test")


def test_validation_short_clips_remain_in_metrics(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_ctc", __import__("pathlib").Path(__file__).parents[1] / "scripts/train_ctc.py"
    )
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    rows = [example("short", [hand()] * 3, "all"), example("ok", [hand()] * 5, "all")]
    model = SequenceGRUCTC(hidden_size=16, layers=1)
    result = training.evaluate(model, rows, 2, tmp_path / "predictions")
    assert result["sequences"] == 2
    assert result["ctc_loss_denominator"] == 1
    assert result["ctc_loss_skipped_ids"] == ["short"]
    assert np.isfinite(result["ctc_loss"])
    saved = np.load(tmp_path / "predictions_logits.npz", allow_pickle=False)
    assert saved["lengths"].tolist() == [3, 5]
    assert saved["logits"].shape == (8, 28)


def test_missing_frame_gradients_are_finite():
    model = SequenceGRUCTC(hidden_size=16, layers=1)
    batch = collate_sequences([example("a", [[], hand(), [], hand(), hand()], "all")])
    logits = model(batch["x"], batch["mask"], batch["lengths"])
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        logits.log_softmax(-1).transpose(0, 1),
        batch["targets"],
        batch["lengths"],
        batch["target_lengths"],
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)



def test_training_import_without_unix_resource(monkeypatch):
    import builtins
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    original_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("resource is unavailable on Windows")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    spec = importlib.util.spec_from_file_location(
        "portable_train_ctc", Path(__file__).parents[1] / "scripts/train_ctc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.peak_process_rss_bytes() is None
    module.resource = SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=123)
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert module.peak_process_rss_bytes() == 123 * 1024
    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert module.peak_process_rss_bytes() == 123
