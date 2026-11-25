"""Strict epoch-zero gate and macro clip scoring for final development selection."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
spec_previous = importlib.util.spec_from_file_location(
    "finetune_static", SCRIPTS / "finetune_static.py"
)
previous = importlib.util.module_from_spec(spec_previous)
spec_previous.loader.exec_module(previous)
sys.modules["finetune_static"] = previous
spec = importlib.util.spec_from_file_location("final_static", SCRIPTS / "final_static.py")
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


def score(accuracy, nll):
    return dict(macro_accuracy=accuracy, macro_nll=nll)


def test_epoch_zero_requires_strict_accuracy_gain():
    baseline = score(0.9, 0.5)
    assert not final.accept_candidate(score(0.9, 0.1), baseline, baseline)
    assert not final.accept_candidate(score(0.89, 0.01), baseline, baseline)
    assert final.accept_candidate(score(0.91, 0.9), baseline, baseline)


def test_nll_only_breaks_improved_accuracy_ties():
    baseline = score(0.9, 0.5)
    best = score(0.92, 0.4)
    assert final.accept_candidate(score(0.92, 0.3), best, baseline)
    assert not final.accept_candidate(score(0.92, 0.4), best, baseline)
    assert not final.accept_candidate(score(0.91, 0.1), best, baseline)
    assert final.accept_candidate(score(0.93, 0.9), best, baseline)


def test_clip_mean_probabilities_and_macro_not_micro():
    # First clip: 2 frames vote A, but mean probabilities favor B.
    logits = torch.tensor([[0.1, 0.0], [0.1, 0.0], [-8.0, 8.0]])
    clips = [
        dict(clip_id="a", target="A", target_index=0, valid_frames=3, x=logits),
        dict(clip_id="b", target="B", target_index=1, valid_frames=1, x=torch.tensor([[0.0, 8.0]])),
        dict(clip_id="c", target="B", target_index=1, valid_frames=1, x=torch.tensor([[0.0, 8.0]])),
    ]
    result = previous.camera_score(torch.nn.Identity(), clips, ["A", "B"])
    assert result["clips"][0]["prediction"] == "B"
    assert result["correct"] == 2
    assert result["macro_accuracy"] == 0.5
    expected = (
        result["clips"][0]["nll"] + (result["clips"][1]["nll"] + result["clips"][2]["nll"]) / 2
    ) / 2
    assert result["macro_nll"] == pytest.approx(expected)
