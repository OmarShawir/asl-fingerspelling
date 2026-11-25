"""Long clips cannot dominate; each source gets exactly half of a training batch."""

import pytest
import torch

from fingerspelling.adaptation import assert_disjoint, balanced_batch, camera_frame_weights


def test_equal_classes_and_clips_despite_different_lengths():
    targets = [0] * 101 + [1] * 5
    ids = ["short"] + ["long"] * 100 + ["other"] * 5
    weights = camera_frame_weights(targets, ids)
    assert float(weights.sum()) == pytest.approx(1)
    assert float(weights[0]) == pytest.approx(0.25)
    assert float(weights[1:101].sum()) == pytest.approx(0.25)
    assert float(weights[101:].sum()) == pytest.approx(0.5)
    assert torch.all(weights[1:101] == weights[1])


def test_source_counts_and_seed_reproducibility():
    a = torch.ones(20)
    b = torch.ones(3)
    x = balanced_batch(a, b, 128, torch.Generator().manual_seed(42))
    y = balanced_batch(a, b, 128, torch.Generator().manual_seed(42))
    assert len(x[0]) == len(x[1]) == 128
    assert all(torch.equal(i, j) for i, j in zip(x, y))
    assert x[0].max() < 20 and x[1].max() < 3


def test_overlap_and_invalid_clip_metadata_rejected():
    assert_disjoint(["s04-a"], ["s03-b"])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(["same"], ["same"])
    with pytest.raises(ValueError, match="multiple labels"):
        camera_frame_weights([0, 1], ["same", "same"])
    with pytest.raises(ValueError):
        camera_frame_weights([], [])
    with pytest.raises(ValueError):
        camera_frame_weights([0, 1], ["a"])
