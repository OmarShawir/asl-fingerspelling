"""Invariants required when moving from image training to webcam inference."""

import numpy as np
import pytest

from fingerspelling.features import prepare_features


def hand_points():
    return np.random.default_rng(7).normal(size=(21, 3)).astype(np.float32)


def test_translation_scale_and_physical_hand_invariance():
    xyz = hand_points()
    original = prepare_features(xyz, "right")
    np.testing.assert_allclose(prepare_features(xyz * 3 + 4, "right"), original, atol=1e-6)
    mirrored = xyz.copy()
    mirrored[:, 0] *= -1
    np.testing.assert_allclose(prepare_features(mirrored, "left"), original, atol=1e-6)


def test_orientation_is_preserved_and_input_unchanged():
    xyz = hand_points()
    saved = xyz.copy()
    rotated = xyz.copy()
    rotated[:, :2] *= -1
    assert not np.allclose(prepare_features(rotated, "right"), prepare_features(xyz, "right"))
    prepare_features(xyz, "left")
    np.testing.assert_array_equal(saved, xyz)


def test_missing_and_degenerate_hands_are_rejected():
    with pytest.raises(ValueError):
        prepare_features(np.full((21, 3), np.nan), "left")
    with pytest.raises(ValueError):
        prepare_features(np.zeros((21, 3)), "right")
