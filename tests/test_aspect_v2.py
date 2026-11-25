"""Aspect-aware geometry, checkpoint dispatch, and strict dimension requirements."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from fingerspelling.features import FEATURE_V1, FEATURE_V2, LABELS, prepare_features
from fingerspelling.models.mlp import StaticMLP, StaticPredictor


def points():
    return np.random.default_rng(13).uniform(20, 180, (21, 3)).astype(np.float32)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_same_pixel_geometry_across_different_canvas_sizes(hand):
    pixel = points()
    results = []
    for width, height in [(640, 480), (1280, 720), (800, 800)]:
        xyz = pixel / np.array([width, height, width], dtype=np.float32)
        results.append(
            prepare_features(
                xyz, hand, feature_version=FEATURE_V2, frame_width=width, frame_height=height
            )
        )
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=2e-7)
    a = prepare_features(pixel / [640, 480, 640], hand)
    b = prepare_features(pixel / [800, 800, 800], hand)
    assert not np.allclose(a, b)


def test_square_is_bit_identical_and_input_not_mutated():
    xyz = points() / 200
    original = xyz.copy()
    for hand in ("left", "right"):
        np.testing.assert_array_equal(
            prepare_features(xyz, hand),
            prepare_features(
                xyz, hand, feature_version=FEATURE_V2, frame_width=200, frame_height=200
            ),
        )
    np.testing.assert_array_equal(xyz, original)


@pytest.mark.parametrize(
    "width,height",
    [
        (None, None),
        (640, None),
        (None, 480),
        (0, 480),
        (-1, 480),
        (640, 0),
        (True, 480),
        (640.0, 480),
    ],
)
def test_v2_rejects_missing_or_invalid_dimensions(width, height):
    with pytest.raises(ValueError, match="documented"):
        prepare_features(
            points(), "left", feature_version=FEATURE_V2, frame_width=width, frame_height=height
        )


def test_checkpoint_dispatch_and_legacy_support(tmp_path):
    model = StaticMLP().eval()
    xyz = points() / 200
    for version in (FEATURE_V1, FEATURE_V2):
        path = tmp_path / (version + ".pt")
        torch.save(
            dict(state_dict=model.state_dict(), labels=LABELS, feature_version=version), path
        )
        predictor = StaticPredictor(path)
        kwargs = {} if version == FEATURE_V1 else dict(frame_width=640, frame_height=480)
        actual = predictor.probabilities(xyz, "left", **kwargs)
        features = prepare_features(xyz, "left", feature_version=version, **kwargs)
        with torch.inference_mode():
            expected = model(torch.from_numpy(features).reshape(1, 63)).softmax(-1).numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (1, 24)
        if version == FEATURE_V2:
            with pytest.raises(ValueError):
                predictor.probabilities(xyz, "left")


def test_training_uses_verified_dimensions_and_shared_v2_features():
    spec = importlib.util.spec_from_file_location(
        "train_static", Path(__file__).parents[1] / "scripts/train_static.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    xyz = points() / 200
    row = ("M/a.jpg", "M", "digest", xyz.tobytes(), "left")
    dimensions = {"M/a.jpg": dict(frame_width=640, frame_height=480, pixel_sha256="digest")}
    x, y = module.tensors([row], FEATURE_V2, dimensions)
    np.testing.assert_array_equal(
        x.numpy()[0],
        prepare_features(
            xyz, "left", feature_version=FEATURE_V2, frame_width=640, frame_height=480
        ),
    )
    assert y.item() == LABELS.index("M")
    with pytest.raises(ValueError):
        module.tensors([row], FEATURE_V2, {})
    dimensions["M/a.jpg"]["pixel_sha256"] = "stale"
    with pytest.raises(ValueError):
        module.tensors([row], FEATURE_V2, dimensions)
