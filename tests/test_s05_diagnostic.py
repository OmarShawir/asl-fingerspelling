"""Reject incorrectly paired clips and incorrectly paired diagnostic frames."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "diagnose_s05", Path(__file__).parents[1] / "scripts/diagnose_s05.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def row(key, target, prediction):
    return dict(clip_id=key, target=target, prediction=prediction, valid_frames=3, top3=[])


def test_join_by_id_not_position_and_all_outcomes():
    before = [row("a", "K", "K"), row("b", "M", "N"), row("c", "V", "V"), row("d", "Q", "P")]
    after = [row("d", "Q", "P"), row("c", "V", "W"), row("b", "M", "M"), row("a", "K", "K")]
    result = diagnostic.join_clips(before, after)
    assert [r["category"] for r in result] == [
        "both_correct",
        "improved",
        "regressed",
        "both_wrong",
    ]


def test_join_rejects_duplicate_missing_and_changed_target():
    a = row("a", "K", "K")
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.join_clips([a, a], [a])
    with pytest.raises(ValueError, match="sets differ"):
        diagnostic.join_clips([a], [])
    with pytest.raises(ValueError, match="Targets differ"):
        diagnostic.join_clips([a], [row("a", "V", "K")])


def test_manifest_requires_exact_frame_and_timestamp():
    times = np.array([0, 33, 67])
    assert diagnostic.validate_frame(dict(frame_index=2, t_ms=67), times) == 2
    with pytest.raises(ValueError, match="timestamp"):
        diagnostic.validate_frame(dict(frame_index=1, t_ms=67), times)
    for index in (-1, 3, 1.5):
        with pytest.raises(ValueError, match="index"):
            diagnostic.validate_frame(dict(frame_index=index, t_ms=0), times)
