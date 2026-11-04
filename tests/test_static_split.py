"""Exact duplicate images must never cross the validation boundary."""

import importlib.util
from pathlib import Path

from fingerspelling.features import LABELS

spec = importlib.util.spec_from_file_location(
    "train_static", Path(__file__).parents[1] / "scripts/train_static.py"
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_duplicate_and_conflicting_images_are_isolated():
    rows = []
    for label in LABELS:
        for i in range(10):
            rows.append((f"{label}/{i}.jpg", label, f"{label}-{i}", b"", "right"))
    rows.append(("copy.jpg", "A", "A-0", b"", "right"))
    rows.extend(
        [
            ("ambiguous-a.jpg", "A", "conflict", b"", "right"),
            ("ambiguous-b.jpg", "B", "conflict", b"", "right"),
        ]
    )
    train, val, stats = training.build_split(rows)
    train_hashes = {r[2] for r in train}
    val_hashes = {r[2] for r in val}
    assert not train_hashes & val_hashes
    assert "conflict" not in train_hashes | val_hashes
    assert len(train) + len(val) == 240
    assert {r[1] for r in train} == {r[1] for r in val} == set(LABELS)
    assert stats["conflicting_hashes_dropped"] == 1
    assert training.build_split(rows) == (train, val, stats)
