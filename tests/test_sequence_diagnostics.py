"""Synthetic full-vocabulary logits and deterministic edit attribution."""

import importlib.util
from pathlib import Path

from fingerspelling.sequence_diagnostics import align, detailed_metrics

spec = importlib.util.spec_from_file_location(
    "diagnose_ctc", Path(__file__).parents[1] / "scripts/diagnose_ctc.py"
)
diagnosis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)


def test_synthetic_logits_all_symbols_repeats_and_literal_spaces():
    results = diagnosis.synthetic_checks()
    assert results["jazz"] == "jazz"
    assert results["all"] == "all"
    assert results["a b"] == "a b"
    assert results[" a b "] == " a b "
    assert results["  "] == "  "
    assert len(results["abcdefghijklmnopqrstuvwxyz "]) == 27


def test_alignment_ties_prefer_diagonal_substitution():
    assert align("ab", "ba") == [("substitution", "a", "b"), ("substitution", "b", "a")]
    assert align("aa", "a") == [("deletion", "a", ""), ("match", "a", "a")]


def test_edit_counts_and_emission_is_not_correctness():
    rows = [
        dict(id="a", target="jazz", prediction=""),
        dict(id="b", target="all", prediction="jallq"),
        dict(id="c", target="a b", prediction="axb"),
    ]
    metrics, clips = detailed_metrics(rows)
    assert metrics["deletions"] == 4
    assert metrics["insertions"] == 2
    assert metrics["substitutions"] == 1
    assert metrics["character_edits"] == 7
    assert metrics["focus"]["j"] == dict(target=1, emitted=1, aligned_correct=0)
    assert metrics["focus"]["q"] == dict(target=0, emitted=1, aligned_correct=0)
    assert metrics["predicted_characters"] == 8
    assert len(clips) == 3
