import json

import pytest

from fingerspelling.config import read_config
from fingerspelling.word_builder import WordBuilder


def make():
    return WordBuilder.from_config(read_config()["static"])


def hold(b, letter, start, end, confidence=0.95):
    for t in range(start, end + 1, 50):
        b.update(t, letter, confidence, True)


def missing(b, start, end):
    for t in range(start, end + 1, 50):
        b.update(t)


def test_hold_once_and_different_letter_without_release():
    b = make()
    hold(b, "L", 0, 2000)
    assert b.text == "L" and len(b.events) == 1
    hold(b, "O", 2050, 2750)
    assert b.text == "LO"


def test_fast_switching_and_low_confidence_do_not_commit():
    b = make()
    for t in range(0, 2000, 50):
        b.update(t, "L" if t % 100 == 0 else "M", 0.95, True)
    hold(b, "L", 2000, 4000, 0.79)
    assert b.text == ""


def test_short_gap_preserves_candidate_but_contributes_no_evidence():
    b = make()
    hold(b, "L", 0, 400)
    missing(b, 450, 500)
    assert b.candidate == "L" and b.evidence_ms == 400
    hold(b, "L", 550, 800)
    assert b.text == "" and b.evidence_ms == 650
    hold(b, "L", 850, 1500)
    assert b.text == "L"
    assert b.events[0]["acceptance_ms"] == 850
    assert b.events[0]["qualified_evidence_ms"] == 700
    missing(b, 1550, 1600)
    hold(b, "L", 1650, 2600)
    assert b.text == "L"


def test_long_release_allows_repeat_only_after_new_hold():
    b = make()
    hold(b, "L", 0, 700)
    missing(b, 750, 1050)
    hold(b, "L", 1100, 1750)
    assert b.text == "L"
    hold(b, "L", 1800, 1800)
    assert b.text == "LL"


def test_visible_low_confidence_never_unlocks_repeat():
    b = make()
    hold(b, "L", 0, 700)
    hold(b, "L", 750, 2000, 0.1)
    hold(b, "L", 2050, 3000)
    assert b.text == "L"


def test_low_confidence_breaks_continuous_absence():
    b = make()
    hold(b, "L", 0, 700)
    missing(b, 750, 850)
    b.update(900, "L", 0.1, True)
    missing(b, 950, 1100)
    hold(b, "L", 1150, 2200)
    assert b.text == "L"


def test_gap_above_grace_discards_candidate_but_does_not_release():
    b = make()
    hold(b, "L", 0, 400)
    missing(b, 450, 600)
    assert b.candidate is None
    hold(b, "L", 650, 1350)
    assert b.events[0]["first_qualified_ms"] == 650
    missing(b, 1400, 1550)
    hold(b, "L", 1600, 2400)
    assert b.text == "L"


def test_absence_duration_checked_on_return_not_frame_count():
    b = make()
    hold(b, "L", 0, 700)
    b.update(750)
    hold(b, "L", 1050, 1750)
    assert b.text == "LL"


def test_hello_and_acceptance_statistics(tmp_path):
    b = make()
    hold(b, "H", 0, 700)
    hold(b, "E", 750, 1450)
    hold(b, "L", 1500, 2200)
    missing(b, 2250, 2550)
    hold(b, "L", 2600, 3300)
    hold(b, "O", 3350, 4050)
    assert b.text == "HELLO"
    assert b.summary() == {
        "accepted_letters": 5,
        "mean_acceptance_ms": 700,
        "median_acceptance_ms": 700,
    }
    b.save_session(tmp_path / "log.json", source="synthetic_test")
    report = json.loads((tmp_path / "log.json").read_text())
    assert report["source"] == "synthetic_test" and len(report["events"]) == 5


def test_edit_keys_preserve_repeat_lock_and_session_counts():
    b = make()
    assert b.summary()["mean_acceptance_ms"] is None
    hold(b, "L", 0, 700)
    b.handle_key(ord(" "))
    assert b.text == "L "
    b.handle_key(8)
    assert b.text == "L"
    b.handle_key(127)
    assert b.text == ""
    hold(b, "L", 750, 1500)
    assert b.text == ""
    hold(b, "O", 1550, 2250)
    assert b.text == "O"
    b.handle_key(ord("C"))
    assert b.text == ""
    assert b.summary()["accepted_letters"] == 2


def test_configuration_and_time_validation():
    with pytest.raises(ValueError):
        WordBuilder(grace_ms=300, release_ms=300)
    b = make()
    b.update(100)
    with pytest.raises(ValueError):
        b.update(100)
    with pytest.raises(ValueError):
        b.update(float("nan"))
