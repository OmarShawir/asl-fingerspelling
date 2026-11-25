"""Index-only audit preserves text and splits people, not rows or files."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "sequence_audit", Path(__file__).parents[1] / "scripts/audit_sequence_data.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def rows():
    return [
        dict(
            sequence_id=str(i),
            participant_id=str(i % 3),
            path="train_landmarks/a.parquet",
            file_id="a",
            phrase="Jj Zz 10/a-",
        )
        for i in range(12)
    ]


def test_character_counts_preserve_case_symbols_and_spaces():
    data = rows()
    stats = audit.summarize(data)
    assert stats["ascii_letter_coverage"]["j"] == dict(
        lowercase=12, uppercase=12, case_insensitive=24, sequences=12
    )
    assert stats["ascii_letter_coverage"]["z"]["case_insensitive"] == 24
    assert stats["whitespace"][" "] == 24
    assert stats["symbols"]["/"] == 12
    assert data[0]["phrase"] == "Jj Zz 10/a-"


def test_three_people_share_file_but_never_share_split():
    plan, selected = audit.sample_plan(rows(), {"train_landmarks/a.parquet": 100}, 100)
    groups = [set(ids) for ids in plan["participants"].values()]
    assert all(len(g) == 1 for g in groups)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert plan["download_bytes"] == 100
    assert len(selected) == 12
    assert all(r["phrase"] == "Jj Zz 10/a-" for r in selected)
    assert "Q" in plan["per_split"]["test"]["missing_letters"]


def test_unknown_sizes_and_insufficient_budget_are_not_estimated():
    plan, selected = audit.sample_plan(rows(), {}, 100)
    assert plan["status"] == "blocked" and not selected
    plan, selected = audit.sample_plan(rows(), {"train_landmarks/a.parquet": 101}, 100)
    assert plan["status"] == "blocked" and not selected
