"""Reproducible software-only timing scenarios; no camera or classifier accuracy claim."""

import json
from pathlib import Path
from statistics import mean, median

from fingerspelling.config import read_config
from fingerspelling.word_builder import WordBuilder


def run_scenarios(settings):
    results = {}

    def builder():
        return WordBuilder.from_config(settings)

    def feed(b, start, end, letter=None, confidence=0.95, visible=True):
        for timestamp in range(start, end + 1, 50):
            b.update(timestamp, letter, confidence, visible)

    def save(name, b, expected):
        assert b.text == expected, (name, b.text, expected)
        results[name] = {"text": b.text, "expected": expected, **b.summary(), "events": b.events}

    b = builder()
    feed(b, 0, 2100, "L")
    save("long_L_once", b, "L")

    b = builder()
    for t in range(0, 1050, 50):
        b.update(t, "L" if t % 100 == 0 else "M", 0.95, True)
    feed(b, 1050, 2100, "L", 0.79)
    save("switching_and_low_confidence", b, "")

    b = builder()
    feed(b, 0, 400, "L")
    feed(b, 450, 500, visible=False)
    feed(b, 550, 1500, "L")
    feed(b, 1550, 1600, visible=False)
    feed(b, 1650, 2500, "L")
    assert b.events[0]["accepted_ms"] == 850
    assert b.events[0]["qualified_evidence_ms"] == 700
    save("two_missing_frames_no_repeat", b, "L")

    b = builder()
    feed(b, 0, 900, "L")
    feed(b, 950, 1250, visible=False)
    feed(b, 1300, 2100, "L")
    save("long_release_allows_LL", b, "LL")

    b = builder()
    feed(b, 0, 700, "H")
    feed(b, 750, 1450, "E")
    feed(b, 1500, 2200, "L")
    feed(b, 2250, 2550, visible=False)
    feed(b, 2600, 3300, "L")
    feed(b, 3350, 4050, "O")
    save("HELLO", b, "HELLO")
    return results


def main():
    settings = read_config()["static"]
    scenarios = run_scenarios(settings)
    durations = [e["acceptance_ms"] for s in scenarios.values() for e in s["events"]]
    report = {
        "source": "synthetic_timestamps_not_camera",
        "settings": settings,
        "scenarios": scenarios,
        "accepted_letters": len(durations),
        "mean_acceptance_ms": mean(durations),
        "median_acceptance_ms": median(durations),
        "metric": "Letter acceptance time: first qualifying prediction to accepted letter; "
        "not user-motion onset latency.",
    }
    path = Path("results/word_builder_simulation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
