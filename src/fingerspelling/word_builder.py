"""Timestamp-driven letter acceptance, independent of camera and classifier."""

import json
import math
from pathlib import Path
from statistics import mean, median

from .features import LABELS


class WordBuilder:
    def __init__(self, hold_ms=700, grace_ms=100, release_ms=300, min_letter_confidence=0.8):
        if not 0 <= grace_ms < release_ms or hold_ms <= 0:
            raise ValueError("Require hold_ms > 0 and 0 <= grace_ms < release_ms")
        if not 0 <= min_letter_confidence <= 1:
            raise ValueError("Confidence threshold must be in [0, 1]")
        self.settings = dict(
            hold_ms=hold_ms,
            grace_ms=grace_ms,
            release_ms=release_ms,
            min_letter_confidence=min_letter_confidence,
        )
        self.text = ""
        self.events = []
        self.candidate = None
        self.candidate_since_ms = None
        self.evidence_ms = 0
        self._previous_qualified_ms = None
        self._last_timestamp_ms = None
        self._absence_since_ms = None
        self._last_accepted = None

    @classmethod
    def from_config(cls, settings):
        return cls(
            **{
                key: settings[key]
                for key in ("hold_ms", "grace_ms", "release_ms", "min_letter_confidence")
            }
        )

    def _reset_candidate(self):
        self.candidate = None
        self.candidate_since_ms = None
        self.evidence_ms = 0
        self._previous_qualified_ms = None

    def _apply_absence(self, timestamp_ms):
        duration = timestamp_ms - self._absence_since_ms
        if duration > self.settings["grace_ms"]:
            self._reset_candidate()
        if duration >= self.settings["release_ms"]:
            self._last_accepted = None

    def update(self, timestamp_ms, letter=None, confidence=0.0, hand_visible=False):
        """Return an acceptance event or None.

        Evidence accumulates only between successive qualified visible samples of
        the same candidate. Missing samples add no evidence, including both gap
        boundaries. A short gap preserves evidence and the first qualifying time.
        Acceptance time is wall-clock elapsed from that first qualifying sample,
        so it can include a grace gap even though stability evidence excludes it.
        """
        if not math.isfinite(timestamp_ms) or (
            self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError("Timestamps must be finite and strictly increasing in milliseconds")
        self._last_timestamp_ms = timestamp_ms
        if not hand_visible:
            if self._absence_since_ms is None:
                self._absence_since_ms = timestamp_ms
            self._previous_qualified_ms = None
            self._apply_absence(timestamp_ms)
            return None
        if self._absence_since_ms is not None:
            self._apply_absence(timestamp_ms)
            self._absence_since_ms = None
        if (
            letter not in LABELS
            or not math.isfinite(confidence)
            or confidence < self.settings["min_letter_confidence"]
        ):
            self._reset_candidate()
            return None
        if letter == self._last_accepted:
            self._reset_candidate()
            return None
        if letter != self.candidate:
            self._reset_candidate()
            self.candidate = letter
            self.candidate_since_ms = timestamp_ms
        elif self._previous_qualified_ms is not None:
            self.evidence_ms += timestamp_ms - self._previous_qualified_ms
        self._previous_qualified_ms = timestamp_ms
        if self.evidence_ms < self.settings["hold_ms"]:
            return None
        event = {
            "letter": letter,
            "first_qualified_ms": self.candidate_since_ms,
            "accepted_ms": timestamp_ms,
            "acceptance_ms": timestamp_ms - self.candidate_since_ms,
            "qualified_evidence_ms": self.evidence_ms,
        }
        self.text += letter
        self.events.append(event)
        self._last_accepted = letter
        self._reset_candidate()
        return event

    def handle_key(self, key):
        """Recognition-only edits; never release the physical repeat lock.

        Edits cancel a pending candidate. They do not erase acceptance statistics.
        The recording loop routes Space to recording before calling this method.
        """
        if key == ord(" "):
            self.text += " "
        elif key in (8, 127):
            self.text = self.text[:-1]
        elif key in (ord("c"), ord("C")):
            self.text = ""
        else:
            return False
        self._reset_candidate()
        return True

    def summary(self):
        durations = [e["acceptance_ms"] for e in self.events]
        return {
            "accepted_letters": len(durations),
            "mean_acceptance_ms": mean(durations) if durations else None,
            "median_acceptance_ms": median(durations) if durations else None,
        }

    def save_session(self, path, source="live_camera"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "source": source,
            "settings": self.settings,
            "text": self.text,
            **self.summary(),
            "events": self.events,
            "metric": "Letter acceptance time from first qualifying prediction to acceptance; "
            "not latency from physical motion onset. Edits do not reduce event count.",
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
        return report
