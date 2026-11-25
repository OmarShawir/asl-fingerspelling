"""Stateful greedy and log-space CTC prefix beam search without a language model."""

import math

import numpy as np

from .sequence_data import VOCABULARY


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def log_probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected one finite logit vector")
    values = values - values.max()
    return values - np.log(np.exp(values).sum())


class GreedyDecoder:
    def __init__(self, vocabulary=VOCABULARY):
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        self.previous = None
        self.text = ""

    def step(self, log_probs):
        if len(log_probs) != len(self.vocabulary) + 1:
            raise ValueError("Vocabulary size mismatch")
        token = int(np.argmax(log_probs))
        if token and token != self.previous:
            self.text += self.vocabulary[token - 1]
        self.previous = token
        return self.text


class PrefixBeamDecoder:
    def __init__(self, width=10, vocabulary=VOCABULARY):
        if width < 1:
            raise ValueError("Positive beam width required")
        self.width = width
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        self.beams = {(): (0.0, -math.inf)}
        self.text = ""

    def step(self, log_probs):
        if len(log_probs) != len(self.vocabulary) + 1:
            raise ValueError("Vocabulary size mismatch")
        values = list(map(float, log_probs))
        if any(math.isnan(x) or x == math.inf for x in values):
            raise ValueError("Invalid log probabilities")
        following = {}

        def add(prefix, blank=-math.inf, nonblank=-math.inf):
            a, b = following.get(prefix, (-math.inf, -math.inf))
            following[prefix] = (logadd(a, blank), logadd(b, nonblank))

        for prefix, (blank, nonblank) in self.beams.items():
            total = logadd(blank, nonblank)
            add(prefix, blank=total + values[0])
            # Every symbol participates: no token threshold, top-k or dictionary.
            for token in range(1, len(values)):
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank=nonblank + values[token])
                    add(prefix + (token,), nonblank=blank + values[token])
                else:
                    add(prefix + (token,), nonblank=total + values[token])
        ranked = sorted(following.items(), key=lambda x: (-logadd(*x[1]), x[0]))
        self.beams = dict(ranked[: self.width])
        best = next(iter(self.beams))
        self.text = "".join(self.vocabulary[i - 1] for i in best)
        return self.text


def make_decoder(settings):
    if settings["name"] == "greedy":
        return GreedyDecoder()
    if settings["name"] == "prefix_beam" and settings["beam_width"] == 10:
        return PrefixBeamDecoder(10)
    raise ValueError("Unsupported selected decoder")
