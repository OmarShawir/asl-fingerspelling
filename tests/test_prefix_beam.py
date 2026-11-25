import itertools
import math
from collections import defaultdict

import numpy as np
import pytest

from fingerspelling.sequence_decode import PrefixBeamDecoder, log_probabilities


def brute(probs, vocabulary):
    scores = defaultdict(float)
    for path in itertools.product(range(len(vocabulary) + 1), repeat=len(probs)):
        previous, text = None, ""
        for token in path:
            if token and token != previous:
                text += vocabulary[token - 1]
            previous = token
        scores[text] += math.prod(probs[t, c] for t, c in enumerate(path))
    return scores


@pytest.mark.parametrize("seed", range(4))
def test_exhaustive_short_paths(seed):
    # At <=3 frames, two nonblank symbols yield <=9 prefixes, so width 10 is exact.
    rng = np.random.default_rng(seed)
    for vocabulary in ("ab", "a "):
        probs = rng.uniform(0.1, 1, (3, 3))
        probs /= probs.sum(1, keepdims=True)
        expected = brute(probs, vocabulary)
        decoder = PrefixBeamDecoder(10, vocabulary)
        for row in probs:
            decoder.step(np.log(row))
        actual = {
            "".join(vocabulary[i - 1] for i in prefix): np.exp(np.logaddexp(*state))
            for prefix, state in decoder.beams.items()
        }
        for text, score in expected.items():
            assert actual[text] == pytest.approx(score, abs=1e-12)
        assert decoder.text == max(expected, key=expected.get)


@pytest.mark.parametrize("text", ["jazz", "all", "a b", "  ", "", "abcdefghijklmnopqrstuvwxyz "])
def test_full_vocabulary_repeats_space_blank(text):
    vocab = "abcdefghijklmnopqrstuvwxyz "
    path = [0]
    for char in text:
        token = vocab.index(char) + 1
        path.extend([token, token, 0])
    decoder = PrefixBeamDecoder(10)
    for token in path:
        logits = np.full(28, -30.0)
        logits[token] = 30
        decoder.step(log_probabilities(logits))
    assert decoder.text == text
    decoder.reset()
    assert decoder.text == "" and decoder.beams == {(): (0.0, -math.inf)}
