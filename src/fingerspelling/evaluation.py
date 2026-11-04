"""Clip-level scoring; no voting and no restriction to observed target letters."""

import numpy as np


def rank_clip(probabilities, labels):
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Expected frames by all checkpoint labels")
    if not len(probabilities):
        return []
    mean = probabilities.mean(axis=0)
    order = np.argsort(-mean, kind="stable")[:3]
    return [{"letter": labels[int(i)], "score": float(mean[i])} for i in order]
