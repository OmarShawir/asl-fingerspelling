import numpy as np

from fingerspelling.evaluation import rank_clip
from fingerspelling.features import LABELS


def test_mean_probabilities_not_majority_vote_and_all_classes():
    p = np.zeros((3, 24), dtype=np.float32)
    p[:2, LABELS.index("A")] = 0.51
    p[:2, LABELS.index("S")] = 0.49
    p[2, LABELS.index("S")] = 1.0
    ranked = rank_clip(p, LABELS)
    assert ranked[0]["letter"] == "S"  # Two A frame votes still lose to S mean probability.
    np.testing.assert_allclose(ranked[0]["score"], 1.98 / 3, rtol=1e-6)
    assert len(ranked) == 3


def test_no_valid_frames_have_no_prediction():
    assert rank_clip(np.empty((0, 24)), LABELS) == []
