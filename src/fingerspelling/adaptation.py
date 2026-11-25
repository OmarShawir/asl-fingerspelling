"""Sampling and split checks for session-held-out adaptation."""

from collections import Counter
import torch


def camera_frame_weights(targets, clip_ids):
    """Equal class mass, then equal clip mass within class, then uniform frames."""
    if not targets or len(targets) != len(clip_ids):
        raise ValueError("Nonempty aligned targets and clip IDs required")
    clip_class = {}
    for label, clip_id in zip(targets, clip_ids):
        if clip_id in clip_class and clip_class[clip_id] != label:
            raise ValueError("One clip cannot have multiple labels")
        clip_class[clip_id] = label
    class_clips = Counter(clip_class.values())
    frames = Counter(clip_ids)
    return torch.tensor(
        [1 / (len(class_clips) * class_clips[y] * frames[c]) for y, c in zip(targets, clip_ids)],
        dtype=torch.float64,
    )


def assert_disjoint(train_ids, validation_ids):
    overlap = set(train_ids) & set(validation_ids)
    if overlap:
        raise ValueError(f"Training/validation overlap: {sorted(overlap)}")


def balanced_batch(kaggle_weights, camera_weights, half_batch, generator):
    """Exactly half of each batch from each source, sampled with replacement."""
    return (
        torch.multinomial(kaggle_weights, half_batch, replacement=True, generator=generator),
        torch.multinomial(camera_weights, half_batch, replacement=True, generator=generator),
    )
