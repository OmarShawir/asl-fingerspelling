"""Shared static features: wrist-centered, scale-normalized, canonical right hand.

Reflect x for a physically left hand. Never rotate: finger orientation is retained.
The input uses MediaPipe normalized image coordinates for both images and webcam.
"""

import numpy as np

FEATURE_VERSION = "wrist-maxradius-canonical-right-v1"
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def prepare_features(points, hand):
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.shape[-2:] != (21, 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite (..., 21, 3) landmarks")
    if hand not in ("left", "right"):
        raise ValueError("Physical hand must be left or right")
    centered = xyz - xyz[..., :1, :]
    if hand == "left":
        centered[..., 0] *= -1
    scale = np.linalg.norm(centered, axis=-1).max(axis=-1, keepdims=True)
    if np.any(scale < 1e-6):
        raise ValueError("Degenerate hand landmarks")
    return (centered / scale[..., None]).reshape(*xyz.shape[:-2], 63)
