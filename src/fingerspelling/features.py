"""Shared static features: wrist-centered, scale-normalized, canonical right hand.

Reflect x for a physically left hand. Never rotate: finger orientation is retained.
The input uses MediaPipe normalized image coordinates for both images and webcam.
"""

import numpy as np

FEATURE_V1 = "wrist-maxradius-canonical-right-v1"
FEATURE_V2 = "width-units-wrist-maxradius-canonical-right-v2"
FEATURE_VERSION = FEATURE_V1  # Preserve legacy callers and checkpoints.
FEATURE_VERSIONS = (FEATURE_V1, FEATURE_V2)
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def validate_dimensions(frame_width, frame_height):
    for name, value in (("frame_width", frame_width), ("frame_height", frame_height)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(
                f"v2 requires a documented positive integer {name}; record new clips with frame dimensions"
            )
    return frame_width, frame_height


def prepare_features(
    points, hand, *, feature_version=FEATURE_V1, frame_width=None, frame_height=None
):
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.shape[-2:] != (21, 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite (..., 21, 3) landmarks")
    if hand not in ("left", "right"):
        raise ValueError("Physical hand must be left or right")
    if feature_version not in FEATURE_VERSIONS:
        raise ValueError(f"Unsupported feature version: {feature_version}")
    if feature_version == FEATURE_V2:
        width, height = validate_dimensions(frame_width, frame_height)
        xyz = xyz.copy()
        xyz[..., 1] *= height / width
    centered = xyz - xyz[..., :1, :]
    if hand == "left":
        centered[..., 0] *= -1
    scale = np.linalg.norm(centered, axis=-1).max(axis=-1, keepdims=True)
    if np.any(scale < 1e-6):
        raise ValueError("Degenerate hand landmarks")
    return (centered / scale[..., None]).reshape(*xyz.shape[:-2], 63)
