"""A randomly initialized classifier over 21 normalized xyz landmarks."""

import torch
from torch import nn

from fingerspelling.features import FEATURE_VERSIONS, prepare_features


class StaticMLP(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class StaticPredictor:
    def __init__(self, checkpoint):
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.feature_version = bundle["feature_version"]
        if self.feature_version not in FEATURE_VERSIONS:
            raise ValueError("Incompatible checkpoint preprocessing")
        self.labels = bundle["labels"]
        self.model = StaticMLP(len(self.labels))
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

    def probabilities(self, points, hand, *, frame_width=None, frame_height=None):
        x = prepare_features(
            points,
            hand,
            feature_version=self.feature_version,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        with torch.inference_mode():
            return self.model(torch.from_numpy(x).reshape(-1, 63)).softmax(-1).numpy()
