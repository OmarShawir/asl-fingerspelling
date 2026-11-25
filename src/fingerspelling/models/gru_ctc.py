"""Own-trained unordered-hand encoder and causal full-resolution GRU/CTC."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequenceGRUCTC(nn.Module):
    def __init__(
        self,
        hand_hidden=96,
        hand_embedding=64,
        hidden_size=128,
        layers=2,
        dropout=0.1,
        classes=28,
    ):
        super().__init__()
        self.config = dict(
            hand_hidden=hand_hidden,
            hand_embedding=hand_embedding,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
            classes=classes,
        )
        self.hand_encoder = nn.Sequential(
            nn.Linear(67, hand_hidden), nn.ReLU(), nn.Linear(hand_hidden, hand_embedding), nn.ReLU()
        )
        self.gru = nn.GRU(
            hand_embedding * 2 + 1,
            hidden_size,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, classes)

    def frame_embeddings(self, x, mask):
        # Mask before encoding too: padded/missing values cannot affect gradients/features.
        encoded = self.hand_encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
        count = mask.sum(-1, keepdim=True)
        mean = (encoded * mask.unsqueeze(-1)).sum(-2) / count.clamp_min(1)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(-2)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum, count.to(x.dtype) / 2), dim=-1)

    def forward(self, x, mask, lengths):
        frames = self.frame_embeddings(x, mask)
        packed = pack_padded_sequence(frames, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=x.shape[1])
        return self.head(outputs)
