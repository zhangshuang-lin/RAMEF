"""WDCNN width control for source-only RAMEF capacity audits.

The convolution sequence, strides, pooling and receptive-field construction
are retained from the project WDCNN. Only channel width changes; the embedding
remains 128-D. This matches parameter count, not every inductive bias or FLOP.
"""
import torch.nn as nn
from fair_baseline_models import ConvBlock1D


class WidthMatchedWDCNN(nn.Module):
    def __init__(self, base_channels: int, embedding_dim: int = 128):
        super().__init__()
        a = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv1d(1, a, 64, 16, 24, bias=False), nn.BatchNorm1d(a), nn.ReLU(), nn.MaxPool1d(2),
            ConvBlock1D(a, 2*a), ConvBlock1D(2*a, 4*a), ConvBlock1D(4*a, 8*a),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(8*a, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 3)
        self.base_channels = a

    def forward(self, x):
        z = self.proj(self.net(x[:, None]).flatten(1))
        return {"embedding": z, "class_logits": self.classifier(z)}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def nearest_width(target_parameters: int):
    # An algebraic count avoids constructing many networks or consulting data.
    # Includes the three-class neural training head, as in the U2-Hz count.
    count = lambda a: 126*a*a + 1118*a + 128 + 387
    return min(range(16, 129), key=lambda a: (abs(count(a)-target_parameters), a))
