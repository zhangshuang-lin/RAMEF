"""Compact, conventional PU baselines with a common 128-D interface."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1):
        super().__init__(nn.Conv1d(cin, cout, kernel, stride, kernel // 2, bias=False),
                         nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2))


class WDCNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 64, 16, 24, bias=False), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            ConvBlock1D(16, 32), ConvBlock1D(32, 64), ConvBlock1D(64, 128),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x): return self.proj(self.net(x[:, None]).flatten(1))


class Residual1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__(); self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 5, 1, 2, bias=False); self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))
    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet1DEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__(); self.stem = nn.Sequential(nn.Conv1d(1, 32, 15, 4, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.blocks = nn.Sequential(Residual1D(32, 32), Residual1D(32, 64, 2),
                                    Residual1D(64, 64), Residual1D(64, 128, 2),
                                    nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(128, embedding_dim)
    def forward(self, x): return self.proj(self.blocks(self.stem(x[:, None])).flatten(1))


class STFTCNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__(); self.register_buffer("window", torch.hann_window(512), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.proj = nn.Linear(256, embedding_dim)
    def forward(self, x):
        z = torch.stft(x, 512, 128, 512, self.window, return_complex=True).abs().clamp_min(1e-6).log1p()
        return self.proj(self.net(z[:, None]).flatten(1))


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength): ctx.strength = strength; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.strength * grad, None


class BaselineNetwork(nn.Module):
    def __init__(self, encoder, dann=False):
        super().__init__(); self.encoder = encoder; self.classifier = nn.Linear(128, 3)
        self.domain = nn.Linear(128, 3) if dann else None
    def forward(self, x, grl=0.):
        z = self.encoder(x); result = {"embedding": z, "class_logits": self.classifier(z)}
        if self.domain is not None: result["domain_logits"] = self.domain(GradientReverse.apply(z, grl))
        return result


def make_model(name):
    if name in ("wdcnn", "wdcnn_dann"): encoder = WDCNNEncoder()
    elif name == "resnet1d": encoder = ResNet1DEncoder()
    elif name == "stft_cnn": encoder = STFTCNNEncoder()
    else: raise ValueError(name)
    return BaselineNetwork(encoder, dann=name == "wdcnn_dann")
