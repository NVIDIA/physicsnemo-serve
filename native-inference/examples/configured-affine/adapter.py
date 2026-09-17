"""Small learned affine model for the model-input contract example."""

import json
from pathlib import Path

import torch


class ConfiguredAffine(torch.nn.Module):
    def __init__(self, offset, mean, std):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.offset = offset
        self.mean = mean
        self.std = std

    def forward(self, value):
        normalized = (value - self.mean) / self.std
        return normalized * self.scale + self.bias + self.offset


def create_model(config, assets):
    normalization = json.loads(Path(assets["normalization"]).read_text())
    return ConfiguredAffine(
        offset=float(config["offset"]),
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
    ).eval()


def create_cases(config, assets):
    return [
        (torch.tensor(values, dtype=torch.float32),)
        for values in (
            [0.0, 1.0, -1.0, 4.0],
            [2.5, -8.0, 0.125, 13.0],
            [-0.5, 100.0, -32.0, 0.25],
        )
    ]
