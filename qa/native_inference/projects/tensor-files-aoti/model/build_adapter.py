"""Build-time model adapter; never imported by native request execution."""

import torch


class Affine(torch.nn.Module):
    def forward(self, value):
        return value * 2.0 + 1.0


def create_model(config, assets):
    return Affine().eval()


def create_cases(config, assets):
    return [
        (torch.tensor(values, dtype=torch.float32),)
        for values in ([0, 1, -1, 4], [2.5, -8, 0.125, 13], [-0.5, 100, -32, 0.25])
    ]
