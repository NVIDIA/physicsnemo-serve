"""A tiny deterministic recipe exercising compile-once, replay-many inputs."""

import torch


class Affine(torch.nn.Module):
    def forward(self, value):
        return value * 2.0 + 1.0


def create_model():
    return Affine().eval()


def create_cases():
    return [
        (torch.tensor(values, dtype=torch.float32),)
        for values in (
            [0.0, 1.0, -1.0, 4.0],
            [2.5, -8.0, 0.125, 13.0],
            [-0.5, 100.0, -32.0, 1.0e-4],
        )
    ]
