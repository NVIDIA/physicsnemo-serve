from __future__ import annotations

import torch


def channel_coordinates() -> torch.Tensor:
    """Return the fixed 3x3 channel grid used by the static-shape MVP."""
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def analytic_solution(coordinates: torch.Tensor) -> torch.Tensor:
    """Steady plane-Poiseuille solution for viscosity 0.5 and unit density."""
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    u = 1.0 - y.square()
    v = torch.zeros_like(x)
    pressure = 1.0 - x
    return torch.stack((u, v, pressure), dim=-1)
