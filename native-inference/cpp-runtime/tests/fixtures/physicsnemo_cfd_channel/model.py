from __future__ import annotations

import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected

from .problem import analytic_solution, channel_coordinates


def create_model() -> torch.nn.Module:
    """Train the tiny PhysicsNeMo surrogate used for the export smoke test."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = FullyConnected(
                in_features=2,
                out_features=3,
                num_layers=3,
                layer_size=32,
            )

            axis_x = torch.linspace(0.0, 1.0, 21)
            axis_y = torch.linspace(-1.0, 1.0, 21)
            xx, yy = torch.meshgrid(axis_x, axis_y, indexing="ij")
            coordinates = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
            targets = analytic_solution(coordinates)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(1200):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(coordinates), targets)
                loss.backward()
                optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)

    model.eval()
    with torch.inference_mode():
        error = model(channel_coordinates()) - analytic_solution(channel_coordinates())
        if error.abs().max().item() > 0.01:
            raise RuntimeError("PhysicsNeMo channel-flow surrogate did not converge")
    return model


def example_inputs() -> tuple[torch.Tensor, ...]:
    return (channel_coordinates(),)
