"""Use identical MLP weights and saved cases for every compiler selection."""

import torch


def create_model(config, assets):
    return torch.nn.Sequential(
        torch.nn.Linear(config["input_dim"], config["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(config["hidden_dim"], config["output_dim"]),
    ).eval()


def create_cases(config, assets):
    return torch.load(
        assets["validation_inputs"], map_location="cpu", weights_only=True
    )
