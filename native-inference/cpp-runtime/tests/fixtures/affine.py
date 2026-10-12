import torch


class Affine(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2.0 + 1.0


def create_model() -> torch.nn.Module:
    return Affine()


def example_inputs() -> tuple[torch.Tensor, ...]:
    return (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),)
