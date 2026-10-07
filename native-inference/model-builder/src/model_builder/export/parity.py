from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParityMetrics:
    max_abs: float
    relative_l2: float


def assert_tensor_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    max_abs_limit: float = 1.0e-4,
    relative_l2_limit: float = 1.0e-4,
) -> ParityMetrics:
    """Require exact tensor metadata and bounded floating-point differences."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: dtype mismatch {actual.dtype} != {expected.dtype}"
        )
    if not actual.is_floating_point() or not expected.is_floating_point():
        raise TypeError(f"{label}: numerical parity requires floating-point tensors")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: output contains NaN or Inf")

    difference = actual.detach().to(torch.float64) - expected.detach().to(torch.float64)
    max_abs = float(difference.abs().max().item()) if difference.numel() else 0.0
    denominator = torch.linalg.vector_norm(expected.detach().to(torch.float64))
    denominator = denominator.clamp_min(torch.finfo(torch.float64).eps)
    relative_l2 = float(torch.linalg.vector_norm(difference) / denominator)
    if max_abs > max_abs_limit or relative_l2 > relative_l2_limit:
        raise AssertionError(
            f"{label}: max_abs={max_abs:.8g}, relative_l2={relative_l2:.8g}; "
            f"limits are {max_abs_limit:.1e} and {relative_l2_limit:.1e}"
        )
    return ParityMetrics(max_abs=max_abs, relative_l2=relative_l2)
