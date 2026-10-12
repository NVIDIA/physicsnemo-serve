"""Load the qualified DoMINO tensor-only ATen boundary sidecar."""

from __future__ import annotations

from pathlib import Path

import torch


OPERATOR_ID = "physicsnemo-cfd.domino-exact-boundary"
OPERATOR_ABI = "b1c60ddada2438469a1d24b4e53ae196425b73648f6d8ae45ecf64043755d7e6"
REQUIRED_OPERATORS = ((OPERATOR_ID, OPERATOR_ABI),)
_FAKES_REGISTERED = False


def _register_fakes() -> None:
    global _FAKES_REGISTERED
    if _FAKES_REGISTERED:
        return

    @torch.library.register_fake("pnmir_domino::nearest_upsample3d_2x")
    def _nearest_upsample3d_2x_fake(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError("nearest 3D upsampling requires a rank-five tensor")
        return value.new_empty(
            (*value.shape[:2], *(dimension * 2 for dimension in value.shape[2:]))
        )

    @torch.library.register_fake("pnmir_domino::sdf_features")
    def _sdf_features_fake(
        sdf_grid: torch.Tensor, scaling_factors: torch.Tensor
    ) -> torch.Tensor:
        if sdf_grid.ndim != 4 or scaling_factors.ndim != 1:
            raise ValueError("invalid DoMINO SDF feature input ranks")
        channels = 5 + scaling_factors.shape[0]
        return sdf_grid.new_empty((sdf_grid.shape[0], channels, *sdf_grid.shape[1:]))

    for name in ("tensor_scalar_add", "tensor_scalar_div", "tensor_scalar_mul"):
        torch.library.register_fake(f"pnmir_domino::{name}")(
            lambda value, _scalar: torch.empty_like(value)
        )
    torch.library.register_fake("pnmir_domino::reciprocal")(
        lambda value: torch.empty_like(value)
    )

    @torch.library.register_fake("pnmir_domino::index_select_dim0")
    def _index_select_dim0_fake(
        value: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        return value.new_empty((index.numel(), *value.shape[1:]))

    @torch.library.register_fake("pnmir_domino::index_select_dim1")
    def _index_select_dim1_fake(
        value: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        return value.new_empty((value.shape[0], index.numel(), *value.shape[2:]))

    torch.library.register_fake("pnmir_domino::tensor_sub")(
        lambda left, _right: torch.empty_like(left)
    )

    @torch.library.register_fake("pnmir_domino::vector_norm_last_dim")
    def _vector_norm_last_dim_fake(value: torch.Tensor) -> torch.Tensor:
        return value.new_empty((*value.shape[:-1], 1))

    _FAKES_REGISTERED = True


def load_exact_ops(library: Path) -> Path:
    """Load the native Tensor-only DoMINO exact-boundary operators."""
    library = library.expanduser().resolve()
    if not library.is_file():
        raise FileNotFoundError(f"DoMINO exact-operator library is missing: {library}")
    torch.ops.load_library(str(library))
    expected = (
        "nearest_upsample3d_2x",
        "sdf_features",
        "tensor_scalar_add",
        "tensor_scalar_div",
        "tensor_scalar_mul",
        "reciprocal",
        "index_select_dim0",
        "index_select_dim1",
        "tensor_sub",
        "vector_norm_last_dim",
    )
    if any(not hasattr(torch.ops.pnmir_domino, name) for name in expected):
        raise RuntimeError("DoMINO exact-operator library has incomplete schemas")
    _register_fakes()
    return library
