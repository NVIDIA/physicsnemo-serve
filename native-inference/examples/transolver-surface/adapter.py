"""Transolver surface model with imported constructor settings and weights.

The C++ workflow prepares coordinates, normals and global conditions outside
this graph, then converts its four standardized outputs into physical fields.
"""

# Build a separate project for every block/tail size needed by the C++ workflow.
POINT_COUNT = 75


def create_model(config, assets):
    from physicsnemo.models.transolver import Transolver

    if (
        config.get("functional_dim") != 2
        or config.get("embedding_dim") != 6
        or config.get("out_dim") != 4
        or config.get("unified_pos") is not False
        or config.get("structured_shape") is not None
        or config.get("use_te") is not False
        or config.get("time_input") is not False
        or config.get("plus") is not False
    ):
        raise ValueError(
            "This example requires the FP32 Transolver surface checkpoint."
        )

    # Preserve the upstream graph and state keys; Builder loads the weights.
    return Transolver(**config)


def create_cases(config, assets):
    """Three repeatable model-space cases; these are not CFD geometry samples."""
    import torch

    generator = torch.Generator().manual_seed(42)
    cases = []
    for index in range(3):
        positions = (
            torch.randn(1, POINT_COUNT, 3, generator=generator, dtype=torch.float32)
            * 0.1
        )
        normals = torch.nn.functional.normalize(
            torch.randn(1, POINT_COUNT, 3, generator=generator, dtype=torch.float32),
            dim=-1,
        )
        embedding = torch.cat((positions, normals), dim=-1).contiguous()
        # Global density (kg/m^3) and speed (m/s), repeated at every surface cell.
        fx = torch.tensor(
            [1.205 + index * 0.01, 30.0 + index * 5.0], dtype=torch.float32
        )
        cases.append((fx.repeat(1, POINT_COUNT, 1), embedding))
    return cases


def export_options(context):
    """Normalize temperature clamp bounds only in the captured ONNX graph."""
    from model_builder.export import ExportOptions
    from model_builder.export.compat import NormalizeClampBounds

    return ExportOptions(
        onnx_passes=(NormalizeClampBounds(),) if context.backend == "tensorrt" else ()
    )
