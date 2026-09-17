"""GeoTransolver surface core: imported weights and synthetic feature inputs.

Geometry preprocessing stays outside this graph. The four inputs are the local
embedding, cached local features, cached geometry context, and global conditions.
"""


def create_model(config, assets):
    import torch
    from physicsnemo.experimental.models.geotransolver import GeoTransolver

    if (
        config.get("functional_dim") != 6
        or config.get("out_dim") != 4
        or config.get("geometry_dim") != 3
        or config.get("global_dim") != 2
        or config.get("include_local_features") is not True
        or config.get("use_te") is not False
        or config.get("structured_shape") is not None
        or config.get("guard_config") is not None
    ):
        raise ValueError(
            "This example requires the FP32 single-stream surface checkpoint."
        )

    class SurfaceCore(GeoTransolver):
        # Keep the upstream constructor and state keys; Builder loads the weights.
        def forward(
            self, local_embedding, local_features, static_context, global_embedding
        ):
            global_context = self.context_builder.global_tokenizer(global_embedding)
            context = torch.cat((static_context, global_context), dim=-1)
            streams = (
                torch.cat(
                    (self.preprocess[0](local_embedding), local_features), dim=-1
                ),
            )
            for block in self.blocks:
                streams = block(streams, context)
            return self.ln_mlp_out[0](streams[0])

    return SurfaceCore(**config)


def create_cases(config, assets):
    """Three repeatable model-space cases; these are not CFD geometry samples."""
    import torch

    generator = torch.Generator().manual_seed(42)
    heads = config["n_head"]
    radii = len(config["radii"])
    shapes = (
        (1, 32, config["functional_dim"]),
        (1, 32, config["n_hidden_local"] * radii),
        (1, heads, config["slice_num"], config["n_hidden"] // heads * (radii + 1)),
        (1, 1, config["global_dim"]),
    )
    return [
        tuple(
            torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
            for shape in shapes
        )
        for _ in range(3)
    ]


def export_options(context):
    from pnmir_export import ExportOptions
    from pnmir_export.compat import NormalizeClampBounds

    return ExportOptions(
        onnx_passes=(NormalizeClampBounds(),) if context.backend == "tensorrt" else ()
    )
