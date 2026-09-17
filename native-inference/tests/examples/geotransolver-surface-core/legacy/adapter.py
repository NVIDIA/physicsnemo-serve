"""Prepared GeoTransolver surface-core adapter; framework imports are deferred."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path

PHYSICSNEMO_VERSION = "2.1.1"
MODEL_IMPORT = "physicsnemo.experimental.models.geotransolver.GeoTransolver"


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def state_sha256(state):
    """Identify every named tensor, including upstream context-builder weights."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint must be a plain tensor state dictionary")
        value = tensor.detach().cpu().contiguous()
        descriptor = _canonical([name, str(value.dtype), list(value.shape)])
        data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (descriptor, data):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _fixtures(config, assets):
    import torch

    if not isinstance(config, dict) or config.get("format_version") != 1:
        raise ValueError("unsupported GeoTransolver configuration")
    if (
        config.get("physicsnemo_version") != PHYSICSNEMO_VERSION
        or config.get("model_import") != MODEL_IMPORT
    ):
        raise ValueError(
            "configuration must select the pinned GeoTransolver implementation"
        )
    if metadata.version("nvidia-physicsnemo") != PHYSICSNEMO_VERSION:
        raise ValueError(
            f"GeoTransolver requires nvidia-physicsnemo=={PHYSICSNEMO_VERSION}"
        )
    if "fixtures" not in assets:
        raise ValueError("GeoTransolver requires the prepared fixtures asset")
    fixtures = torch.load(
        Path(assets["fixtures"]), weights_only=True, map_location="cpu"
    )
    if not isinstance(fixtures, dict) or fixtures.get("format_version") != 1:
        raise ValueError("unsupported GeoTransolver fixture format")
    if _canonical(fixtures.get("config")) != _canonical(config):
        raise ValueError(
            "configuration differs from the prepared GeoTransolver fixtures"
        )
    cases = fixtures.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("GeoTransolver fixtures require at least three verified cases")
    for case in cases:
        values = case.get("inputs")
        expected = case.get("expected")
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("GeoTransolver fixtures require four inputs per case")
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            for value in (*values, expected)
        ):
            raise ValueError(
                "GeoTransolver fixtures must contain finite float32 tensors"
            )
    return fixtures


def create_model(config, assets):
    import torch
    from physicsnemo.experimental.models.geotransolver import GeoTransolver

    fixtures = _fixtures(config, assets)
    args = config["model_args"]
    if (
        args.get("functional_dim") != 6
        or args.get("out_dim") != 4
        or args.get("global_dim") != 2
        or args.get("use_te") is not False
        or args.get("include_local_features") is not True
    ):
        raise ValueError(
            "configuration must describe the FP32 surface GeoTransolver checkpoint"
        )

    class SurfaceCachedCore(GeoTransolver):
        # Inherit the upstream constructor and every state key. Only the executed
        # forward is partitioned; cached geometry features arrive as tensor inputs.
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

    model = SurfaceCachedCore(**args)

    def verify_loaded_state(loaded, incompatible):
        if state_sha256(loaded.state_dict()) != fixtures.get("state_sha256"):
            raise ValueError(
                "loaded checkpoint state differs from the prepared GeoTransolver fixtures; prepare new fixtures for these weights"
            )
        handle.remove()

    handle = model.register_load_state_dict_post_hook(verify_loaded_state)
    return model


def create_cases(config, assets):
    return [
        tuple(value.clone() for value in case["inputs"])
        for case in _fixtures(config, assets)["cases"]
    ]


def export_options(context):
    """Keep ONNX compatibility separate from model construction and AOTI."""
    from pnmir_export import ExportOptions
    from pnmir_export.compat import NormalizeClampBounds

    return ExportOptions(
        onnx_passes=(NormalizeClampBounds(),) if context.backend == "tensorrt" else ()
    )
