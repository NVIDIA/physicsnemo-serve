# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load only the pinned, digest-checked checkpoint and interpolation artifacts."""

from importlib.metadata import distribution, version
import json
import os
from pathlib import Path

import numpy as np

from aifs2ens_artifacts import EARTH2STUDIO_REVISION, InputContract, sha256_file

CHECKPOINT_SHA256 = "e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb"
MATRICES = {
    "interpolation_sha256": (
        "aifs-ens-2.0_interpolation_matrix",
        "9533e90f8433424400ab53c7fafc87ba1a04453093311c0b5bd0b35fedc1fb83.npz",
    ),
    "inverse_interpolation_sha256": (
        "aifs-ens-2.0_inverse_interpolation_matrix",
        "7f0be51c7c1f522592c7639e0d3f95bcbff8a044292aa281c1e73b842736d9bf.npz",
    ),
}


def verified_checkpoint(contract):
    if contract.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("contract does not identify the pinned checkpoint")
    root = os.environ.get("AIFS2ENS_CHECKPOINT_DIR")
    if not root:
        raise RuntimeError(
            "AIFS2ENS_CHECKPOINT_DIR must name the pinned asset directory"
        )
    checkpoint = Path(root) / "aifs-ens-crps-2.0.ckpt"
    if sha256_file(checkpoint) != contract.checkpoint_sha256:
        raise ValueError("checkpoint digest differs from its pinned identity")
    return checkpoint


def _verify_runtime():
    metadata = distribution("earth2studio").read_text("direct_url.json")
    provenance = json.loads(metadata or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != EARTH2STUDIO_REVISION:
        raise RuntimeError(
            "install the pinned Earth2Studio source revision for AIFS2ENS"
        )
    if version("anemoi-models") != "0.11.2":
        raise RuntimeError("AIFS2ENS requires anemoi-models==0.11.2")


def prepare_assets(contract=None):
    if contract is None:
        path = os.environ.get("AIFS2ENS_INPUT_CONTRACT")
        if not path:
            raise RuntimeError("AIFS2ENS_INPUT_CONTRACT is required")
        contract = InputContract.model_validate_json(Path(path).read_text())
    checkpoint = verified_checkpoint(contract)
    _verify_runtime()
    from earth2studio.models.auto import Package
    from plugin_sdk import cleanup_earth2_runtime_resources

    for field, (cache_name, filename) in MATRICES.items():
        expected = getattr(contract, field)
        if expected is None:
            raise ValueError(f"the qualified contract must bind {field}")
        package = Package(
            "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/",
            cache_options={
                "cache_storage": Package.default_cache(cache_name),
                "same_names": True,
            },
        )
        try:
            path = Path(package.resolve(filename))
            if sha256_file(path) != expected:
                raise ValueError(f"interpolation artifact digest differs: {field}")
        finally:
            cleanup_earth2_runtime_resources(package)
    return checkpoint


def verify_model_contract(model, contract):
    inputs = model.input_coords()
    outputs = model.output_coords(inputs)
    for coords, variables, lead in (
        (inputs, contract.input_variables, [-6, 0]),
        (outputs, contract.output_variables, [6]),
    ):
        expected = {
            "variable": variables,
            "lat": contract.lat,
            "lon": contract.lon,
            "lead_time": np.array(lead, dtype="timedelta64[h]"),
        }
        for name, values in expected.items():
            if name not in coords or not np.array_equal(coords[name], values):
                raise ValueError(
                    f"loaded model {name} differs from the qualified contract"
                )


def load_model(contract, device):
    checkpoint = prepare_assets(contract)
    import torch
    from earth2studio.models.auto import Package
    from earth2studio.models.px import AIFS2ENS

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("AIFS2ENS model execution requires a CUDA device")
    package = Package(str(checkpoint.parent), cache=False)
    model = AIFS2ENS.load_model(package, preload_invariants=False).to(device).eval()
    verify_model_contract(model, contract)
    return package, model


def runtime_identity(model, device):
    """Record the executing process, including settings a service launcher may set."""
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": version("flash-attn"),
        "anemoi_models": version("anemoi-models"),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu_name": torch.cuda.get_device_name(device),
        "model_seed": model.seed,
    }
