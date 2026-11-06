# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent stock Earth2Studio ensemble reference; imports no plugin code."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

POLICY = "earth2studio-native-v1"
RNG_POLICY = "torch-seed-once-native-v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


class FrozenSource:
    """Frozen physical input times exposed through Earth2Studio's datasource API."""

    def __init__(self, values, contract, start_time):
        import xarray as xr

        start = np.datetime64(start_time.removesuffix("Z"), "ns")
        self.array = xr.DataArray(
            values[0],
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": np.array([start - np.timedelta64(6, "h"), start]),
                "variable": contract["input_variables"],
                "lat": contract["lat"],
                "lon": contract["lon"],
            },
        )

    def __call__(self, time, variable):
        return self.array.sel(
            time=np.asarray(time, dtype="datetime64[ns]"), variable=list(variable)
        )


class CaptureIO:
    """Record all member slots from stock run.ensemble, excluding initial yield."""

    def __init__(
        self,
        root,
        *,
        nensemble,
        nsteps,
        variables,
        contract,
        start_time,
        validate_device,
    ):
        self.root = root
        self.nensemble, self.nsteps = nensemble, nsteps
        self.variables = list(variables)
        self.contract = contract
        self.start = np.datetime64(start_time.removesuffix("Z"), "ns")
        self.validate_device = validate_device
        self.arrays = []
        self.coverage = np.zeros((nsteps, nensemble, len(self.variables)), dtype=bool)

    def add_array(self, coords, array_name):
        if self.arrays:
            raise ValueError("reference arrays were declared twice")
        expected = {
            "ensemble": np.arange(self.nensemble),
            "time": np.array([self.start]),
            "lead_time": np.arange(self.nsteps + 1) * np.timedelta64(6, "h"),
            "lat": self.contract["lat"],
            "lon": self.contract["lon"],
        }
        if list(array_name) != self.variables:
            raise ValueError("reference output variables differ")
        for name, values in expected.items():
            if name not in coords or not np.array_equal(coords[name], values):
                raise ValueError(f"reference {name} coordinates differ")
        shape = (
            self.nensemble,
            1,
            1,
            len(self.variables),
            len(expected["lat"]),
            len(expected["lon"]),
        )
        self.arrays = [
            np.lib.format.open_memmap(
                self.root / f"step-{step:04d}.npy",
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
            for step in range(1, self.nsteps + 1)
        ]

    def write(self, data, coords, array_name):
        lead = np.asarray(coords["lead_time"])
        if np.array_equal(lead, np.array([0], dtype="timedelta64[h]")):
            return
        self.validate_device()
        if lead.shape != (1,) or lead[0] % np.timedelta64(6, "h"):
            raise ValueError("reference emitted unexpected forecast lead time")
        step = int(lead[0] / np.timedelta64(6, "h"))
        if not 1 <= step <= self.nsteps:
            raise ValueError("reference emitted forecast outside requested horizon")
        members = np.asarray(coords["ensemble"])
        if (
            members.dtype.kind not in "iu"
            or members.ndim != 1
            or len(set(members)) != len(members)
            or np.any(members < 0)
            or np.any(members >= self.nensemble)
        ):
            raise ValueError("reference member identities differ")
        for name, values in (
            ("time", [self.start]),
            ("lat", self.contract["lat"]),
            ("lon", self.contract["lon"]),
        ):
            if not np.array_equal(coords[name], values):
                raise ValueError(f"reference {name} coordinates differ")
        if len(data) != len(array_name):
            raise ValueError("reference output variable coverage differs")
        for name, tensor in zip(array_name, data):
            if name not in self.variables:
                raise ValueError("unknown reference output variable")
            index = self.variables.index(name)
            values = (
                tensor.detach().cpu().numpy()
                if hasattr(tensor, "detach")
                else np.asarray(tensor)
            )
            expected = (
                len(members),
                1,
                1,
                len(self.contract["lat"]),
                len(self.contract["lon"]),
            )
            if values.shape != expected or values.dtype != np.float32:
                raise ValueError("reference output shape/dtype differs")
            if self.coverage[step - 1, members, index].any():
                raise ValueError("duplicate reference member write")
            self.arrays[step - 1][members, :, :, index] = values
            self.coverage[step - 1, members, index] = True

    def finish(self):
        if not self.coverage.all() or len(self.arrays) != self.nsteps:
            raise ValueError("reference ensemble forecast coverage is incomplete")
        steps = []
        for step, array in enumerate(self.arrays, 1):
            array.flush()
            path = self.root / f"step-{step:04d}.npy"
            steps.append(
                {
                    "step": step,
                    "path": path.name,
                    "sha256": digest(path),
                    "shape": list(array.shape),
                }
            )
        return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--nsteps", required=True, type=int, choices=range(1, 61))
    parser.add_argument("--nensemble", required=True, type=int, choices=range(1, 17))
    parser.add_argument("--batch-size", required=True, type=int, choices=range(1, 17))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**63 - 1:
        raise ValueError("reference seed is outside supported integer range")
    args.output.mkdir(parents=True)
    contract = json.loads(args.contract.read_text())
    checkpoint = args.checkpoint_dir / "aifs-ens-crps-2.0.ckpt"
    if (
        digest(checkpoint)
        != "e383027ee9a20469b920ed11f9af2ecc9268ebcf826447638c5c9fa296fbbbdb"
    ):
        raise ValueError("reference requires the pinned ECMWF checkpoint bytes")
    if digest(args.input) != args.input_sha256:
        raise ValueError("reference input bytes differ from the frozen fixture")

    import torch
    from earth2studio.models.auto import Package
    from earth2studio.models.px import AIFS2ENS
    from earth2studio.perturbation import Zero
    from earth2studio.run import ensemble

    provenance = json.loads(
        importlib.metadata.distribution("earth2studio").read_text("direct_url.json")
    )
    if (
        provenance.get("vcs_info", {}).get("commit_id")
        != "f8af4030cfd43ec79bf4c0cf6c23463a87bfef97"
    ):
        raise ValueError("reference requires the pinned Earth2Studio source")
    model = (
        AIFS2ENS.load_model(
            Package(str(args.checkpoint_dir), cache=False),
            preload_invariants=False,
            seed=None,
        )
        .to("cuda")
        .eval()
    )
    from anemoi.models.layers.sparse_projector import SparseProjector

    projectors = [
        module for module in model.modules() if isinstance(module, SparseProjector)
    ]
    if len(projectors) != 1:
        raise ValueError("expected one native checkpoint noise projector")

    def validate_device():
        # Anemoi moves this unregistered sparse matrix inside native forward.
        if projectors[0].projection_matrix.device.type != "cuda":
            raise ValueError("native noise projection did not execute on CUDA")

    actual = model.input_coords()
    for name, values in (
        ("variable", contract["input_variables"]),
        ("lead_time", np.array([-6, 0], dtype="timedelta64[h]")),
        ("lat", contract["lat"]),
        ("lon", contract["lon"]),
    ):
        if not np.array_equal(actual[name], values):
            raise ValueError(f"reference {name} differs from qualified contract")
    values = np.load(args.input, allow_pickle=False, mmap_mode="r")
    if values.dtype != np.float32 or values.shape != (
        1,
        2,
        len(contract["input_variables"]),
        len(contract["lat"]),
        len(contract["lon"]),
    ):
        raise ValueError("reference input shape/dtype differs from model")
    variables = list(model.output_coords(actual)["variable"])
    if variables != contract["output_variables"]:
        raise ValueError("reference output variable contract differs")
    io = CaptureIO(
        args.output,
        nensemble=args.nensemble,
        nsteps=args.nsteps,
        variables=variables,
        contract=contract,
        start_time=args.start_time,
        validate_device=validate_device,
    )
    source = FrozenSource(values, contract, args.start_time)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        torch.manual_seed(args.seed)
        ensemble(
            time=[args.start_time.removesuffix("Z")],
            nsteps=args.nsteps,
            nensemble=args.nensemble,
            prognostic=model,
            data=source,
            io=io,
            perturbation=Zero(),
            batch_size=min(args.nensemble, args.batch_size),
            device="cuda",
            verbose=False,
        )
    steps = io.finish()
    torch.cuda.synchronize()
    report = {
        "status": "passed",
        "seed_base": args.seed,
        "model_seed": model.seed,
        "nensemble": args.nensemble,
        "batch_size": min(args.nensemble, args.batch_size),
        "nsteps": args.nsteps,
        "input_sha256": args.input_sha256,
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "reference_script_sha256": digest(__file__),
        "steps": steps,
        "output_variables": variables,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": importlib.metadata.version("flash-attn"),
        "anemoi_models": importlib.metadata.version("anemoi-models"),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "numerical_policy": POLICY,
        "rng_policy": RNG_POLICY,
        "noise_projection_device": projectors[0].projection_matrix.device.type,
        "model_adapters": [],
        "gpu_name": torch.cuda.get_device_name(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
