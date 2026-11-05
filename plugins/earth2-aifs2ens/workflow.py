# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A complete AIFS ENS v2 ensemble in one GPU execute task."""

from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import shutil
import threading
import uuid

from plugin_sdk import (
    PluginWorkflow,
    build_execution_context,
    cleanup_earth2_runtime_resources,
    cleanup_python_and_torch_runtime,
)

PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from aifs2ens_artifacts import (  # noqa: E402
    InputContract,
    load_ensemble_input,
    stage_inputs,
    write_json,
)
from aifs2ens_contract import ForecastRequest, NUMERICAL_POLICY, RNG_POLICY  # noqa: E402


def _positive_setting(name, default=None):
    value = os.environ.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be configured as a positive integer") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be configured as a positive integer")
    return parsed


def _gpu_profile():
    return {
        "executor_class": "earth2-gpu",
        "device_kind": "gpu",
        "gpus_required": 1,
        "memory_mb": _positive_setting("AIFS2ENS_GPU_MEMORY_MB"),
        "tags": ["earth2", "aifs2ens", "gpu"],
    }


def _validate_batch_size(request):
    maximum = _positive_setting("AIFS2ENS_MAX_BATCH_SIZE", "2")
    if request.batch_size > maximum:
        raise ValueError(
            f"batch_size {request.batch_size} exceeds the deployment's qualified maximum "
            f"{maximum}; specify batch_size <= {maximum}"
        )


class AIFS2ENSWorkflow(PluginWorkflow):
    input_model = ForecastRequest
    cache_scope = "process"
    model_cache_names = ["AIFS2ENS"]

    def __init__(self):
        self._package = None
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def prepare(self, request, ctx):
        inputs = ForecastRequest.model_validate(dict(request.raw_fields))
        profile = _gpu_profile()  # Validate deployment before downloading inputs.
        _validate_batch_size(inputs)
        # The hook runner reserves stdout for its JSON result.
        with redirect_stdout(sys.stderr):
            staged = self._stage_inputs(inputs, ctx)
        return {
            "operation": "forecast_ensemble",
            "parameters": staged,
            "resource_profile": profile,
            "batch_profile": {"enabled": False},
        }

    def _contract(self):
        path = os.environ.get("AIFS2ENS_INPUT_CONTRACT")
        if not path:
            raise RuntimeError(
                "AIFS2ENS_INPUT_CONTRACT must name the qualified input contract"
            )
        return InputContract.model_validate_json(Path(path).read_text())

    def _stage_inputs(self, request, ctx):
        from aifs2ens_data import WeatherSource

        return stage_inputs(
            request,
            self._contract(),
            ctx.run_dir,
            WeatherSource(),
            max_output_bytes=_positive_setting(
                "AIFS2ENS_MAX_OUTPUT_BYTES", str(4 * 1024**3)
            ),
        )

    def _load_model(self, device):
        from aifs2ens_runtime import load_model

        return load_model(self._contract(), device)

    def _runtime_identity(self):
        from aifs2ens_runtime import runtime_identity

        return runtime_identity(self._model, self._device)

    def _ensure_model(self, device):
        if self._model is None:
            self._package, self._model = self._load_model(device)
            self._device = device
        elif self._device != device:
            raise RuntimeError("a cached AIFS2ENS worker cannot change GPU device")

    def warmup(self, ctx):
        device = str(ctx.get("device", "cpu"))
        if not device.startswith("cuda"):
            return {"model_names": []}
        with self._lock:
            self._ensure_model(device)
        return {"model_names": ["AIFS2ENS"]}

    def execute(self, ctx):
        params = dict(ctx.get("parameters") or {})
        execution = build_execution_context(ctx)
        if ctx.get("operation") != "forecast_ensemble":
            raise ValueError("unsupported AIFS2ENS operation")
        if set(params) != {"manifest_path", "manifest_sha256"}:
            raise ValueError(
                "ensemble execution requires an exact staged-input identity"
            )
        loaded = load_ensemble_input(params["manifest_path"], params["manifest_sha256"])
        _validate_batch_size(loaded.request)
        with self._lock:
            return self._run_ensemble(loaded, execution)

    def _run_ensemble(self, loaded, ctx):
        from aifs2ens_outputs import validate_ensemble_output
        from aifs2ens_rollout import run_ensemble_rollout
        from aifs2ens_storage import ZarrEnsembleSink

        if loaded.contract != self._contract():
            raise ValueError("staged input contract differs from the deployed model")
        self._ensure_model(self._device or "cuda")
        ctx.run_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.run_dir / f"forecast-ensemble-{uuid.uuid4().hex}.zarr"
        try:
            sink = ZarrEnsembleSink(path, loaded)
            run_ensemble_rollout(
                self._model,
                loaded,
                sink,
                device=self._device,
                abort_requested=ctx.abort_requested,
            )
            validate_ensemble_output(path, loaded, expected_hashes=sink.field_hashes)
            request = loaded.request
            payload = {
                "complete": True,
                "dataset_path": str(path),
                "nensemble": request.nensemble,
                "nsteps": request.nsteps,
                "batch_size": request.batch_size,
                "seed_base": request.seed_base,
                "ifs_member_id": request.ifs_member_id,
                "manifest_sha256": loaded.manifest_sha256,
                "input_sha256": loaded.input_sha256,
                "runtime": self._runtime_identity(),
                "checkpoint_sha256": loaded.contract.checkpoint_sha256,
                "numerical_policy": NUMERICAL_POLICY,
                "rng_policy": RNG_POLICY,
            }
            write_json(
                ctx.run_dir / "forecast_metadata.json",
                {
                    **payload,
                    "request": request.model_dump(mode="json"),
                    "input_provenance": loaded.provenance,
                },
            )
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        ctx.outputs.register(
            "forecast_dataset", path, media_type="application/x-zarr", primary=True
        )
        return payload

    def cleanup(self):
        with self._lock:
            if self._package is not None:
                cleanup_earth2_runtime_resources(self._package)
            self._model = None
            self._package = None
            self._device = None
            cleanup_python_and_torch_runtime()


WORKFLOW = AIFS2ENSWorkflow
