"""Capture the *installed pinned provider* while it runs its normal benchmark.

The wrapper's own prepare/predict/decode and shared metric evaluator execute
unchanged. Hooks observe values; they never replace model inputs, RNG results,
forwards, decoding or metric definitions. Capture copies/synchronizations make
this correctness evidence unsuitable for performance measurements.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np

from qa.native_inference.cfd_contract import (
    ARRAY_NAMES,
    ASSET_PINS,
    FieldContractError,
    MODEL_REVISION,
    array_descriptor,
    block_plan,
    inference_seed,
    validate_evidence,
    validate_reference_request,
    validate_report,
    verify_file,
)
from qa.scripts.cfd_parity_contract import sha256_file, write_json_atomic


def verify_local_model_assets(assets):
    if not isinstance(assets, Mapping) or set(assets) != set(ASSET_PINS):
        raise FieldContractError("local model assets require checkpoint and statistics")
    verified = {}
    for name, pin in ASSET_PINS.items():
        record = assets[name]
        if not isinstance(record, Mapping) or any(
            record.get(key) != pin[key] for key in ("sha256", "size_bytes")
        ):
            raise FieldContractError(f"local {name} identity differs from the pin")
        if not isinstance(record.get("path"), str):
            raise FieldContractError(f"local {name} path must be a string")
        path = Path(record["path"])
        if not path.is_absolute() or path.name != pin["filename"]:
            raise FieldContractError(
                f"local {name} must use its pinned absolute filename"
            )
        # Preserve the checkpoint filename (and parsed epoch) when a provider
        # cache uses a named symlink to content-addressed blob storage.
        path = path.parent.resolve() / path.name
        verified[name] = {"path": str(path), **verify_file(path, pin)}
    return verified


def local_model_assets(checkpoint, statistics):
    if checkpoint is None and statistics is None:
        return None
    if checkpoint is None or statistics is None:
        raise FieldContractError("--checkpoint and --stats must be supplied together")
    return verify_local_model_assets(
        {
            name: {
                "path": str(Path(path).absolute()),
                **{key: ASSET_PINS[name][key] for key in ("sha256", "size_bytes")},
            }
            for name, path in (("checkpoint", checkpoint), ("statistics", statistics))
        }
    )


def reference_model_asset_arguments(reference_path):
    """Reuse captured verified paths for evaluation; retain legacy remote mode."""
    reference = json.loads(Path(reference_path).read_text())
    if "local_model_assets" not in reference:
        return []
    assets = verify_local_model_assets(reference["local_model_assets"])
    return [
        "--checkpoint",
        assets["checkpoint"]["path"],
        "--stats",
        assets["statistics"]["path"],
    ]


def build_capture_config(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    input_root: Path,
    output_dir: Path,
    *,
    model_assets=None,
) -> dict[str, Any]:
    validate_reference_request(profile, request)
    config = copy.deepcopy(profile["config"])
    dataset = config.pop("dataset")
    dataset.update(
        root=str(input_root.resolve()),
        case_ids=[case["case_id"] for case in request["cases"]],
    )
    models = config.pop("models")
    if model_assets is not None:
        assets = verify_local_model_assets(model_assets)
        # Pinned provider resolve_model_assets returns explicit checkpoint and
        # stats_path before constructing its remote Package. Keep package as the
        # model identity while avoiding redundant HF metadata/token requests.
        models[0].update(
            checkpoint=assets["checkpoint"]["path"],
            stats_path=assets["statistics"]["path"],
        )
    reproducibility = config.pop("reproducibility")
    run = config["run"]
    run.update(
        output_dir=str(output_dir.resolve()),
        seed=request["seed"],
        save_inference_mesh=request["save_inference_mesh"],
        metrics_cache={"enabled": False},
    )
    config["benchmark"] = {
        "mode": "matrix",
        "models": models,
        "datasets": [dataset],
        "reproducibility": reproducibility,
    }
    config["metrics"] = list(request["metrics"])
    config["reports"]["visual_case_ids"] = list(request["visual_case_ids"])
    return config


def verify_provider(profile: Mapping[str, Any]) -> dict[str, Any]:
    provider = profile["provider"]
    module = importlib.import_module(provider["module"])
    distribution = importlib.metadata.distribution(provider["distribution"])
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    checks = {
        "version": str(getattr(module, "__version__", "")),
        "commit": direct_url.get("vcs_info", {}).get("commit_id"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
    }
    for key, value in checks.items():
        if value != provider[key]:
            raise FieldContractError(
                f"installed provider {key} mismatch: {value!r} != {provider[key]!r}"
            )
    return {**provider, **checks}


def environment() -> dict[str, Any]:
    import torch
    import vtkmodules.all as vtk

    if not torch.cuda.is_available():
        raise FieldContractError(
            "reference capture requires CUDA; run it in the Lepton GPU job"
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "vtk": vtk.vtkVersion.GetVTKVersion(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "math": {
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "preferred_blas_library": str(torch.backends.cuda.preferred_blas_library())
            .split(".")[-1]
            .lower(),
        },
    }


class ProviderCapture:
    def __init__(self, *, profile, request, root: Path, image: str, source_sha: str):
        self.profile, self.request, self.root = profile, request, root
        self.manifest = {
            "schema_version": 1,
            "profile_id": profile["profile_id"],
            "runtime": "python-provider",
            "model": "transolver_surface",
            "model_revision": MODEL_REVISION,
            "provider": profile["provider"],
            "source_sha": source_sha,
            "image": image,
            "cases": [],
            "capture_is_performance_measurement": False,
        }
        self.current: dict[str, Any] | None = None
        self.load_count = 0

    def _save(self, case_id: str, name: str, value) -> dict[str, Any]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.kind == "f":
            if value.dtype != np.float32:
                raise FieldContractError(f"{name} must preserve provider FP32 values")
            value = value.astype("<f4", copy=False)
        else:
            value = value.astype("<i8", copy=False)
        directory = self.root / case_id
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        return array_descriptor(path, self.root, value)

    @contextmanager
    def observe(self):
        import torch
        from physicsnemo.cfd.evaluation.models.wrappers.transolver.wrapper import (
            TransolverWrapper,
        )

        original_load = TransolverWrapper.load
        original_prepare = TransolverWrapper.prepare_inputs
        original_predict = TransolverWrapper.predict
        original_decode = TransolverWrapper.decode_outputs
        capture = self

        def load(wrapper, checkpoint_path, stats_path, device, **kwargs):
            if capture.load_count:
                raise FieldContractError("Transolver reference loaded more than once")
            checkpoint = Path(checkpoint_path)
            if checkpoint.is_dir():
                checkpoint = checkpoint / ASSET_PINS["checkpoint"]["filename"]
            stats = Path(stats_path)
            if stats.is_dir():
                stats = stats / ASSET_PINS["statistics"]["filename"]
            capture.manifest["checkpoint"] = verify_file(
                checkpoint, ASSET_PINS["checkpoint"]
            )
            capture.manifest["statistics"] = verify_file(
                stats, ASSET_PINS["statistics"]
            )
            capture.manifest["local_model_assets"] = local_model_assets(
                checkpoint, stats
            )
            result = original_load(
                wrapper, checkpoint_path, stats_path, device, **kwargs
            )
            if (
                wrapper._batch_resolution != 60000
                or wrapper._inference_mode != "surface"
                or wrapper._cuda_bf16_autocast
            ):
                raise FieldContractError(
                    "provider effective model configuration changed"
                )
            capture.load_count += 1
            return result

        def prepare(wrapper, case):
            case_id = case.case_id
            if case_id not in capture.profile["native_reference"]["case_ids"]:
                raise FieldContractError(f"unexpected provider case {case_id}")
            request_case = next(
                item for item in capture.request["cases"] if item["case_id"] == case_id
            )
            verify_file(
                case.mesh_path,
                {
                    "sha256": request_case["sha256"],
                    "size_bytes": request_case["size_bytes"],
                },
            )
            # CanonicalCase.reference_geometry is the loaded boundary mesh, not
            # a filesystem STL path. Match build_surface_data_dict's run layout.
            stl = (
                Path(case.mesh_path).parent
                / f"drivaer_{case_id.removeprefix('run_')}.stl"
            )
            verify_file(
                stl,
                {
                    "sha256": request_case["geometry_sha256"],
                    "size_bytes": request_case["geometry_size_bytes"],
                },
            )
            result = original_prepare(wrapper, case)
            batch = result["batch"]
            count = int(batch["embeddings"].shape[1])
            if case.reference_geometry is None or count != int(
                case.reference_geometry.n_cells
            ):
                raise FieldContractError(
                    "provider did not prepare every source surface cell"
                )
            capture.current = {
                "case_id": case_id,
                "point_count": count,
                "extent": "all-source-cells",
                "run_seed": capture.request["seed"],
                "inference_seed": inference_seed(capture.request["seed"], case_id),
                "batch_resolution": 60000,
                "block_points": block_plan(count),
                "output_location": "cell",
                "source_order": True,
                "inputs": {
                    key: request_case[key]
                    for key in (
                        "sha256",
                        "size_bytes",
                        "geometry_sha256",
                        "geometry_size_bytes",
                    )
                },
                "arrays": {
                    "fx": capture._save(case_id, "fx", batch["fx"]),
                    "embedding": capture._save(
                        case_id, "embedding", batch["embeddings"]
                    ),
                    "source_cell_ids": capture._save(
                        case_id, "source_cell_ids", np.arange(count, dtype=np.int64)
                    ),
                },
            }
            capture.manifest["environment"] = environment()
            return result

        def predict(wrapper, model_input):
            if capture.current is None:
                raise FieldContractError(
                    "predict executed without captured preparation"
                )
            case = capture.current
            outputs, permutations = [], []
            randperm = torch.randperm

            def observe_permutation(*args, **kwargs):
                indices = randperm(*args, **kwargs)
                permutations.append(indices.detach().clone())
                return indices

            def observe_forward(module, arguments, output):
                outputs.append(output.detach().cpu())

            hook = wrapper._model.register_forward_hook(observe_forward)
            try:
                with patch.object(torch, "randperm", observe_permutation):
                    result = original_predict(wrapper, model_input)
            finally:
                hook.remove()
            if (
                len(permutations) != 1
                or [int(value.shape[1]) for value in outputs] != case["block_points"]
            ):
                raise FieldContractError(
                    "provider permutation/block execution does not match frozen contract"
                )
            permutation = permutations[0].cpu()
            standardized = torch.empty((case["point_count"], 4), dtype=torch.float32)
            standardized[permutation] = torch.cat(outputs, dim=1).squeeze(0)
            case["arrays"]["permutation"] = capture._save(
                case["case_id"], "permutation", permutation
            )
            case["arrays"]["standardized"] = capture._save(
                case["case_id"], "standardized", standardized
            )
            return result

        def decode(wrapper, raw_output, case, model_input=None):
            result = original_decode(wrapper, raw_output, case, model_input)
            if capture.current is None or capture.current["case_id"] != case.case_id:
                raise FieldContractError("decode case does not match prepared inputs")
            fields = np.column_stack((result["pressure"], result["shear_stress"]))
            capture.current["arrays"]["physical"] = capture._save(
                case.case_id, "physical", fields
            )
            if set(capture.current["arrays"]) != set(ARRAY_NAMES):
                raise FieldContractError("provider capture incomplete")
            capture.manifest["cases"].append(capture.current)
            capture.current = None
            return result

        with (
            patch.object(TransolverWrapper, "load", load),
            patch.object(TransolverWrapper, "prepare_inputs", prepare),
            patch.object(TransolverWrapper, "predict", predict),
            patch.object(TransolverWrapper, "decode_outputs", decode),
        ):
            yield

    def finish(self, report: Any) -> Path:
        validate_report(self.profile, report)
        validate_evidence(self.profile, self.manifest, self.root)
        report_path = self.root / "benchmark-output" / "benchmark_results.json"
        if json.loads(report_path.read_text()) != report:
            raise FieldContractError(
                "saved scientific report differs from provider return value"
            )
        self.manifest["report"] = {
            "path": report_path.relative_to(self.root).as_posix(),
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        }
        path = self.root / "fields.json"
        write_json_atomic(path, self.manifest)
        return path
