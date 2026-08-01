# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-local adapter for the pinned PhysicsNeMo-CFD surface benchmark.

This module intentionally imports only the standard library, PyYAML, the
PhysicsNeMo Serve plugin SDK, and import-light shared CFD runtime helpers.
PhysicsNeMo-CFD is loaded only by the supervised child process in the dedicated
executor environment.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

import yaml

from physicsnemo_cfd_runtime.artifacts import (
    collect_globbed_outputs as _collect_globbed_outputs,
    media_type_for_path as _media_type,
    register_audit_outputs,
    register_validated_artifacts_once,
    validate_audit_outputs as _validated_audit_outputs,
)
from physicsnemo_cfd_runtime.safe_files import (
    bounded_log_tail as _bounded_log_tail,
    copy_verified_file_nofollow as _copy_verified_file_nofollow,
    create_attempt_directory,
    create_fresh_child_directory as _create_fresh_child_directory,
    ensure_empty_file as _ensure_empty_audit_log,
    validated_directory as _validated_directory,
    validated_output_file as _validated_output_file,
    validated_run_directory as _validated_run_directory,
    write_json_exclusive as _write_json_exclusive,
)
from physicsnemo_cfd_runtime.supervisor import (
    AbortProbe as _AbortProbe,
    run_supervised_process,
    supervisor_failure_diagnostics as _supervisor_failure_diagnostics,
)
from plugin_sdk import (
    ExecutionContext,
    PluginCancelledError,
    PluginWorkflow,
    PrepareContext,
    PrepareResult,
    RawRequest,
    ResourceProfile,
    build_execution_context,
    coerce_model,
    model_to_jsonable,
)

_CASE_ID_RE = re.compile(r"^run_([1-9][0-9]*)$")
_MAX_CASE_ID_LENGTH = 64
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SECRET_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "signature",
    "sig",
    "token",
    "x-amz-credential",
    "x-amz-security-token",
    "x-amz-signature",
}
_EXPECTED_MODEL_PACKAGES = {
    "domino_surface": (
        "hf://nvidia/domino_drivaerml@35b1bf1edafdaa2600d16182825890cd51c07427"
    ),
    "geotransolver_surface": (
        "hf://nvidia/geotransolver_drivaerml@626c1158e14f6994382924055aa871f863ff8a8c"
    ),
    "transolver_surface": (
        "hf://nvidia/transolver_drivaerml@96477aeb86d24c26ccf0797bca1b3851268017d0"
    ),
    "xmgn_surface": (
        "hf://nvidia/xmgn_drivaerml_surface@33909568711c0f60bd5fa6f8809e6d51c117f821"
    ),
    "fignet_surface": (
        "hf://nvidia/figconvnet_drivaerml_surface@49afb15f873c31134896f2e81fa8a3bff9c54790"
    ),
}


@dataclass(frozen=True)
class SurfaceCase:
    case_id: str
    mesh_uri: str
    sha256: str
    size_bytes: int
    geometry_uri: str
    geometry_sha256: str
    geometry_size_bytes: int


@dataclass(frozen=True)
class SurfaceBenchmarkInput:
    models: list[str]
    cases: list[SurfaceCase]
    metrics: list[str] = field(default_factory=list)
    seed: int = 42
    save_inference_mesh: bool = False
    visual_case_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseDigest:
    case_id: str
    sha256: str
    size_bytes: int
    geometry_sha256: str
    geometry_size_bytes: int


@dataclass(frozen=True)
class SurfaceBenchmarkOutput:
    report_path: str
    csv_path: str
    html_path: str
    resolved_config_path: str
    diagnostics_path: str
    model_names: list[str]
    case_ids: list[str]
    duration_seconds: float
    provider: dict[str, str]
    preset_sha256: str
    selected_metrics: list[str]
    case_digests: list[CaseDigest]
    registered_artifact_names: list[str]


def load_surface_plugin_config(manifest_path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable ``configuration`` manifest section."""
    path = Path(manifest_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"plugin manifest must be an object: {path}")
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("plugin manifest is missing configuration")

    required_top_level = {"provider", "benchmark"}
    missing = sorted(required_top_level - set(configuration))
    if missing:
        raise ValueError("configuration is missing: " + ", ".join(missing))

    config = _required_mapping(configuration, "benchmark")
    required_benchmark_fields = {
        "module_argv",
        "domain",
        "dataset",
        "models",
        "metrics",
        "limits",
        "execution",
        "benchmark_config",
        "outputs",
    }
    missing = sorted(required_benchmark_fields - set(config))
    if missing:
        raise ValueError("configuration.benchmark is missing: " + ", ".join(missing))
    if config["domain"] != "surface":
        raise ValueError("configuration.benchmark.domain must be 'surface'")

    provider = _required_mapping(configuration, "provider")
    expected_provider = {
        "repository": "https://github.com/NVIDIA/physicsnemo-cfd.git",
        "tag": "v0.0.2",
        "version": "0.0.2",
        "commit": "921f14dc2ac14c04aabffaba3290deb792379dd8",
        "physicsnemo_version": "2.1.1",
        "python_version": "3.12",
    }
    if provider != expected_provider:
        raise ValueError("configuration.provider does not match the approved pin set")

    expected_argv = [
        "{python}",
        "-m",
        "physicsnemo.cfd.evaluation.benchmarks.run",
        "--config",
        "{resolved_config}",
    ]
    if config["module_argv"] != expected_argv:
        raise ValueError(
            "configuration.benchmark.module_argv violates the fixed argv contract"
        )

    execution = _required_mapping(config, "execution")
    if int(execution.get("timeout_seconds", 0)) != 6 * 60 * 60:
        raise ValueError("surface benchmark timeout must remain exactly six hours")
    if int(execution.get("max_log_bytes", 0)) <= 0:
        raise ValueError(
            "configuration.benchmark.execution.max_log_bytes must be positive"
        )

    benchmark_config = _required_mapping(config, "benchmark_config")
    reproducibility = _required_mapping(benchmark_config, "reproducibility")
    if reproducibility.get("log_env") is not False:
        raise ValueError("environment logging must remain disabled")

    dataset = _required_mapping(config, "dataset")
    if dataset.get("name") != "drivaerml" or dataset.get("layout") != (
        "run_{number}/boundary_{number}.vtp"
    ):
        raise ValueError("surface benchmark requires the fixed DrivAerML VTP layout")
    model_assets = _required_mapping(config, "model_assets")
    if (
        model_assets.get("source")
        != "physicsnemo.cfd.evaluation.assets.builtin_packages"
        or model_assets.get("client_overrides_allowed") is not False
    ):
        raise ValueError("surface benchmark requires pinned built-in model packages")
    models = _required_mapping(config, "models")
    if set(models) != set(_EXPECTED_MODEL_PACKAGES):
        raise ValueError(
            "surface benchmark model catalog does not match the approved set"
        )
    for model_name, expected_package in _EXPECTED_MODEL_PACKAGES.items():
        model = _required_mapping(models, model_name)
        if (
            model.get("name") != model_name
            or model.get("inference_domain") != "surface"
            or model.get("package") != expected_package
        ):
            raise ValueError(f"surface model {model_name} violates its pinned contract")
    limits = _required_mapping(config, "limits")
    if limits.get("user_headers_allowed") is not False:
        raise ValueError("user-supplied prefetch headers must remain disabled")
    if limits.get("max_case_id_length") != _MAX_CASE_ID_LENGTH:
        raise ValueError(
            f"surface case_id length must remain capped at {_MAX_CASE_ID_LENGTH}"
        )
    if (
        execution.get("sequential") is not True
        or execution.get("distributed") is not False
    ):
        raise ValueError("surface benchmark must remain sequential and non-distributed")
    if execution.get("gpu_device") != "cuda:0":
        raise ValueError("surface benchmark must remain pinned to cuda:0")
    if execution.get("batch_size") != 1:
        raise ValueError("surface benchmark batch_size must remain pinned to 1")
    reports = _required_mapping(_required_mapping(config, "outputs"), "reports")
    if reports != {
        "json": "benchmark_results.json",
        "csv": "benchmark_results.csv",
        "html": "benchmark_results.html",
    }:
        raise ValueError("surface benchmark requires the fixed JSON/CSV/HTML reports")

    return copy.deepcopy(configuration)


def preset_sha256(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def normalize_surface_request(
    raw_fields: Mapping[str, Any], config: Mapping[str, Any]
) -> SurfaceBenchmarkInput:
    config = _benchmark_section(config)
    allowed_fields = {
        "models",
        "cases",
        "metrics",
        "seed",
        "save_inference_mesh",
        "visual_case_ids",
    }
    unknown = sorted(set(raw_fields) - allowed_fields)
    if unknown:
        raise ValueError("unsupported request fields: " + ", ".join(unknown))

    model_catalog = _required_mapping(config, "models")
    models = _string_list(raw_fields.get("models"), "models")
    if not models:
        raise ValueError("models must contain at least one model")
    if len(models) != len(set(models)):
        raise ValueError("models must not contain duplicates")
    unsupported_models = [name for name in models if name not in model_catalog]
    if unsupported_models:
        raise ValueError(
            "unsupported surface model(s): " + ", ".join(unsupported_models)
        )

    limits = _required_mapping(config, "limits")
    raw_cases = raw_fields.get("cases")
    if not isinstance(raw_cases, list):
        raise TypeError("cases must be an array")
    max_cases = int(limits["max_cases"])
    if not 1 <= len(raw_cases) <= max_cases:
        raise ValueError(f"cases must contain between 1 and {max_cases} items")

    cases: list[SurfaceCase] = []
    total_size = 0
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise TypeError(f"cases[{index}] must be an object")
        unknown_case_fields = sorted(
            set(raw_case)
            - {
                "case_id",
                "mesh_uri",
                "sha256",
                "size_bytes",
                "geometry_uri",
                "geometry_sha256",
                "geometry_size_bytes",
            }
        )
        if unknown_case_fields:
            raise ValueError(
                f"cases[{index}] contains unsupported fields: "
                + ", ".join(unknown_case_fields)
            )
        case_id = str(raw_case.get("case_id") or "")
        max_case_id_length = int(limits["max_case_id_length"])
        if len(case_id) > max_case_id_length:
            raise ValueError(
                f"cases[{index}].case_id must contain at most "
                f"{max_case_id_length} characters"
            )
        if _CASE_ID_RE.fullmatch(case_id) is None:
            raise ValueError(
                f"cases[{index}].case_id must match run_<positive integer> "
                "without leading zeros"
            )
        mesh_uri = _validate_mesh_uri(raw_case.get("mesh_uri"), index=index)
        geometry_uri = _validate_mesh_uri(
            raw_case.get("geometry_uri"), index=index, field_name="geometry_uri"
        )
        sha256 = str(raw_case.get("sha256") or "").lower()
        if _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(f"cases[{index}].sha256 must be 64 hexadecimal characters")
        geometry_sha256 = str(raw_case.get("geometry_sha256") or "").lower()
        if _SHA256_RE.fullmatch(geometry_sha256) is None:
            raise ValueError(
                f"cases[{index}].geometry_sha256 must be 64 hexadecimal characters"
            )
        size_bytes = _strict_int(
            raw_case.get("size_bytes"), f"cases[{index}].size_bytes"
        )
        geometry_size_bytes = _strict_int(
            raw_case.get("geometry_size_bytes"),
            f"cases[{index}].geometry_size_bytes",
        )
        max_object_size = int(limits["max_object_size_bytes"])
        if not 1 <= size_bytes <= max_object_size:
            raise ValueError(
                f"cases[{index}].size_bytes must be between 1 and {max_object_size}"
            )
        if not 1 <= geometry_size_bytes <= max_object_size:
            raise ValueError(
                f"cases[{index}].geometry_size_bytes must be between "
                f"1 and {max_object_size}"
            )
        total_size += size_bytes
        total_size += geometry_size_bytes
        cases.append(
            SurfaceCase(
                case_id=case_id,
                mesh_uri=mesh_uri,
                sha256=sha256,
                size_bytes=size_bytes,
                geometry_uri=geometry_uri,
                geometry_sha256=geometry_sha256,
                geometry_size_bytes=geometry_size_bytes,
            )
        )

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cases must not contain duplicate case_id values")
    if total_size > int(limits["max_request_size_bytes"]):
        raise ValueError("combined case size exceeds max_request_size_bytes")

    metrics_config = _required_mapping(config, "metrics")
    allowed_metrics = _string_list(metrics_config.get("allowed"), "metrics.allowed")
    raw_metrics = raw_fields.get("metrics", metrics_config.get("defaults"))
    metrics = _string_list(raw_metrics, "metrics")
    if not metrics:
        raise ValueError("metrics must contain at least one metric")
    if len(metrics) != len(set(metrics)):
        raise ValueError("metrics must not contain duplicates")
    unsupported_metrics = [name for name in metrics if name not in allowed_metrics]
    if unsupported_metrics:
        raise ValueError(
            "unsupported surface metric(s): " + ", ".join(unsupported_metrics)
        )

    seed = _strict_int(raw_fields.get("seed", 42), "seed")
    if not 0 <= seed <= int(limits["max_seed"]):
        raise ValueError(f"seed must be between 0 and {limits['max_seed']}")

    save_inference_mesh = raw_fields.get("save_inference_mesh", False)
    if not isinstance(save_inference_mesh, bool):
        raise TypeError("save_inference_mesh must be a boolean")

    visual_case_ids = _string_list(
        raw_fields.get("visual_case_ids", []), "visual_case_ids"
    )
    max_visual_cases = int(limits["max_visual_cases"])
    if len(visual_case_ids) > max_visual_cases:
        raise ValueError(
            f"visual_case_ids may contain at most {max_visual_cases} items"
        )
    if len(visual_case_ids) != len(set(visual_case_ids)):
        raise ValueError("visual_case_ids must not contain duplicates")
    unknown_visual_cases = [case for case in visual_case_ids if case not in case_ids]
    if unknown_visual_cases:
        raise ValueError("visual_case_ids must be a subset of cases[].case_id")

    return SurfaceBenchmarkInput(
        models=models,
        cases=cases,
        metrics=metrics,
        seed=seed,
        save_inference_mesh=save_inference_mesh,
        visual_case_ids=visual_case_ids,
    )


def build_prefetch_plan(inputs: SurfaceBenchmarkInput) -> list[dict[str, Any]]:
    """Create integrity-mode prefetch items; callers cannot add HTTP headers."""
    plan: list[dict[str, Any]] = []
    for case in inputs.cases:
        plan.append(
            {
                "kind": "http_fetch",
                "source_uri": case.mesh_uri,
                "target_artifact_name": f"surface-mesh-{case.case_id}",
                "required": True,
                "expected_sha256": case.sha256,
                "expected_size_bytes": case.size_bytes,
                "media_type": "application/vnd.vtk.vtp",
            }
        )
        plan.append(
            {
                "kind": "http_fetch",
                "source_uri": case.geometry_uri,
                "target_artifact_name": f"surface-geometry-{case.case_id}",
                "required": True,
                "expected_sha256": case.geometry_sha256,
                "expected_size_bytes": case.geometry_size_bytes,
                "media_type": "model/stl",
            }
        )
    return plan


def build_resolved_config(
    inputs: SurfaceBenchmarkInput,
    config: Mapping[str, Any],
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = _benchmark_section(config)
    model_catalog = _required_mapping(config, "models")
    benchmark_config = copy.deepcopy(_required_mapping(config, "benchmark_config"))
    if (
        _required_mapping(benchmark_config, "reproducibility").get("log_env")
        is not False
    ):
        raise ValueError("environment logging must remain disabled")

    dataset = _required_mapping(config, "dataset")
    execution = _required_mapping(config, "execution")
    reports = copy.deepcopy(_required_mapping(benchmark_config, "reports"))
    reports["enabled"] = bool(inputs.visual_case_ids)
    reports["visual_case_ids"] = list(inputs.visual_case_ids)
    if not inputs.visual_case_ids:
        reports["visuals"] = []

    return {
        "run": {
            "device": execution["gpu_device"],
            "output_dir": str(Path(output_dir).resolve()),
            "seed": inputs.seed,
            "batch_size": execution["batch_size"],
            "save_inference_mesh": inputs.save_inference_mesh,
            "distributed": execution["distributed"],
            "fail_on_all_skipped": True,
            "fail_on_any_metric_nan": False,
            "metrics_cache": {
                "enabled": True,
                "path": str((Path(output_dir) / "metrics_cache").resolve()),
            },
        },
        "benchmark": {
            "mode": "matrix",
            "models": [copy.deepcopy(model_catalog[name]) for name in inputs.models],
            "datasets": [
                {
                    "name": dataset["name"],
                    "root": str(Path(dataset_root).resolve()),
                    "case_ids": [case.case_id for case in inputs.cases],
                    "kwargs": copy.deepcopy(dataset["kwargs"]),
                }
            ],
            "reproducibility": copy.deepcopy(benchmark_config["reproducibility"]),
        },
        "output": copy.deepcopy(benchmark_config["output"]),
        "metrics": list(inputs.metrics),
        "reports": reports,
    }


def materialize_drivaerml_layout(
    inputs: SurfaceBenchmarkInput,
    prefetch_artifacts: Sequence[Mapping[str, Any]],
    *,
    dataset_root: str | Path,
    abort_requested: Callable[[], bool] | None = None,
) -> Path:
    root = Path(dataset_root)
    if root.is_symlink():
        raise ValueError(f"dataset root must not be a symlink: {root}")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    root_resolved = _validated_directory(root, label="dataset root")
    artifact_names = [
        str(artifact.get("name") or "") for artifact in prefetch_artifacts
    ]
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError("prefetch_artifacts must not contain duplicate names")
    by_name = dict(zip(artifact_names, prefetch_artifacts))

    abort_probe = (
        _AbortProbe(abort_requested, label="mesh staging")
        if abort_requested is not None
        else None
    )

    try:
        for case in inputs.cases:
            _raise_if_staging_aborted(abort_probe)
            artifact_name = f"surface-mesh-{case.case_id}"
            artifact = by_name.get(artifact_name)
            if artifact is None:
                raise ValueError(f"missing prefetched artifact '{artifact_name}'")

            source = Path(str(artifact.get("storage_path") or ""))
            verified_digest = str(
                artifact.get("sha256") or artifact.get("verified_sha256") or ""
            ).lower()
            if verified_digest != case.sha256:
                raise ValueError(
                    f"prefetched artifact digest mismatch for {case.case_id}"
                )

            suffix = _CASE_ID_RE.fullmatch(case.case_id)
            assert suffix is not None
            case_dir = _create_fresh_child_directory(root_resolved, case.case_id)
            destination = case_dir / f"boundary_{suffix.group(1)}.vtp"
            _copy_verified_file_nofollow(
                source,
                destination,
                expected_size=case.size_bytes,
                expected_sha256=case.sha256,
                artifact_label=case.case_id,
                cancellation_check=lambda: _raise_if_staging_aborted(abort_probe),
            )
            _raise_if_staging_aborted(abort_probe)

            geometry_artifact_name = f"surface-geometry-{case.case_id}"
            geometry_artifact = by_name.get(geometry_artifact_name)
            if geometry_artifact is None:
                raise ValueError(
                    f"missing prefetched artifact '{geometry_artifact_name}'"
                )
            geometry_source = Path(str(geometry_artifact.get("storage_path") or ""))
            verified_geometry_digest = str(
                geometry_artifact.get("sha256")
                or geometry_artifact.get("verified_sha256")
                or ""
            ).lower()
            if verified_geometry_digest != case.geometry_sha256:
                raise ValueError(
                    f"prefetched geometry artifact digest mismatch for {case.case_id}"
                )
            geometry_destination = case_dir / f"drivaer_{suffix.group(1)}.stl"
            _copy_verified_file_nofollow(
                geometry_source,
                geometry_destination,
                expected_size=case.geometry_size_bytes,
                expected_sha256=case.geometry_sha256,
                artifact_label=f"{case.case_id} geometry",
                cancellation_check=lambda: _raise_if_staging_aborted(abort_probe),
            )
            _raise_if_staging_aborted(abort_probe)
    finally:
        if abort_probe is not None:
            abort_probe.close()

    return root_resolved


def _raise_if_staging_aborted(abort_probe: _AbortProbe | None) -> None:
    if abort_probe is not None and abort_probe.check():
        raise PluginCancelledError("PhysicsNeMo-CFD mesh staging was cancelled")


def benchmark_command(
    config: Mapping[str, Any],
    resolved_config_path: str | Path,
    *,
    python_executable: str | None = None,
) -> list[str]:
    config = _benchmark_section(config)
    replacements = {
        "{python}": python_executable or sys.executable,
        "{resolved_config}": str(Path(resolved_config_path).resolve()),
    }
    argv: list[str] = []
    for token in config["module_argv"]:
        if token in replacements:
            argv.append(replacements[token])
        elif "{" in token or "}" in token:
            raise ValueError(f"unsupported module_argv placeholder: {token}")
        else:
            argv.append(str(token))
    return argv


def register_known_outputs(
    ctx: ExecutionContext,
    config: Mapping[str, Any],
    *,
    output_dir: str | Path,
    resolved_config_path: str | Path,
    diagnostics_path: str | Path,
    log_path: str | Path,
    include_meshes: bool,
    include_visuals: bool,
) -> dict[str, str]:
    config = _benchmark_section(config)
    managed_run_root = ctx.run_dir.resolve(strict=True)
    raw_output_root = Path(output_dir)
    if raw_output_root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {raw_output_root}")
    output_root = raw_output_root.resolve(strict=True)
    if not output_root.is_relative_to(managed_run_root):
        raise ValueError(f"output root escapes managed run directory: {output_root}")
    outputs_config = _required_mapping(config, "outputs")
    report_names = _required_mapping(outputs_config, "reports")

    required_reports: list[tuple[str, Path]] = []
    for logical_name in ("json", "csv", "html"):
        filename = report_names[logical_name]
        path = _validated_output_file(output_root, output_root / str(filename))
        if not path.exists():
            raise RuntimeError(f"PhysicsNeMo-CFD did not produce {filename}")
        required_reports.append((str(logical_name), path))

    json.loads(required_reports[0][1].read_text(encoding="utf-8"))
    audit_outputs = _validated_audit_outputs(
        ctx,
        resolved_config_path=resolved_config_path,
        diagnostics_path=diagnostics_path,
        log_path=log_path,
    )
    optional_outputs: list[tuple[str, Path, str]] = []
    for filename in outputs_config.get("optional_known_files", []):
        path = _validated_output_file(output_root, output_root / str(filename))
        if path.exists():
            optional_outputs.append((path.name, path, _media_type(path)))
    if include_visuals:
        optional_outputs.extend(
            _collect_globbed_outputs(
                output_root,
                str(outputs_config["visual_glob"]),
                allowed_suffixes={".png"},
            )
        )
    if include_meshes:
        for pattern in outputs_config["mesh_globs"]:
            optional_outputs.extend(
                _collect_globbed_outputs(
                    output_root,
                    str(pattern),
                    allowed_suffixes={".vtp"},
                )
            )

    registered: dict[str, str] = {}
    for logical_name, path in required_reports:
        ctx.outputs.register(
            path.name,
            path,
            media_type=_media_type(path),
            primary=logical_name == "json",
        )
        registered[str(logical_name)] = str(path)

    register_validated_artifacts_once(ctx, audit_outputs)
    seen_optional_names: set[str] = set()
    for name, path, media_type in optional_outputs:
        if name in seen_optional_names:
            continue
        seen_optional_names.add(name)
        ctx.outputs.register(name, path, media_type=media_type)
    return registered


class SurfaceBenchmarkWorkflow(PluginWorkflow):
    """Manifest-configured surface benchmark workflow."""

    cache_scope = "process"
    input_model = SurfaceBenchmarkInput
    output_model = SurfaceBenchmarkOutput
    manifest_path: Path | None = None

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self._configuration = load_surface_plugin_config(self.manifest_path)

    def _get_configuration(self) -> dict[str, Any]:
        configuration = getattr(self, "_configuration", None)
        if configuration is not None:
            return configuration
        manifest_path = getattr(self, "manifest_path", None)
        if manifest_path is None:
            raise RuntimeError("surface benchmark workflow has no manifest path")
        configuration = load_surface_plugin_config(manifest_path)
        self._configuration = configuration
        return configuration

    def prepare(self, request: RawRequest, ctx: PrepareContext) -> PrepareResult:
        inputs = normalize_surface_request(
            request.raw_fields, self._get_configuration()
        )
        return PrepareResult(
            inputs=inputs,
            resource_profile=ResourceProfile(
                executor_class="physicsnemo-cfd-gpu",
                gpus_required=1,
                memory_mb=65000,
                tags=["physicsnemo-cfd", "gpu"],
            ),
            prefetch_plan=build_prefetch_plan(inputs),
        )

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = coerce_model(
            SurfaceBenchmarkInput,
            ctx.get("parameters", {}),
            label="input",
        )
        exec_ctx = build_execution_context(ctx)
        result = self._execute_benchmark(
            inputs,
            exec_ctx,
            prefetch_artifacts=ctx.get("prefetch_artifacts", []),
        )
        return model_to_jsonable(result)

    def _execute_benchmark(
        self,
        inputs: SurfaceBenchmarkInput,
        ctx: ExecutionContext,
        *,
        prefetch_artifacts: Any,
    ) -> SurfaceBenchmarkOutput:
        if not isinstance(prefetch_artifacts, list):
            raise TypeError("prefetch_artifacts must be an array")
        configuration = self._get_configuration()
        run_root = _validated_run_directory(ctx.run_dir)
        attempt_root = create_attempt_directory(
            run_root,
            prefix="physicsnemo-cfd-surface-attempt-",
        )
        inputs_root = _create_fresh_child_directory(attempt_root, "inputs")
        dataset_root = _create_fresh_child_directory(inputs_root, "drivaerml")
        materialize_drivaerml_layout(
            inputs,
            prefetch_artifacts,
            dataset_root=dataset_root,
            abort_requested=ctx.abort_requested,
        )

        output_dir = _create_fresh_child_directory(attempt_root, "benchmark-output")
        resolved_config_path = attempt_root / "resolved_config.json"
        diagnostics_path = attempt_root / "benchmark_diagnostics.json"
        log_path = attempt_root / "benchmark.log"
        resolved_config = build_resolved_config(
            inputs,
            configuration,
            dataset_root=dataset_root,
            output_dir=output_dir,
        )
        _write_json_exclusive(resolved_config_path, resolved_config)

        argv = benchmark_command(
            configuration,
            resolved_config_path,
            python_executable=os.environ.get("PHYSICSNEMO_CFD_PYTHON_EXECUTABLE")
            or sys.executable,
        )
        execution = _required_mapping(_benchmark_section(configuration), "execution")
        supervisor_started = time.monotonic()
        try:
            process_result = run_supervised_process(
                argv,
                cwd=attempt_root,
                log_path=log_path,
                timeout_seconds=float(execution["timeout_seconds"]),
                termination_grace_seconds=float(execution["termination_grace_seconds"]),
                max_log_bytes=int(execution["max_log_bytes"]),
                abort_requested=ctx.abort_requested,
            )
        except Exception as exc:
            _ensure_empty_audit_log(log_path)
            diagnostics = _supervisor_failure_diagnostics(
                argv,
                exc,
                started=supervisor_started,
                log_path=log_path,
                max_log_bytes=int(execution["max_log_bytes"]),
            )
            _write_json_exclusive(diagnostics_path, diagnostics)
            register_audit_outputs(
                ctx,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
            )
            raise
        diagnostics = {
            "argv": argv,
            "returncode": process_result.returncode,
            "duration_seconds": process_result.duration_seconds,
            "log_bytes": process_result.log_bytes,
            "log_truncated": process_result.log_truncated,
            "timed_out": process_result.timed_out,
            "cancelled": process_result.cancelled,
        }
        _write_json_exclusive(diagnostics_path, diagnostics)

        if process_result.cancelled:
            register_audit_outputs(
                ctx,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
            )
            raise PluginCancelledError("PhysicsNeMo-CFD benchmark was cancelled")
        if process_result.timed_out:
            register_audit_outputs(
                ctx,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
            )
            raise TimeoutError(
                "PhysicsNeMo-CFD benchmark exceeded the six-hour timeout"
            )
        if process_result.returncode != 0:
            register_audit_outputs(
                ctx,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
            )
            tail = _bounded_log_tail(log_path)
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(
                f"PhysicsNeMo-CFD benchmark exited with code "
                f"{process_result.returncode}{detail}"
            )

        registered_before = len(ctx.outputs.registered_outputs())
        try:
            registered = register_known_outputs(
                ctx,
                configuration,
                output_dir=output_dir,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
                include_meshes=inputs.save_inference_mesh,
                include_visuals=bool(inputs.visual_case_ids),
            )
        except Exception:
            register_audit_outputs(
                ctx,
                resolved_config_path=resolved_config_path,
                diagnostics_path=diagnostics_path,
                log_path=log_path,
            )
            raise
        report_path = registered["json"]
        registered_artifact_names = [
            artifact.name
            for artifact in ctx.outputs.registered_outputs()[registered_before:]
        ]
        provider = {
            str(key): str(value)
            for key, value in _required_mapping(configuration, "provider").items()
        }
        return SurfaceBenchmarkOutput(
            report_path=report_path,
            csv_path=registered["csv"],
            html_path=registered["html"],
            resolved_config_path=str(resolved_config_path),
            diagnostics_path=str(diagnostics_path),
            model_names=list(inputs.models),
            case_ids=[case.case_id for case in inputs.cases],
            duration_seconds=process_result.duration_seconds,
            provider=provider,
            preset_sha256=preset_sha256(configuration),
            selected_metrics=list(inputs.metrics),
            case_digests=[
                CaseDigest(
                    case_id=case.case_id,
                    sha256=case.sha256,
                    size_bytes=case.size_bytes,
                    geometry_sha256=case.geometry_sha256,
                    geometry_size_bytes=case.geometry_size_bytes,
                )
                for case in inputs.cases
            ],
            registered_artifact_names=registered_artifact_names,
        )


def _required_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration.{key} must be an object")
    return dict(value)


def _benchmark_section(config: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise ValueError("configuration.benchmark must be an object")
    return dict(benchmark)


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise TypeError(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    return result


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _validate_mesh_uri(value: Any, *, index: int, field_name: str = "mesh_uri") -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"cases[{index}].{field_name} must be a non-empty string")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"cases[{index}].{field_name} must use https://")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"cases[{index}].{field_name} must not contain userinfo")
    if parsed.fragment:
        raise ValueError(f"cases[{index}].{field_name} must not contain a fragment")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError:
        pass
    else:
        raise ValueError(f"cases[{index}].{field_name} must use a DNS hostname")
    query_keys = {
        key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
    }
    if query_keys & _SECRET_QUERY_KEYS:
        raise ValueError(f"cases[{index}].{field_name} contains a secret-bearing query")
    return value
