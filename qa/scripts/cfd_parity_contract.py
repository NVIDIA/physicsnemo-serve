# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract for PhysicsNeMo-CFD wrapper-to-provider parity QA."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCHEMA_VERSION = 2
JsonObject = dict[str, Any]
MetricKey = tuple[str, str, str, str, str]
PROVIDER_PROVENANCE_FIELDS = (
    "repository",
    "tag",
    "version",
    "commit",
    "physicsnemo_version",
    "python_version",
)


class ParityContractError(ValueError):
    """Raised when parity evidence or configuration violates the contract."""


def read_json_object(path: str | Path) -> JsonObject:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParityContractError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParityContractError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ParityContractError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParityContractError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ParityContractError(f"{label} must be a SHA-256 digest")
    return digest.lower()


def validate_image_digest_reference(value: object, label: str = "image") -> str:
    image = _string(value, label)
    if re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image) is None:
        raise ParityContractError(
            f"{label} must be an immutable image reference ending in "
            "@sha256:<64 lowercase hex characters>"
        )
    return image


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityContractError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ParityContractError(f"{label} must be finite and non-negative")
    return result


def validate_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ParityContractError(f"profile schema_version must be {SCHEMA_VERSION}")
    for key in ("profile_id", "workflow_id", "domain"):
        _string(profile.get(key), f"profile.{key}")

    provider = _mapping(profile.get("provider"), "profile.provider")
    for key in ("distribution", "module", *PROVIDER_PROVENANCE_FIELDS):
        _string(provider.get(key), f"profile.provider.{key}")
    _sha256(profile.get("preset_sha256"), "profile.preset_sha256")

    rest = _mapping(profile.get("rest"), "profile.rest")
    for key in ("service", "suite", "evidence_subdir", "request_path"):
        _string(rest.get(key), f"profile.rest.{key}")

    request = _mapping(profile.get("request"), "profile.request")
    for key in ("models", "metrics", "visual_case_ids"):
        values = _list(request.get(key), f"profile.request.{key}")
        if any(not isinstance(value, str) or not value for value in values):
            raise ParityContractError(
                f"profile.request.{key} must contain non-empty strings"
            )
    if isinstance(request.get("seed"), bool) or not isinstance(
        request.get("seed"), int
    ):
        raise ParityContractError("profile.request.seed must be an integer")
    if not isinstance(request.get("save_inference_mesh"), bool):
        raise ParityContractError(
            "profile.request.save_inference_mesh must be a boolean"
        )
    case_pattern = _string(
        request.get("case_id_pattern"), "profile.request.case_id_pattern"
    )
    try:
        re.compile(case_pattern)
    except re.error as exc:
        raise ParityContractError(
            f"profile.request.case_id_pattern is invalid: {exc}"
        ) from exc
    report_metric_outputs = _mapping(
        profile.get("report_metric_outputs"),
        "profile.report_metric_outputs",
    )
    if set(report_metric_outputs) != set(request["metrics"]):
        raise ParityContractError(
            "profile.report_metric_outputs must cover every requested metric exactly"
        )
    for metric, outputs in report_metric_outputs.items():
        names = _list(outputs, f"profile.report_metric_outputs.{metric}")
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names) != len(set(names))
        ):
            raise ParityContractError(
                f"profile.report_metric_outputs.{metric} must contain "
                "unique non-empty strings"
            )

    layout = _list(profile.get("input_layout"), "profile.input_layout")
    if not layout:
        raise ParityContractError("profile.input_layout must not be empty")
    for index, item in enumerate(layout):
        entry = _mapping(item, f"profile.input_layout[{index}]")
        for key in ("digest_field", "size_field", "relative_path"):
            _string(entry.get(key), f"profile.input_layout[{index}].{key}")

    runner = _mapping(profile.get("runner"), "profile.runner")
    for key in ("python", "module"):
        _string(runner.get(key), f"profile.runner.{key}")
    timeout = runner.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ParityContractError(
            "profile.runner.timeout_seconds must be a positive integer"
        )
    environment = _mapping(runner.get("environment", {}), "profile.runner.environment")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in environment.items()
    ):
        raise ParityContractError(
            "profile.runner.environment must map strings to strings"
        )

    config = _mapping(profile.get("config"), "profile.config")
    for key in ("run", "models", "dataset", "reproducibility", "output", "reports"):
        expected_type = list if key == "models" else Mapping
        if not isinstance(config.get(key), expected_type):
            raise ParityContractError(
                f"profile.config.{key} must be "
                f"{'an array' if key == 'models' else 'an object'}"
            )
    model_names = [
        _string(
            _mapping(model, "profile.config model").get("name"),
            "profile.config model name",
        )
        for model in _list(config["models"], "profile.config.models")
    ]
    if model_names != request["models"]:
        raise ParityContractError(
            "profile.config model names must match profile.request.models"
        )
    dataset = _mapping(config["dataset"], "profile.config.dataset")
    _string(dataset.get("name"), "profile.config.dataset.name")

    comparison = _mapping(profile.get("comparison"), "profile.comparison")
    _number(comparison.get("default_rtol"), "profile.comparison.default_rtol")
    _number(comparison.get("default_atol"), "profile.comparison.default_atol")
    metric_tolerances = _mapping(
        comparison.get("metrics", {}), "profile.comparison.metrics"
    )
    for metric, tolerance in metric_tolerances.items():
        _string(metric, "profile.comparison metric name")
        values = _mapping(tolerance, f"profile.comparison.metrics.{metric}")
        _number(values.get("rtol"), f"profile.comparison.metrics.{metric}.rtol")
        _number(values.get("atol"), f"profile.comparison.metrics.{metric}.atol")
    model_tolerances = _mapping(
        comparison.get("models", {}), "profile.comparison.models"
    )
    for model, tolerance in model_tolerances.items():
        if model not in request["models"]:
            raise ParityContractError(
                f"profile.comparison.models contains unknown model {model!r}"
            )
        values = _mapping(tolerance, f"profile.comparison.models.{model}")
        _number(
            values.get("default_rtol"),
            f"profile.comparison.models.{model}.default_rtol",
        )
        _number(
            values.get("default_atol"),
            f"profile.comparison.models.{model}.default_atol",
        )
        overrides = _mapping(
            values.get("metrics", {}),
            f"profile.comparison.models.{model}.metrics",
        )
        for metric, metric_tolerance in overrides.items():
            metric_values = _mapping(
                metric_tolerance,
                f"profile.comparison.models.{model}.metrics.{metric}",
            )
            _number(
                metric_values.get("rtol"),
                f"profile.comparison.models.{model}.metrics.{metric}.rtol",
            )
            _number(
                metric_values.get("atol"),
                f"profile.comparison.models.{model}.metrics.{metric}.atol",
            )


def validate_request(profile: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    expected = _mapping(profile.get("request"), "profile.request")
    for key in (
        "models",
        "metrics",
        "seed",
        "save_inference_mesh",
        "visual_case_ids",
    ):
        if request.get(key) != expected.get(key):
            raise ParityContractError(
                f"parity request {key!r} does not match parity profile: "
                f"{request.get(key)!r} != {expected.get(key)!r}"
            )
    cases = _list(request.get("cases"), "parity request cases")
    if not cases:
        raise ParityContractError("parity request cases must not be empty")
    pattern = re.compile(str(expected["case_id_pattern"]))
    seen: set[str] = set()
    for index, value in enumerate(cases):
        case = _mapping(value, f"parity request cases[{index}]")
        case_id = _string(case.get("case_id"), f"parity request cases[{index}].case_id")
        if pattern.fullmatch(case_id) is None:
            raise ParityContractError(
                f"case_id {case_id!r} does not match {pattern.pattern!r}"
            )
        if case_id in seen:
            raise ParityContractError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        for layout_index, layout_value in enumerate(
            _list(profile.get("input_layout"), "profile.input_layout")
        ):
            entry = _mapping(
                layout_value,
                f"profile.input_layout[{layout_index}]",
            )
            digest_field = _string(
                entry.get("digest_field"),
                f"profile.input_layout[{layout_index}].digest_field",
            )
            size_field = _string(
                entry.get("size_field"),
                f"profile.input_layout[{layout_index}].size_field",
            )
            _sha256(
                case.get(digest_field),
                f"parity request cases[{index}].{digest_field}",
            )
            size = case.get(size_field)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ParityContractError(
                    f"parity request cases[{index}].{size_field} "
                    "must be a positive integer"
                )


def mount_relative_path(path: str, mount_target: str) -> str:
    candidate = PurePosixPath(_string(path, "artifact path"))
    root = PurePosixPath(_string(mount_target, "mount target"))
    if not candidate.is_absolute() or not root.is_absolute():
        raise ParityContractError("artifact path and mount target must be absolute")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ParityContractError(
            f"artifact path {path!r} is outside mount target {mount_target!r}"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ParityContractError(f"unsafe mount-relative path: {relative}")
    return relative.as_posix()


def resolve_existing_mount_path(mount_target: str, relative_path: str) -> Path:
    relative = PurePosixPath(_string(relative_path, "mount-relative path"))
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ParityContractError(f"unsafe mount-relative path: {relative_path!r}")
    root = Path(mount_target).resolve(strict=True)
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ParityContractError(
            f"resolved path {candidate} escapes mount target {root}"
        ) from exc
    return candidate


def output_by_name(results: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    execution = _mapping(results.get("execution"), "results.execution")
    outputs = _list(execution.get("outputs"), "results.execution.outputs")
    matches = [
        _mapping(output, "results.execution output")
        for output in outputs
        if isinstance(output, Mapping) and output.get("name") == name
    ]
    if len(matches) != 1:
        raise ParityContractError(
            f"results must contain exactly one output named {name!r}"
        )
    return matches[0]


def resolved_dataset_root(resolved_config: Mapping[str, Any], dataset_name: str) -> str:
    benchmark = _mapping(resolved_config.get("benchmark"), "resolved_config.benchmark")
    datasets = _list(benchmark.get("datasets"), "resolved_config.benchmark.datasets")
    matches = [
        _mapping(dataset, "resolved_config dataset")
        for dataset in datasets
        if isinstance(dataset, Mapping) and dataset.get("name") == dataset_name
    ]
    if len(matches) != 1:
        raise ParityContractError(
            f"resolved config must contain one {dataset_name!r} dataset"
        )
    return _string(matches[0].get("root"), "resolved dataset root")


def _validate_provenance(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    value: object,
    *,
    source: str,
) -> JsonObject:
    provenance = _mapping(value, f"{source} provenance")
    provider = _mapping(
        provenance.get("provider"),
        f"{source} provenance.provider",
    )
    expected_provider = _mapping(profile.get("provider"), "profile.provider")
    for key in PROVIDER_PROVENANCE_FIELDS:
        if provider.get(key) != expected_provider.get(key):
            raise ParityContractError(
                f"{source} provider {key} does not match the parity profile"
            )

    preset_sha256 = _sha256(
        provenance.get("preset_sha256"),
        f"{source} provenance.preset_sha256",
    )
    expected_preset_sha256 = _sha256(
        profile.get("preset_sha256"),
        "profile.preset_sha256",
    )
    if preset_sha256 != expected_preset_sha256:
        raise ParityContractError(
            f"{source} preset SHA-256 does not match the parity profile"
        )

    expected_cases = [
        _mapping(case, "parity request case")
        for case in _list(request.get("cases"), "parity request cases")
    ]
    expected_case_ids = [str(case["case_id"]) for case in expected_cases]
    case_digests = _list(
        provenance.get("case_digests"),
        f"{source} provenance.case_digests",
    )
    digests_by_case: dict[str, Mapping[str, Any]] = {}
    for item in case_digests:
        digest = _mapping(item, f"{source} provenance case digest")
        case_id = _string(
            digest.get("case_id"),
            f"{source} provenance case digest case_id",
        )
        if case_id in digests_by_case:
            raise ParityContractError(f"{source} contains duplicate case digests")
        digests_by_case[case_id] = digest
    if set(digests_by_case) != set(expected_case_ids):
        raise ParityContractError(f"{source} case digests do not match request cases")

    normalized_case_digests: list[JsonObject] = []
    for case in expected_cases:
        case_id = str(case["case_id"])
        digest = digests_by_case[case_id]
        normalized: JsonObject = {"case_id": case_id}
        for key in ("size_bytes", "geometry_size_bytes"):
            if key in case:
                if digest.get(key) != case.get(key):
                    raise ParityContractError(
                        f"{source} case digest field {key} does not match the request"
                    )
                normalized[key] = digest[key]
        for key in ("sha256", "geometry_sha256"):
            if key in case:
                actual = _sha256(
                    digest.get(key),
                    f"{source} case digest field {key}",
                )
                expected = _sha256(
                    case.get(key),
                    f"parity request case field {key}",
                )
                if actual != expected:
                    raise ParityContractError(
                        f"{source} case digest field {key} does not match the request"
                    )
                normalized[key] = actual
        normalized_case_digests.append(normalized)
    return {
        "provider": dict(provider),
        "preset_sha256": preset_sha256,
        "case_digests": normalized_case_digests,
    }


def validate_execution_result(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    source_label: str,
    run_id: object,
    workflow: object,
    status: object,
    payload: object,
) -> JsonObject:
    """Validate provenance and selections reported by one wrapper execution."""

    validate_profile(profile)
    validate_request(profile, request)
    source = _string(source_label, "execution result source label")
    result_run_id = _string(run_id, f"{source} result run_id")
    if status != "succeeded":
        raise ParityContractError(f"{source} execution did not succeed")
    if workflow != profile.get("workflow_id"):
        raise ParityContractError(
            f"{source} workflow does not match the parity profile"
        )

    result_payload = _mapping(payload, f"{source} result payload")
    expected_case_ids = [
        _mapping(case, "parity request case")["case_id"]
        for case in _list(request.get("cases"), "parity request cases")
    ]
    for key, expected in (
        ("model_names", request["models"]),
        ("case_ids", expected_case_ids),
        ("selected_metrics", request["metrics"]),
    ):
        if result_payload.get(key) != expected:
            raise ParityContractError(
                f"{source} result {key} does not match the submitted request"
            )
    provenance = _validate_provenance(
        profile,
        request,
        result_payload,
        source=f"{source} result",
    )
    return {
        "run_id": result_run_id,
        **provenance,
    }


def build_handoff(
    *,
    evidence_dir: str | Path,
    profile: Mapping[str, Any],
    parity_run_id: str,
    image: str,
    mount_target: str,
) -> JsonObject:
    validate_profile(profile)
    evidence = Path(evidence_dir)
    request = read_json_object(evidence / "request.json")
    results = read_json_object(evidence / "results.json")
    resolved_config = read_json_object(evidence / "resolved_config.json")

    execution = _mapping(results.get("execution"), "results.execution")
    validated_result = validate_execution_result(
        profile,
        request,
        source_label="REST",
        run_id=execution.get("run_id"),
        workflow=execution.get("workflow"),
        status=execution.get("status"),
        payload=results.get("payload"),
    )
    report_output = output_by_name(results, "benchmark_results.json")
    config_output = output_by_name(results, "resolved_config.json")
    report_path = evidence / "benchmark_results.json"
    if not report_path.is_file():
        raise ParityContractError(f"missing downloaded REST report: {report_path}")

    dataset = _mapping(
        _mapping(profile.get("config"), "profile.config").get("dataset"),
        "profile.config.dataset",
    )
    input_root = resolved_dataset_root(
        resolved_config, _string(dataset.get("name"), "profile dataset name")
    )
    handoff: JsonObject = {
        "schema_version": SCHEMA_VERSION,
        "parity_run_id": _string(parity_run_id, "parity run ID"),
        "profile_id": profile["profile_id"],
        "workflow_id": profile["workflow_id"],
        "domain": profile["domain"],
        "image": validate_image_digest_reference(image),
        "mount_target": _string(mount_target, "mount target"),
        "rest_run_id": validated_result["run_id"],
        "request": request,
        "provenance": {
            "provider": validated_result["provider"],
            "preset_sha256": validated_result["preset_sha256"],
            "case_digests": validated_result["case_digests"],
        },
        "rest": {
            "input_root_relpath": mount_relative_path(input_root, mount_target),
            "report_relpath": mount_relative_path(
                _string(report_output.get("storage_path"), "REST report storage path"),
                mount_target,
            ),
            "report_sha256": sha256_file(report_path),
            "report_size_bytes": report_path.stat().st_size,
            "resolved_config_relpath": mount_relative_path(
                _string(config_output.get("storage_path"), "REST config storage path"),
                mount_target,
            ),
        },
    }
    validate_handoff(profile, handoff)
    return handoff


def validate_handoff(profile: Mapping[str, Any], handoff: Mapping[str, Any]) -> None:
    validate_profile(profile)
    if handoff.get("schema_version") != SCHEMA_VERSION:
        raise ParityContractError(f"handoff schema_version must be {SCHEMA_VERSION}")
    for key in (
        "parity_run_id",
        "profile_id",
        "workflow_id",
        "domain",
        "mount_target",
        "rest_run_id",
    ):
        _string(handoff.get(key), f"handoff.{key}")
    validate_image_digest_reference(handoff.get("image"), "handoff.image")
    for key in ("profile_id", "workflow_id", "domain"):
        if handoff.get(key) != profile.get(key):
            raise ParityContractError(f"handoff {key} does not match profile")
    request = _mapping(handoff.get("request"), "handoff.request")
    validate_request(profile, request)
    _validate_provenance(
        profile,
        request,
        handoff.get("provenance"),
        source="handoff",
    )
    rest = _mapping(handoff.get("rest"), "handoff.rest")
    for key in (
        "input_root_relpath",
        "report_relpath",
        "report_sha256",
        "resolved_config_relpath",
    ):
        _string(rest.get(key), f"handoff.rest.{key}")
    digest = str(rest["report_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ParityContractError(
            "handoff.rest.report_sha256 must be lowercase SHA-256"
        )
    size = rest.get("report_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ParityContractError(
            "handoff.rest.report_size_bytes must be a positive integer"
        )


def build_direct_config(
    profile: Mapping[str, Any],
    handoff: Mapping[str, Any],
    *,
    input_root: str | Path,
    output_dir: str | Path,
) -> JsonObject:
    validate_handoff(profile, handoff)
    request = _mapping(handoff.get("request"), "handoff.request")
    return build_direct_config_for_request(
        profile,
        request,
        input_root=input_root,
        output_dir=output_dir,
    )


def build_direct_config_for_request(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    input_root: str | Path,
    output_dir: str | Path,
) -> JsonObject:
    validate_profile(profile)
    validate_request(profile, request)
    config = _mapping(profile.get("config"), "profile.config")
    run = copy.deepcopy(dict(_mapping(config.get("run"), "profile.config.run")))
    run.update(
        {
            "output_dir": str(output_dir),
            "seed": request["seed"],
            "save_inference_mesh": request["save_inference_mesh"],
        }
    )
    run["metrics_cache"] = {
        "enabled": False,
        "path": str(Path(output_dir) / "metrics_cache"),
    }

    dataset = copy.deepcopy(
        dict(_mapping(config.get("dataset"), "profile.config.dataset"))
    )
    dataset["root"] = str(input_root)
    dataset["case_ids"] = [
        _mapping(case, "handoff request case")["case_id"]
        for case in _list(request.get("cases"), "handoff request cases")
    ]
    return {
        "run": run,
        "benchmark": {
            "mode": "matrix",
            "models": copy.deepcopy(
                _list(config.get("models"), "profile.config.models")
            ),
            "datasets": [dataset],
            "reproducibility": copy.deepcopy(
                dict(
                    _mapping(
                        config.get("reproducibility"),
                        "profile.config.reproducibility",
                    )
                )
            ),
        },
        "output": copy.deepcopy(
            dict(_mapping(config.get("output"), "profile.config.output"))
        ),
        "metrics": list(request["metrics"]),
        "reports": {
            **copy.deepcopy(
                dict(_mapping(config.get("reports"), "profile.config.reports"))
            ),
            "visual_case_ids": list(request["visual_case_ids"]),
        },
    }


def verify_staged_inputs(
    profile: Mapping[str, Any],
    handoff: Mapping[str, Any],
    *,
    input_root: str | Path,
) -> list[JsonObject]:
    validate_handoff(profile, handoff)
    request = _mapping(handoff.get("request"), "handoff.request")
    return verify_staged_inputs_for_request(
        profile,
        request,
        input_root=input_root,
    )


def verify_staged_inputs_for_request(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    input_root: str | Path,
) -> list[JsonObject]:
    validate_profile(profile)
    validate_request(profile, request)
    root = Path(input_root).resolve(strict=True)
    if not root.is_dir():
        raise ParityContractError(f"input root is not a directory: {root}")
    pattern = re.compile(
        str(_mapping(profile["request"], "profile.request")["case_id_pattern"])
    )
    layout = _list(profile.get("input_layout"), "profile.input_layout")
    verified: list[JsonObject] = []
    for case_value in _list(request["cases"], "parity request cases"):
        case = _mapping(case_value, "parity request case")
        case_id = _string(case.get("case_id"), "parity request case_id")
        match = pattern.fullmatch(case_id)
        if match is None:
            raise ParityContractError(f"invalid case_id {case_id!r}")
        substitutions = {"case_id": case_id, **match.groupdict()}
        for index, layout_value in enumerate(layout):
            entry = _mapping(layout_value, f"profile.input_layout[{index}]")
            relative_text = str(entry["relative_path"]).format(**substitutions)
            relative = PurePosixPath(relative_text)
            if relative.is_absolute() or any(
                part in {"", ".", ".."} for part in relative.parts
            ):
                raise ParityContractError(
                    f"unsafe staged input relative path: {relative_text!r}"
                )
            path = (root / Path(*relative.parts)).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ParityContractError(
                    f"staged input escapes input root: {path}"
                ) from exc
            if not path.is_file():
                raise ParityContractError(f"staged input is not a file: {path}")
            expected_size = case.get(entry["size_field"])
            if (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or expected_size <= 0
            ):
                raise ParityContractError(
                    f"invalid expected size for {case_id}: {expected_size!r}"
                )
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ParityContractError(
                    f"size mismatch for {path}: {actual_size} != {expected_size}"
                )
            expected_digest = _sha256(
                case.get(entry["digest_field"]),
                f"expected SHA-256 for {case_id}",
            )
            actual_digest = sha256_file(path)
            if actual_digest != expected_digest:
                raise ParityContractError(
                    f"SHA-256 mismatch for {path}: {actual_digest} != {expected_digest}"
                )
            verified.append(
                {
                    "case_id": case_id,
                    "path": str(path),
                    "size_bytes": actual_size,
                    "sha256": actual_digest,
                }
            )
    return verified


def _flatten_metric(
    value: object, *, prefix: str, destination: dict[str, float]
) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise ParityContractError(f"metric {prefix!r} must not be empty")
        for key, child in sorted(value.items()):
            name = _string(key, f"metric component under {prefix}")
            _flatten_metric(
                child,
                prefix=f"{prefix}.{name}",
                destination=destination,
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParityContractError(f"metric {prefix!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ParityContractError(f"metric {prefix!r} must be finite")
    destination[prefix] = number


def normalize_report(report: object) -> tuple[list[JsonObject], dict[MetricKey, float]]:
    rows = _list(report, "benchmark report")
    structure: list[JsonObject] = []
    values: dict[MetricKey, float] = {}
    seen_rows: set[tuple[str, str]] = set()
    for row_index, row_value in enumerate(rows):
        row = _mapping(row_value, f"benchmark report[{row_index}]")
        model = _string(row.get("model"), f"benchmark report[{row_index}].model")
        dataset = _string(row.get("dataset"), f"benchmark report[{row_index}].dataset")
        row_key = (model, dataset)
        if row_key in seen_rows:
            raise ParityContractError(
                f"duplicate benchmark report row for {model}/{dataset}"
            )
        seen_rows.add(row_key)
        cases = _list(row.get("cases"), f"benchmark report[{row_index}].cases")
        case_ids = [_string(case, "benchmark report case") for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ParityContractError("benchmark report contains duplicate cases")
        skipped = row.get("skipped") is True
        if skipped:
            raise ParityContractError(
                f"benchmark report row {model}/{dataset} was skipped"
            )
        summary_metrics: dict[str, float] = {}
        _flatten_metric(
            _mapping(row.get("metrics"), "benchmark summary metrics"),
            prefix="",
            destination=summary_metrics,
        )
        for metric, number in summary_metrics.items():
            metric_name = metric.removeprefix(".")
            values[(model, dataset, "summary", "", metric_name)] = number

        per_case = _list(row.get("per_case"), "benchmark per_case")
        per_case_structure: list[JsonObject] = []
        seen_case_ids: set[str] = set()
        for case_value in per_case:
            case = _mapping(case_value, "benchmark per_case item")
            case_id = _string(case.get("case_id"), "benchmark per_case case_id")
            if case_id in seen_case_ids:
                raise ParityContractError(
                    f"duplicate per-case report for {model}/{dataset}/{case_id}"
                )
            seen_case_ids.add(case_id)
            metrics: dict[str, float] = {}
            _flatten_metric(
                _mapping(case.get("metrics"), "benchmark case metrics"),
                prefix="",
                destination=metrics,
            )
            for metric, number in metrics.items():
                metric_name = metric.removeprefix(".")
                values[(model, dataset, "case", case_id, metric_name)] = number
            per_case_structure.append(
                {
                    "case_id": case_id,
                    "metric_dtype": case.get("metric_dtype"),
                    "metric_keys": sorted(name.removeprefix(".") for name in metrics),
                }
            )
        structure.append(
            {
                "model": model,
                "dataset": dataset,
                "cases": case_ids,
                "skipped": skipped,
                "summary_metric_keys": sorted(
                    name.removeprefix(".") for name in summary_metrics
                ),
                "per_case": sorted(
                    per_case_structure, key=lambda item: str(item["case_id"])
                ),
            }
        )
        if set(case_ids) != seen_case_ids:
            raise ParityContractError(
                f"summary and per-case IDs differ for {model}/{dataset}"
            )
    structure.sort(key=lambda item: (str(item["model"]), str(item["dataset"])))
    return structure, values


def validate_report_coverage(
    profile: Mapping[str, Any],
    request: Mapping[str, Any],
    report: object,
) -> int:
    """Require the report to cover every configured model, case, and metric."""

    validate_profile(profile)
    validate_request(profile, request)
    structure, values = normalize_report(report)
    expected_models = list(request["models"])
    expected_case_ids = [
        _mapping(case, "parity request case")["case_id"]
        for case in _list(request.get("cases"), "parity request cases")
    ]
    config = _mapping(profile.get("config"), "profile.config")
    dataset = _mapping(config.get("dataset"), "profile.config.dataset")
    expected_dataset = _string(dataset.get("name"), "profile dataset name")
    metric_outputs = _mapping(
        profile.get("report_metric_outputs"),
        "profile.report_metric_outputs",
    )
    expected_metric_keys = sorted(
        {
            name
            for requested_metric in request["metrics"]
            for name in _list(
                metric_outputs[requested_metric],
                f"profile.report_metric_outputs.{requested_metric}",
            )
        }
    )
    rows_by_model = {str(row["model"]): row for row in structure}
    if set(rows_by_model) != set(expected_models):
        raise ParityContractError(
            "benchmark report models do not match the parity request"
        )
    for model in expected_models:
        row = rows_by_model[model]
        if row["dataset"] != expected_dataset:
            raise ParityContractError(
                f"benchmark report dataset for {model} does not match "
                f"{expected_dataset!r}"
            )
        if row["cases"] != expected_case_ids:
            raise ParityContractError(
                f"benchmark report cases for {model} do not match the parity request"
            )
        if row["summary_metric_keys"] != expected_metric_keys:
            raise ParityContractError(
                f"benchmark summary metrics for {model} do not match the profile"
            )
        per_case = {
            str(case["case_id"]): case
            for case in _list(row["per_case"], "normalized report per_case")
        }
        if set(per_case) != set(expected_case_ids):
            raise ParityContractError(
                f"benchmark per-case IDs for {model} do not match the parity request"
            )
        for case_id in expected_case_ids:
            if per_case[case_id]["metric_keys"] != expected_metric_keys:
                raise ParityContractError(
                    f"benchmark metrics for {model}/{case_id} do not match the profile"
                )

    expected_value_count = (
        len(expected_models) * (1 + len(expected_case_ids)) * len(expected_metric_keys)
    )
    if len(values) != expected_value_count:
        raise ParityContractError(
            "benchmark report scalar metric count does not match the profile: "
            f"{len(values)} != {expected_value_count}"
        )
    return expected_value_count


def compare_reports(
    *,
    rest_report: object,
    direct_report: object,
    comparison: Mapping[str, Any],
    rest_label: str = "rest",
    direct_label: str = "direct",
    metric_outputs: Mapping[str, Any] | None = None,
) -> JsonObject:
    for value, name in (
        (rest_label, "rest_label"),
        (direct_label, "direct_label"),
    ):
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None
        ):
            raise ParityContractError(
                f"{name} must be a lowercase JSON field identifier"
            )
    if rest_label == direct_label:
        raise ParityContractError("report comparison labels must be distinct")
    rest_display = "REST" if rest_label == "rest" else rest_label.replace("_", "-")
    direct_display = direct_label.replace("_", "-")

    rest_structure, rest_values = normalize_report(rest_report)
    direct_structure, direct_values = normalize_report(direct_report)
    errors: list[str] = []
    if rest_structure != direct_structure:
        errors.append(f"{rest_display} and {direct_display} report structures differ")

    rest_keys = set(rest_values)
    direct_keys = set(direct_values)
    if missing := sorted(rest_keys - direct_keys):
        errors.append(
            f"{direct_display} report is missing {len(missing)} metric value(s)"
        )
    if extra := sorted(direct_keys - rest_keys):
        errors.append(
            f"{direct_display} report has {len(extra)} unexpected metric value(s)"
        )

    default_rtol = _number(comparison.get("default_rtol"), "comparison.default_rtol")
    default_atol = _number(comparison.get("default_atol"), "comparison.default_atol")
    overrides = _mapping(comparison.get("metrics", {}), "comparison.metrics")
    model_overrides = _mapping(comparison.get("models", {}), "comparison.models")
    output_metric_owners: dict[str, str] = {}
    for requested_metric, output_names in _mapping(
        metric_outputs or {},
        "metric_outputs",
    ).items():
        requested_name = _string(requested_metric, "requested metric name")
        for output_name_value in _list(
            output_names,
            f"metric_outputs.{requested_name}",
        ):
            output_name = _string(
                output_name_value,
                f"metric_outputs.{requested_name} output",
            )
            prior = output_metric_owners.setdefault(output_name, requested_name)
            if prior != requested_name:
                raise ParityContractError(
                    f"report metric {output_name!r} belongs to multiple "
                    "requested metrics"
                )
    metric_results: list[JsonObject] = []
    for key in sorted(rest_keys & direct_keys):
        model, dataset, scope, case_id, metric = key
        root_metric = metric.split(".", 1)[0]
        requested_metric = output_metric_owners.get(root_metric, root_metric)
        metric_override = _mapping(
            overrides.get(requested_metric, {}),
            f"comparison metric {requested_metric}",
        )
        model_override = _mapping(
            model_overrides.get(model, {}), f"comparison model {model}"
        )
        model_metric_overrides = _mapping(
            model_override.get("metrics", {}),
            f"comparison model {model} metrics",
        )
        model_metric_override = _mapping(
            model_metric_overrides.get(
                root_metric,
                model_metric_overrides.get(requested_metric, {}),
            ),
            f"comparison model {model} metric {root_metric}",
        )
        rtol = _number(
            model_metric_override.get(
                "rtol",
                model_override.get(
                    "default_rtol",
                    metric_override.get("rtol", default_rtol),
                ),
            ),
            f"{model}/{root_metric}.rtol",
        )
        atol = _number(
            model_metric_override.get(
                "atol",
                model_override.get(
                    "default_atol",
                    metric_override.get("atol", default_atol),
                ),
            ),
            f"{model}/{root_metric}.atol",
        )
        rest_value = rest_values[key]
        direct_value = direct_values[key]
        raw_absolute_difference = abs(rest_value - direct_value)
        absolute_difference = (
            raw_absolute_difference if math.isfinite(raw_absolute_difference) else None
        )
        if rest_value == 0:
            relative_difference = 0.0 if direct_value == 0 else None
        else:
            raw_relative_difference = raw_absolute_difference / abs(rest_value)
            relative_difference = (
                raw_relative_difference
                if math.isfinite(raw_relative_difference)
                else None
            )
        matches = math.isclose(
            rest_value,
            direct_value,
            rel_tol=rtol,
            abs_tol=atol,
        )
        if not matches:
            errors.append(
                f"{model}/{dataset}/{scope}/{case_id or '-'} {metric} differs"
            )
        metric_results.append(
            {
                "model": model,
                "dataset": dataset,
                "scope": scope,
                "case_id": case_id or None,
                "metric": metric,
                rest_label: rest_value,
                direct_label: direct_value,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "rtol": rtol,
                "atol": atol,
                "matches": matches,
            }
        )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        f"{rest_label}_structure": rest_structure,
        f"{direct_label}_structure": direct_structure,
        "metrics": metric_results,
    }
