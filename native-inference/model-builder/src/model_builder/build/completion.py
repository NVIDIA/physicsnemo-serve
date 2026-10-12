"""Framework-free validation of completed build receipts and retained artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from . import inputs
from .tensors import tensor_contracts, tensor_names
from model_builder.export.aoti_options import validate_aoti_options
from model_builder.export.tensorrt_profiles import (
    plugin_names,
    profile_version,
    required_operators,
    requires_byte_identical,
    selection_metadata,
)


def _completion_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("completed build contains an invalid artifact path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("completed build artifact path escapes its output root")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise RuntimeError("completed build artifact path contains a symlink")
    if not candidate.is_file():
        raise RuntimeError(f"completed build artifact is missing: {relative}")
    return candidate


def _verify_completion_file(root: Path, record: dict) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError("completed build artifact record must be an object")
    path = _completion_file(root, record.get("path"))
    if (
        type(record.get("size_bytes")) is not int
        or record["size_bytes"] < 0
        or not isinstance(record.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise RuntimeError("completed build artifact has invalid integrity metadata")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if (
        path.stat().st_size != record["size_bytes"]
        or digest.hexdigest() != record["sha256"]
    ):
        raise RuntimeError(
            f"completed build artifact integrity mismatch: {record['path']}"
        )
    return path


def _completion_inventory(root: Path, records: list) -> set[str]:
    if not isinstance(records, list) or not records:
        raise RuntimeError("completed build artifact inventory is empty")
    paths = set()
    for record in records:
        _verify_completion_file(root, record)
        if record["path"] in paths:
            raise RuntimeError(
                "completed build artifact inventory contains duplicate paths"
            )
        paths.add(record["path"])
    return paths


def validate_container_completion(plan: dict, *, read_recipe) -> None:
    """Verify claimed output bytes and coverage, without rerunning inference."""
    output = plan["output"]
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError(
            "container exited successfully without a build output directory"
        )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError("container exited successfully but " + message)

    try:
        receipt = json.loads(_completion_file(output, "build.json").read_text())
        require(
            receipt.get("format_version") == 1 and receipt.get("status") == "complete",
            "the build receipt is not complete",
        )
        requested = receipt.get("requested_backends")
        require(
            isinstance(requested, list)
            and len(requested) == len(plan["backends"])
            and set(requested) == set(plan["backends"]),
            "the requested backend coverage differs",
        )
        require(receipt.get("device") == plan["device"], "the build device differs")
        require(
            type(receipt.get("case_count")) is int and receipt["case_count"] > 0,
            "the build has no verified cases",
        )
        variants = receipt.get("variants")
        require(
            isinstance(variants, dict) and set(variants) == set(plan["backends"]),
            "the variant coverage differs",
        )
        release_record = receipt["release"]
        require(
            release_record.get("path") == "model/model-release.json",
            "the model release path is invalid",
        )
        release = json.loads(
            _verify_completion_file(output, release_record).read_text()
        )
        require(
            release.get("format_version") == 1
            and release.get("model")
            == {"name": plan["recipe"]["name"], "version": plan["recipe"]["version"]},
            "the model release identity differs",
        )
        if plan["recipe"].get("format_version") == 2:
            identities = inputs.input_identities(plan.get("model_inputs"))
            require(
                isinstance(identities, dict)
                and receipt.get("model_inputs") == identities
                and release.get("model_inputs") == identities,
                "the model input identities differ from the requested build",
            )
        require(
            set(release["variants"]) == set(variants),
            "the model release coverage differs",
        )
        require(
            release["runtime"]["sha256"] == receipt["runtime"]["sha256"],
            "the runtime identities differ",
        )
        source_files = _completion_inventory(output, receipt["source"]["files"])
        if plan["recipe"].get("format_version") == 2:
            effective_path = "source/effective-recipe.json"
            require(
                effective_path in source_files,
                "the model input effective recipe is not inventoried",
            )
            effective_file = _completion_file(output, effective_path)
            retained = inputs.resolve_inputs(
                read_recipe(effective_file), effective_file
            )
            require(
                inputs.input_identities(retained) == identities,
                "the retained model input identities differ from the requested build",
            )
            for field in ("config", "checkpoint"):
                expected_path = "source/model-inputs/" + (
                    "config.json" if field == "config" else "checkpoint.pt"
                )
                require(
                    Path(retained[field]["path"]) == (output / expected_path).resolve(),
                    f"the retained model input {field} path is invalid",
                )
            for descriptor in (
                retained["config"],
                retained["checkpoint"],
                *retained["assets"].values(),
            ):
                relative = (
                    Path(descriptor["path"]).relative_to(output.resolve()).as_posix()
                )
                require(
                    relative in source_files,
                    "a retained model input is not inventoried",
                )
            canonical_path = "source/model-inputs/effective-config.json"
            require(
                canonical_path in source_files,
                "the effective model input configuration is not inventoried",
            )
            _verify_completion_file(
                output, {"path": canonical_path, **identities["effective_config"]}
            )
        target = plan["device"].split(":", 1)
        expected_device = {
            "type": target[0],
            "index": int(target[1]) if len(target) == 2 else 0,
        }
        for backend, variant in variants.items():
            byte_identical = (
                backend == "tensorrt"
                and requires_byte_identical(
                    plan["recipe"].get("tensorrt_profile", "baseline")
                )
            ) or (
                backend == "aoti"
                and plan["recipe"].get("aoti_profile") == "aten-boundary-exact-v3"
            )
            exact_label = "DoMINO"
            if backend == "tensorrt":
                exact_label = {
                    "geotransolver-exact": "GeoTransolver",
                    "geotransolver-exact-v2": "GeoTransolver",
                    "layout-order-exact-v2": "Transolver",
                }.get(plan["recipe"].get("tensorrt_profile"), "DoMINO")
            require(variant.get("status") == "complete", f"{backend} is not complete")
            require(
                variant.get("package_base") == "model"
                and variant.get("evidence_base") == "build",
                f"{backend} artifact roots are invalid",
            )
            model_root = output / "model"
            files = _completion_inventory(model_root, variant["files"])
            published = release["variants"][backend]
            require(
                published.get("package") == variant["package"]
                and published.get("files") == variant["files"],
                f"{backend} release inventory differs from the build",
            )
            manifest_relative = variant["package"] + "/model.json"
            require(
                manifest_relative in files,
                f"{backend} package manifest is not inventoried",
            )
            manifest = json.loads(
                _completion_file(model_root, manifest_relative).read_text()
            )
            if "inputs" in plan["recipe"] or "outputs" in plan["recipe"]:
                expected_inputs, expected_outputs = tensor_contracts(plan["recipe"])
                require(
                    manifest.get("inputs") == expected_inputs
                    and manifest.get("outputs") == expected_outputs,
                    f"{backend} package tensor contract differs from the requested recipe",
                )
            artifacts = manifest.get("artifacts")
            require(
                isinstance(artifacts, list) and bool(artifacts),
                f"{backend} has no compiled artifact",
            )
            for artifact in artifacts:
                require(
                    artifact.get("backend") == backend
                    and variant["package"] + "/" + artifact["path"] in files,
                    f"{backend} compiled artifact is not inventoried",
                )
                profile = plan["recipe"].get("aoti_profile", "baseline")
                if backend == "aoti" and profile != "baseline":
                    require(
                        artifact.get("correctness_profile", {}).get("name") == profile,
                        "AOTI artifact profile differs from the requested recipe",
                    )
                if backend == "aoti":
                    if profile == "aten-boundary-exact-v3":
                        correctness = artifact.get("correctness_profile", {})
                        require(
                            correctness.get("version") == 3
                            and all(
                                correctness.get(key, {}).get("shape_padding") is False
                                for key in (
                                    "requested_compiler_settings",
                                    "applied_compiler_settings",
                                )
                            ),
                            "DoMINO AOTI v3 requires shape padding disabled",
                        )
                        if "domino_exact_ops" in plan["recipe"].get("assets", {}):
                            # Framework-free constants: validation must work on
                            # the host even when only the worker has Torch.
                            operator = {
                                "id": "physicsnemo-cfd.domino-exact-boundary",
                                "abi": "b1c60ddada2438469a1d24b4e53ae196425b73648f6d8ae45ecf64043755d7e6",
                            }
                            boundaries = correctness.get("tensor_only_boundaries", {})
                            require(
                                artifact.get("required_operators") == [operator]
                                and boundaries.get("operator") == operator
                                and type(boundaries.get("rewritten_nodes")) is int
                                and boundaries["rewritten_nodes"] > 0,
                                "DoMINO AOTI tensor-only operator metadata is incomplete",
                            )
                    options = validate_aoti_options(
                        plan["recipe"].get("aoti_options", {}), profile
                    )
                    recorded = artifact.get("compiler_options")
                    if options or recorded is not None:
                        require(
                            isinstance(recorded, dict),
                            "AOTI compiler options metadata is missing",
                        )
                        for key in ("requested", "applied"):
                            require(
                                validate_aoti_options(recorded.get(key), profile)
                                == options,
                                f"AOTI compiler options {key} differ from the requested recipe",
                            )
                        if "effective" in recorded:
                            effective = recorded["effective"]
                            require(
                                isinstance(effective, dict)
                                and all(
                                    effective.get(key) is value
                                    for key, value in options.items()
                                ),
                                "AOTI effective compiler options differ from the requested recipe",
                            )
                profile = plan["recipe"].get("tensorrt_profile", "baseline")
                if backend == "tensorrt" and profile != "baseline":
                    require(
                        artifact.get("correctness_profile", {}).get("name") == profile,
                        "TensorRT artifact profile differs from the requested recipe",
                    )
                if backend == "tensorrt" and byte_identical:
                    correctness = artifact.get("correctness_profile", {})
                    names = plugin_names(profile)
                    counts = correctness.get("replacement_counts", {})
                    libraries = correctness.get("plugin_libraries", {})
                    require(
                        correctness.get("version") == profile_version(profile)
                        and correctness.get("plugins") == list(names)
                        and artifact.get("required_operators")
                        == required_operators(profile)
                        and isinstance(counts, dict)
                        and set(counts) == set(names)
                        and all(
                            type(value) is int and value > 0
                            for value in counts.values()
                        )
                        and isinstance(libraries, dict)
                        and set(libraries) == set(names)
                        and all(
                            correctness.get(key) == value
                            for key, value in selection_metadata(profile).items()
                        ),
                        f"{exact_label} exact operator metadata is incomplete",
                    )
                    for name in names:
                        record = libraries[name]
                        require(
                            isinstance(record, dict)
                            and isinstance(record.get("filename"), str)
                            and isinstance(record.get("sha256"), str)
                            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                            f"{exact_label} exact plugin identity is invalid",
                        )
                        if plan.get("model_inputs") is not None:
                            selected = (
                                plan["model_inputs"]
                                .get("assets", {})
                                .get(f"tensorrt_{name}_plugin", {})
                            )
                            require(
                                record["sha256"] == selected.get("sha256"),
                                f"{exact_label} exact plugin differs from the selected asset",
                            )
            graphs = _completion_inventory(output, variant["graphs"])
            require(
                "exported/" + backend + "/" + variant["graph"]["entrypoint"] in graphs,
                f"{backend} exported graph is not inventoried",
            )
            check = json.loads(
                _verify_completion_file(output, variant["checks"]).read_text()
            )
            require(
                check.get("format_version") == 1
                and check.get("passed") is True
                and check.get("backend") == backend,
                f"{backend} checks do not report success",
            )
            require(
                check["runtime"]["sha256"] == receipt["runtime"]["sha256"],
                f"{backend} check runtime identity differs",
            )
            if byte_identical:
                require(
                    check.get("require_byte_identical") is True
                    and check.get("limits") == {"max_abs": 0.0, "relative_l2": 0.0},
                    f"{exact_label} exact checks do not require byte-identical outputs",
                )
            cases = check.get("cases")
            require(
                isinstance(cases, list) and len(cases) == receipt["case_count"],
                f"{backend} check case coverage differs",
            )
            for case_index, case in enumerate(cases):
                metadata = case.get("metadata", {})
                require(
                    case.get("passed") is True
                    and metadata.get("schema_version") == 1
                    and metadata.get("completed") is True
                    and metadata.get("backend") == backend
                    and metadata.get("execution_device") == expected_device,
                    f"{backend} native case metadata does not confirm success",
                )
                require(
                    [value["name"] for value in case.get("outputs", [])]
                    == tensor_names(plan["recipe"], "outputs"),
                    f"{backend} native output coverage differs",
                )
                if "outputs" in plan["recipe"]:
                    require(
                        [
                            {key: value.get(key) for key in ("name", "dtype", "shape")}
                            for value in metadata.get("outputs", [])
                        ]
                        == expected_outputs,
                        f"{backend} native output tensor contract differs from the requested recipe",
                    )
                for output_index, metrics in enumerate(case["outputs"]):
                    directory = f"checks/{backend}/case-{case_index}/"
                    native_path = _completion_file(
                        output, directory + f"output-{output_index}.bin"
                    )
                    reference_path = _completion_file(
                        output, directory + f"reference-{output_index}.bin"
                    )
                    if byte_identical:
                        native = native_path.read_bytes()
                        reference = reference_path.read_bytes()
                        digest = hashlib.sha256(native).hexdigest()
                        require(
                            all(
                                type(metrics.get(key)) in (int, float)
                                and metrics[key] == 0.0
                                for key in ("max_abs", "relative_l2")
                            )
                            and native == reference
                            and metrics.get("actual_sha256") == digest
                            and metrics.get("reference_sha256") == digest,
                            f"{exact_label} native evidence is not byte-identical",
                        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise RuntimeError(f"container completed output is invalid: {error}") from error


def _verify_target_completion(plan):
    if not plan.get("required_gpu_arch"):
        return
    if plan["executor"] == "container":
        receipt = json.loads(
            _completion_file(plan["output"], "execution.json").read_text()
        )
        checked = receipt.get("target_check")
    else:
        checked = plan.get("target_check")
    if not isinstance(checked, dict) or any(
        checked.get(key) != expected
        for key, expected in {
            "device": plan["device"],
            "required_gpu_arch": plan["required_gpu_arch"],
            "actual_gpu_arch": plan["required_gpu_arch"],
        }.items()
    ):
        raise RuntimeError("Builder did not establish the required GPU architecture.")
    receipt = json.loads(_completion_file(plan["output"], "build.json").read_text())
    capability = receipt.get("environment", {}).get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(x) is not int for x in capability)
        or f"sm{capability[0]}{capability[1]}" != plan["required_gpu_arch"]
    ):
        raise RuntimeError(
            "Built model environment differs from the required GPU architecture."
        )
    plan["target_check"] = checked
