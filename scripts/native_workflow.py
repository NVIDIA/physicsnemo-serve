"""Native workflow authoring without importing a model or Python execute hook."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import re
import shutil
import socket
import struct
import subprocess
import sys

import yaml
from jsonschema.validators import validator_for

from plugin_runtime import load_plugin_manifest


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "native-inference/deploy"
BUILDER = "nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:5956694872b0141f14f18fff7783a0eecf219c732475c54f5cb3e5311a943ee8"
TENSORRT_BUILDER = "nvcr.io/dycvht5ows21/physicsnemo-serve@sha256:70c372c4eb3bc4b5468621ec0fe942514216229bf8bf49a96a32c6305dad8f00"
TARGET = "h100-sxm-80gb-linux-amd64-v1"
IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
CODECS = {"json-array-fp32", "raw-fp32-le"}


class UniqueSafeLoader(yaml.SafeLoader):
    """Release policy must never depend on silently discarded YAML fields."""


def _yaml_mapping(loader, node, deep=False):
    pairs = loader.construct_pairs(node, deep=deep)
    value = {}
    for name, item in pairs:
        require(name not in value, f"duplicate YAML field: {name}")
        value[name] = item
    return value


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    value = {}
    for name, item in pairs:
        require(name not in value, f"duplicate JSON field: {name}")
        value[name] = item
    return value


def read_json(path):
    def nonfinite(value):
        raise ValueError(f"non-finite JSON value in {path}: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            nonfinite(value)
        return result

    value = json.loads(
        Path(path).read_text(),
        object_pairs_hook=_pairs,
        parse_constant=nonfinite,
        parse_float=finite_float,
    )
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def contained(root, relative, label, *, directory=False):
    require(isinstance(relative, str) and relative, f"{label}: relative path required")
    path = Path(relative)
    require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
        f"{label} must stay inside the workflow/project: {relative}",
    )
    current = Path(root)
    for part in path.parts:
        current /= part
        require(
            not current.is_symlink(), f"{label}: symlink is not admitted: {relative}"
        )
    require(
        current.resolve().is_relative_to(Path(root).resolve()),
        f"{label} must stay inside {root}",
    )
    require(
        current.is_dir() if directory else current.is_file(),
        f"{label} missing: {current}",
    )
    return current.resolve()


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(data)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def inventory(root, *, exclude=()):
    root = Path(root)
    result = []
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        *exclude,
    }
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in ignored or part == ".env" or part.startswith(".env.")
            for part in relative.parts
        ):
            continue
        if path.name == "model-build.lock.json":
            continue
        require(not path.is_symlink(), f"snapshot symlink is not admitted: {path}")
        if path.is_dir():
            continue
        require(path.is_file(), f"snapshot requires a regular file: {path}")
        result.append({"path": relative.as_posix(), **identity(path)})
    return result


def _schema(root, container, key):
    inline = container.get(key + "_inline")
    if inline is None:
        inline = read_json(contained(root, container.get(key), key))
    require(isinstance(inline, dict), f"{key}: JSON schema object required")

    def check_references(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in ("$ref", "$dynamicRef", "$recursiveRef"):
                    require(
                        isinstance(item, str) and item.startswith("#"),
                        f"{key}: schema references must be local fragments",
                    )
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)

    check_references(inline)
    validator_for(inline).check_schema(inline)
    return inline


def _project(path, backend, target):
    sys.path.insert(0, str(ROOT / "native-inference/model-builder/src"))
    try:
        from model_builder.build.authoring_config import load
        from model_builder.export.tensorrt_profiles import (
            plugin_names,
            validate_profile_backends,
        )

        # Keep qualified AOTI project identities stable. TensorRT is explicitly
        # selected for projects that may also advertise an AOTI alternative.
        loaded = load(
            path, SimpleNamespace(backend=[backend]) if backend == "tensorrt" else None
        )
        if backend == "tensorrt":
            document = loaded["document"]
            selected_profile = document.get("profiles", {}).get(loaded["profile"], {})
            # The release selector must not expand the project's declared
            # backends, or change the captured project identity.
            loaded["effective"]["backends"] = selected_profile.get(
                "backends", document.get("backends", ["aoti"])
            )
        validate_profile_backends(
            loaded["effective"].get("tensorrt_profile", "baseline"), [backend]
        )
        if backend == "tensorrt":
            profile = loaded["effective"].get("tensorrt_profile", "baseline")
            required = {
                "tensorrt_" + name + "_plugin" for name in plugin_names(profile)
            }
            missing = required - set(loaded["effective"].get("assets", {}))
            require(
                not missing,
                f"{path}: {profile} requires model assets: {', '.join(sorted(missing))}",
            )
    finally:
        sys.path.pop(0)
    effective = copy.deepcopy(loaded["effective"])
    require(
        backend in effective["backends"],
        f"{path}: selected backend {backend} is not declared",
    )
    require(
        effective["device"] in ("cuda", "cuda:0"),
        f"{path}: native GPU release requires cuda device",
    )
    expected_arch = target["compilation"]["cuda_architecture"]
    require(
        effective.get("required_gpu_arch", expected_arch) == expected_arch,
        f"{path}: required_gpu_arch conflicts with target {expected_arch}",
    )
    for field in ("adapter", "checkpoint", "runtime", "toolchain_lock"):
        value = effective.get(field)
        if value is not None:
            relative = (
                Path(value).relative_to(path.parent).as_posix()
                if Path(value).is_relative_to(path.parent)
                else value
            )
            contained(path.parent, relative, f"model.{field}")
            effective[field] = relative
    if isinstance(effective.get("config"), str):
        value = Path(effective["config"])
        relative = (
            value.relative_to(path.parent).as_posix()
            if value.is_relative_to(path.parent)
            else str(value)
        )
        contained(path.parent, relative, "model.config")
        effective["config"] = relative
    for name, value in effective.get("assets", {}).items():
        value = Path(value)
        relative = (
            value.relative_to(path.parent).as_posix()
            if value.is_relative_to(path.parent)
            else str(value)
        )
        contained(path.parent, relative, f"model.assets.{name}")
        effective["assets"][name] = relative
    output = Path(effective.pop("output_root", "builds"))
    require(
        output.is_relative_to(path.parent),
        f"{path}: output_root must stay inside the model project",
    )
    excluded = output.relative_to(path.parent).parts
    require(excluded, f"{path}: output_root cannot replace the project directory")
    effective["output_root"] = output.relative_to(path.parent).as_posix()
    for source in effective.get("source", []):
        value = path.parent / source
        contained(path.parent, source, "model.source", directory=value.is_dir())
    return {
        "project": str(path),
        "project_document": effective,
        "project_files": inventory(path.parent, exclude=(excluded[0],)),
    }


def _mapping(mapping, projects, request_schema, result_schema):
    require(
        isinstance(mapping, dict), "configuration.native.tensor_adapter is required"
    )
    alias = mapping.get("model_alias", "primary")
    require(alias in projects, f"tensor_adapter.model_alias is not declared: {alias}")
    model = projects[alias]["project_document"]
    for direction, model_key, field_key, schema in [
        ("inputs", "input_names", "request_field", request_schema),
        ("outputs", "output_names", "response_field", result_schema),
    ]:
        fields = mapping.get(direction)
        require(
            isinstance(fields, dict) and fields,
            f"tensor_adapter.{direction} must map named tensors",
        )
        require(
            set(fields) == set(model.get(model_key, [])),
            f"tensor_adapter.{direction} names {sorted(fields)} disagree with model {model_key} {model.get(model_key, [])}",
        )
        for name, settings in fields.items():
            require(
                isinstance(settings, dict), f"tensor mapping object required: {name}"
            )
            require(
                settings.get("codec") in CODECS,
                f"unsupported tensor codec for {name}: {settings.get('codec')}",
            )
            if direction == "outputs" and settings["codec"] == "raw-fp32-le":
                require(
                    isinstance(name, str)
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,124}", name),
                    "raw output tensor name must be a safe identifier of at most 124 characters",
                )
            field = settings.get(field_key)
            require(
                isinstance(field, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", field),
                f"tensor {name}: HTTP field must be a safe identifier of at most 128 characters",
            )
            require(
                field in schema.get("properties", {}),
                f"tensor {name}: {field_key} {field!r} is missing from HTTP schema",
            )
            field_schema = schema["properties"][field]
            expected_type = (
                "array" if settings["codec"] == "json-array-fp32" else "object"
            )
            require(
                field_schema.get("type") == expected_type,
                f"tensor {name}: {settings['codec']} requires HTTP {expected_type} schema",
            )
        mapped_fields = [settings[field_key] for settings in fields.values()]
        require(
            len(mapped_fields) == len(set(mapped_fields)),
            f"tensor_adapter.{direction}: duplicate HTTP field mapping",
        )


def _self_tests(value, projects):
    require(
        set(value) == set(projects),
        "self-test fixtures must cover every fixed model alias",
    )
    for alias, case in value.items():
        require(isinstance(case, dict), f"self-test {alias}: object required")
        model = projects[alias]["project_document"]
        for field, names in (
            ("inputs", "input_names"),
            ("expected_outputs", "output_names"),
        ):
            tensors = case.get(field)
            require(
                isinstance(tensors, dict) and set(tensors) == set(model.get(names, [])),
                f"self-test {alias}.{field} must cover model {names}",
            )
            for name, values in tensors.items():
                require(
                    isinstance(values, list)
                    and values
                    and len(values) <= 1_048_576
                    and all(
                        type(x) in (int, float) and math.isfinite(x) for x in values
                    ),
                    f"self-test {alias}.{field}.{name}: bounded finite FP32 array required",
                )
        for tolerance in ("rtol", "atol"):
            number = case.get(tolerance)
            require(
                type(number) in (int, float) and math.isfinite(number) and number >= 0,
                f"self-test {alias}.{tolerance}: finite nonnegative tolerance required",
            )


def resolve_release(
    plugin_root,
    profile="release.yaml",
    *,
    workspace=None,
    require_output=False,
    model_only=False,
):
    root = Path(plugin_root).resolve()
    manifest = load_plugin_manifest(root / "plugin.yaml")
    require(
        manifest["runtime"]["kind"] == "cpp", "native release requires runtime.kind cpp"
    )
    path = contained(root, profile, "release_profile")
    release = yaml.load(path.read_text(), Loader=UniqueSafeLoader)
    require(
        isinstance(release, dict) and release.get("schema_version") == 1,
        "release.yaml requires schema_version 1",
    )
    workflow = manifest["metadata"]["id"]
    require(
        release.get("workflow") == workflow,
        "release workflow differs from plugin metadata.id",
    )
    adapter = manifest["runtime"]["entrypoint"]
    require(
        adapter in ("tensor.v1", "custom.v1", "transolver_surface"),
        f"unsupported linked adapter: {adapter}",
    )
    backend = release.get("backend")
    require(backend in ("aoti", "tensorrt"), "backend must be aoti or tensorrt")
    target_name = release.get("target_profile")
    require(
        isinstance(target_name, str) and IDENTIFIER.fullmatch(target_name),
        "target_profile must name a versioned target catalog entry",
    )
    target_path = DEPLOY / "targets" / (target_name + ".json")
    require(target_path.is_file(), f"unknown target_profile: {target_name}")
    catalog = read_json(target_path)
    require(catalog.get("schema_version") == 1, "unsupported target catalog schema")
    target = {
        "profile": catalog["id"],
        "compilation": catalog["compilation"],
        "deployment": catalog["deployment"],
    }
    require(
        target["compilation"]["precision"] == "fp32",
        "initial native adapter requires FP32",
    )
    images = release.get("images", {})
    require(
        isinstance(images, dict)
        and all(
            isinstance(images.get(name), str) and IMAGE.fullmatch(images[name])
            for name in ("builder", "server")
        ),
        "images.builder and images.server must be immutable registry/repository@sha256 digests",
    )
    schemas = {}
    for content_type in [] if model_only else manifest["ingress"]["content_types"]:
        require(
            content_type in ("application/json", "multipart/form-data"),
            f"unsupported native content type: {content_type}",
        )
        key = "form_schema" if content_type == "multipart/form-data" else "json_schema"
        schemas[key] = _schema(root, manifest["ingress"], key)
    request_schema = next(iter(schemas.values()), {})
    require(
        all(schema == request_schema for schema in schemas.values()),
        "native JSON and multipart ingress must use the same canonical parameters schema",
    )
    result_schema = (
        {} if model_only else _schema(root, manifest["outputs"], "result_schema")
    )
    deployment_manifest = copy.deepcopy(manifest)
    for key, schema in schemas.items():
        deployment_manifest["ingress"].pop(key, None)
        deployment_manifest["ingress"][key + "_inline"] = schema
    deployment_manifest["outputs"].pop("result_schema", None)
    deployment_manifest["outputs"]["result_schema_inline"] = result_schema
    executor = manifest["runtime"]["executor_class"]
    if adapter == "transolver_surface":
        executor = "native.transolver." + backend
        deployment_manifest["runtime"]["executor_class"] = executor
        for stage in deployment_manifest["pipeline"]["stages"]:
            if stage["phase"] == "execute":
                stage["queue"] = "execute." + executor
    runtime = {
        "executor_class": executor,
        "request_timeout_seconds": 600,
        "drain_timeout_seconds": 60,
        "startup_timeout_seconds": 300,
        "artifact_root": "/outputs/native-workflows",
        "input_cache": "/outputs/.cache/native-inputs",
        **release.get("runtime", {}),
    }
    for name in (
        "request_timeout_seconds",
        "drain_timeout_seconds",
        "startup_timeout_seconds",
    ):
        require(
            type(runtime[name]) is int and 0 < runtime[name] <= 7200,
            f"runtime.{name} must be an integer from 1 through 7200",
        )
    for name in ("artifact_root", "input_cache"):
        require(
            isinstance(runtime[name], str)
            and PurePosixPath(runtime[name]).is_absolute()
            and not runtime[name].startswith("//")
            and "\\" not in runtime[name],
            f"runtime.{name} must be an absolute Linux container path",
        )
    require(
        runtime["executor_class"] == executor,
        "runtime executor_class conflicts with the selected backend or manifest",
    )
    output = release.get("output", {})
    repository = output.get("repository")
    required = [] if repository else ["output.repository"]
    if require_output:
        require(
            repository,
            "output.repository must name your image registry/repository before release",
        )
    if repository:
        require(
            isinstance(repository, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9._:/-]*", repository)
            and ":" not in repository.rsplit("/", 1)[-1],
            "output.repository must be a registry/repository without a tag or digest",
        )
    assets = []
    for item in [] if model_only else release.get("assets", []):
        require(
            isinstance(item, dict),
            "release assets must contain source/destination objects",
        )
        source = contained(root, item.get("source"), "assets.source")
        destination = item.get("destination")
        require(
            isinstance(destination, str)
            and destination
            and not Path(destination).is_absolute()
            and ".." not in Path(destination).parts,
            "asset destination must stay inside the image asset directory",
        )
        require(
            Path(destination).parts[0] == "assets",
            "asset destination must be under assets/",
        )
        assets.append(
            {
                "name": item.get("name", Path(destination).stem),
                "source": str(source),
                "destination": destination,
                **identity(source),
            }
        )
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    resolved = {
        "schema_version": 1,
        "workflow": workflow,
        "manifest_version": manifest["metadata"]["version"],
        "adapter": adapter,
        "backend": backend,
        "target": target,
        "native_stack_contract": catalog["native_stack_contract"],
        "images": images,
        "plugin": {
            "source": str(root / "plugin.yaml"),
            "document": deployment_manifest,
        },
        "request_schema": request_schema,
        "result_schema": result_schema,
        "response_schema": result_schema,
        "runtime": runtime,
        "assets": assets,
        "input_files": {
            item["name"]: {
                "required": item["required"],
                "media_types": item["media_types"],
                "max_size_bytes": item["max_size_mb"] * 1024 * 1024,
            }
            for item in ([] if model_only else manifest["ingress"].get("files", []))
        },
        "output": {"repository": repository},
        "source_sha": source_sha,
        "configuration_required": required,
        "paths": {
            "workflow_root": str(root),
            "repository_root": str(ROOT),
            "workspace": str(
                Path(workspace).resolve() if workspace else root / ".native-release"
            ),
        },
    }
    lepton = release.get("lepton", {})
    allowed_location = {
        "workspace_id",
        "workspace_url",
        "node_group",
        "pull_secret",
        "resource_shape",
        "nfs_mount_base",
        "lustre_dir",
        "lustre_storage",
        "mount_target",
        "job_timeout",
        "poll_interval",
        "command_timeout",
    }
    require(
        isinstance(lepton, dict) and not set(lepton) - allowed_location,
        "release.lepton accepts execution location and bounded timeouts; use environment authentication",
    )
    resolved["lepton"] = lepton
    input_artifacts = {}
    for name, item in (
        {} if model_only else release.get("input_artifacts", {})
    ).items():
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name),
            "input_artifacts requires simple fixed names",
        )
        require(isinstance(item, dict), f"input_artifacts.{name}: descriptor required")
        if "url" in item:
            require(
                isinstance(item["url"], str) and item["url"].startswith("https://"),
                f"input_artifacts.{name}: HTTPS required",
            )
            require(
                isinstance(item.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]),
                f"input_artifacts.{name}: pinned SHA256 required",
            )
            require(
                type(item.get("size_bytes")) is int and item["size_bytes"] > 0,
                f"input_artifacts.{name}: positive byte size required",
            )
            input_artifacts[name] = {
                key: item[key] for key in ("url", "sha256", "size_bytes")
            }
        else:
            source = contained(
                root, item.get("source"), f"input_artifacts.{name}.source"
            )
            expected = identity(source)
            require(
                all(item.get(key, value) == value for key, value in expected.items()),
                f"input_artifacts.{name}: bytes differ from declared identity",
            )
            input_artifacts[name] = {"source": str(source), **expected}
    resolved["input_artifacts"] = input_artifacts
    has_models, has_project = "models" in release, "model_project" in release
    require(
        has_models != has_project,
        "release requires exactly one of model_project or models",
    )
    selections = (
        release["models"]
        if has_models
        else {"primary": {"project": release["model_project"]}}
    )
    require(
        isinstance(selections, dict)
        and selections
        and all(
            isinstance(x, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", x)
            for x in selections
        ),
        "models must be a nonempty fixed alias map",
    )
    if adapter == "transolver_surface":
        resolved.update(
            cfd_models=selections,
            provider_profile=release.get("provider_profile"),
            adapter_config=manifest.get("configuration", {}).get("native", {}),
        )
        return release_module().resolve_cfd_profile(resolved)
    projects = {}
    for alias, item in selections.items():
        require(isinstance(item, dict), f"model alias {alias}: project object required")
        project = contained(root, item.get("project"), "model_project")
        projects[alias] = _project(project, backend, target)
    if model_only:
        resolved.update(model_aliases=projects, model_only=True)
        return resolved
    native = manifest.get("configuration", {}).get("native", {})
    mapping = (
        native.get("tensor_adapter")
        if adapter == "tensor.v1"
        else native.get("custom", {})
    )
    if adapter == "tensor.v1":
        _mapping(mapping, projects, request_schema, result_schema)
    resolved.update(model_aliases=projects, adapter_config=mapping)
    if adapter == "custom.v1":
        selection = release.get("native_workflow", {})
        native_root = contained(
            root,
            selection.get("source_dir"),
            "native_workflow.source_dir",
            directory=True,
        )
        target_name = selection.get("cmake_target")
        require(
            isinstance(target_name, str) and IDENTIFIER.fullmatch(target_name),
            "native_workflow.cmake_target is required",
        )
        require(
            (native_root / "CMakeLists.txt").is_file(),
            "custom workflow requires native/CMakeLists.txt",
        )
        resolved["native_workflow"] = {
            "source_dir": str(native_root),
            "cmake_target": target_name,
            "files": inventory(native_root),
        }
    qualification = release.get("qualification", {})
    cases_path = contained(root, qualification.get("cases"), "qualification.cases")
    cases = read_json(cases_path)
    require(
        cases.get("schema_version") == 1
        and isinstance(cases.get("cases"), list)
        and cases["cases"],
        "qualification cases require schema_version 1 and nonempty cases",
    )
    names = [case.get("name") for case in cases["cases"]]
    require(
        all(isinstance(name, str) and IDENTIFIER.fullmatch(name) for name in names)
        and len(set(names)) == len(names),
        "qualification case names must be unique simple identifiers",
    )
    required_cases = qualification.get("required_cases")
    require(
        isinstance(required_cases, list)
        and required_cases
        and set(required_cases) == set(names)
        and len(required_cases) == len(names),
        "qualification.required_cases must match the complete nonempty case inventory",
    )
    case_file_inventory = {}
    for case in cases["cases"]:
        validator_for(request_schema)(request_schema).validate(case.get("request"))
        validator_for(result_schema)(result_schema).validate(case.get("expected"))
        files = case.get("files", {})
        require(
            isinstance(files, dict) and set(files) <= set(resolved["input_files"]),
            f"case {case['name']}: files must name declared ingress fields",
        )
        require(
            all(
                not rule["required"] or name in files
                for name, rule in resolved["input_files"].items()
            ),
            f"case {case['name']}: missing required fixture file",
        )
        for name, descriptor in files.items():
            rule = resolved["input_files"][name]
            require(
                name not in input_artifacts and isinstance(descriptor, dict),
                f"case {case['name']}: conflicting fixture input",
            )
            source = contained(
                root, descriptor.get("source"), f"case {case['name']}.files.{name}"
            )
            file_id = identity(source)
            require(
                descriptor.get("media_type") in rule["media_types"]
                and file_id["size_bytes"] <= rule["max_size_bytes"],
                f"case {case['name']}: fixture file violates ingress policy",
            )
            require(
                all(
                    descriptor.get(key, value) == value
                    for key, value in file_id.items()
                ),
                f"case {case['name']}: fixture file differs from declared identity",
            )
            relative_source = source.relative_to(root).as_posix()
            files[name] = {
                "source": relative_source,
                "media_type": descriptor["media_type"],
                **file_id,
            }
            case_file_inventory[relative_source] = {"path": relative_source, **file_id}
        goldens = case.get("expected_artifacts")
        if adapter == "tensor.v1":
            raw_outputs = {
                name + ".f32"
                for name, settings in mapping["outputs"].items()
                if settings["codec"] == "raw-fp32-le"
            }
            if raw_outputs:
                require(
                    isinstance(goldens, dict) and set(goldens) == raw_outputs,
                    f"case {case['name']}: expected_artifacts must cover every raw output",
                )
        if goldens is not None:
            require(
                isinstance(goldens, dict) and len(goldens) <= 64,
                "expected_artifacts must be a bounded identity map",
            )
            for name, item in goldens.items():
                require(
                    isinstance(name, str)
                    and len(name) <= 128
                    and IDENTIFIER.fullmatch(name)
                    and isinstance(item, dict)
                    and set(item) == {"sha256", "size_bytes"}
                    and isinstance(item["sha256"], str)
                    and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    and type(item["size_bytes"]) is int
                    and item["size_bytes"] >= 0,
                    f"expected_artifacts.{name} requires a SHA256 and nonnegative byte size",
                )
    tests_path = contained(
        root, qualification.get("self_test"), "qualification.self_test"
    )
    self_tests = read_json(tests_path)
    _self_tests(self_tests, projects)
    resolved["self_tests"] = self_tests
    resolved["qualification"] = {
        "policy": qualification.get("policy"),
        "required_cases": required_cases,
        "cases_path": str(cases_path),
        "cases_document": cases,
        "fixtures": [
            {"path": str(p.relative_to(root)), **identity(p)}
            for p in (cases_path, tests_path)
        ]
        + list(case_file_inventory.values()),
    }
    startup = release.get("startup", {})
    require(isinstance(startup, dict), "startup.workflow_cases requires an object")
    selected = startup.get("workflow_cases", [])
    require(
        isinstance(startup, dict)
        and set(startup) <= {"workflow_cases"}
        and isinstance(selected, list)
        and len(selected) <= 16
        and all(isinstance(name, str) for name in selected)
        and len(set(selected)) == len(selected)
        and set(selected) <= set(names),
        "startup.workflow_cases must select unique existing qualification case names",
    )
    resolved["startup_cases"] = []
    for name in selected:
        case = next(case for case in cases["cases"] if case["name"] == name)

        def has_assertion(value):
            return (
                any(has_assertion(item) for item in value.values())
                if isinstance(value, dict)
                else True
            )

        require(
            has_assertion(case["expected"]) or case.get("expected_artifacts"),
            f"startup case {name} requires an expected payload/artifact assertion",
        )
        fixture = {
            key: copy.deepcopy(value)
            for key, value in case.items()
            if key
            in {"name", "request", "expected", "expected_artifacts", "rtol", "atol"}
        }
        if case.get("files"):
            fixture["input_artifacts"] = {
                field: {
                    "path": str(root / descriptor["source"]),
                    **{
                        key: descriptor[key]
                        for key in ("sha256", "size_bytes", "media_type")
                    },
                }
                for field, descriptor in case["files"].items()
            }
        resolved["startup_cases"].append(fixture)
    contract = {
        "workflow": workflow,
        "manifest_version": resolved["manifest_version"],
        "request_schema": request_schema,
        "result_schema": result_schema,
        "input_files": resolved["input_files"],
        "content_types": manifest["ingress"]["content_types"],
        "default_content_type": manifest["ingress"]["default_content_type"],
        "operations": manifest["ingress"]["operations"],
    }
    resolved["public_contract_digest"] = hashlib.sha256(
        (
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()
    require(
        isinstance(resolved["qualification"]["policy"], str)
        and resolved["qualification"]["policy"],
        "qualification.policy is required",
    )
    return resolved


def validate_native(plugin_root, profile="release.yaml"):
    resolved = resolve_release(plugin_root, profile)
    return {
        "adapter": resolved["adapter"],
        "target_profile": resolved["target"]["profile"],
        "model_aliases": list(
            resolved.get("model_aliases", resolved.get("cfd_models", {}))
        ),
        "configuration_required": resolved["configuration_required"],
        "execution": "not_run",
        "gpu_qualification": "not_run",
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def init_native(
    plugin_root,
    *,
    template="tensor",
    backend="aoti",
    stack_image=None,
    force=False,
    content_type="application/json",
):
    root = Path(plugin_root)
    require(template in ("tensor", "cpp"), "native template must be tensor or cpp")
    require(backend in ("aoti", "tensorrt"), "backend must be aoti or tensorrt")
    require(
        content_type in ("application/json", "multipart/form-data"),
        "unsupported native content type",
    )
    require(
        content_type != "multipart/form-data" or template == "tensor",
        "multipart native scaffold currently supports the tensor template; author custom ingress/source explicitly",
    )
    image = stack_image or (BUILDER if backend == "aoti" else TENSORRT_BUILDER)
    require(
        isinstance(image, str) and IMAGE.fullmatch(image),
        "select a qualified immutable stack with --native-image registry/repository@sha256:digest",
    )
    require(
        IDENTIFIER.fullmatch(root.name),
        "plugin directory must have a simple workflow identifier",
    )
    if root.exists():
        require(force, f"Plugin directory already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    workflow = re.sub(r"[^a-z0-9-]+", "-", root.name.lower()).strip("-")
    adapter = "tensor.v1" if template == "tensor" else "custom.v1"
    executor = "native." + workflow
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["values"],
        "properties": {
            "values": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            }
        },
    }
    mapping = {
        "model_alias": "primary",
        "inputs": {"input": {"request_field": "values", "codec": "json-array-fp32"}},
        "outputs": {"output": {"response_field": "values", "codec": "json-array-fp32"}},
    }
    request_schema = schema
    raw_descriptor = {
        "artifact": "features",
        "dtype": "float32",
        "shape": [4],
        "byte_order": "little",
        "layout": "contiguous",
        "size_bytes": 16,
    }
    if content_type == "multipart/form-data":
        mapping["inputs"]["input"]["codec"] = "raw-fp32-le"
        request_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["values"],
            "properties": {
                "values": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(raw_descriptor),
                    "properties": {
                        key: {"const": value} for key, value in raw_descriptor.items()
                    },
                }
            },
        }
    manifest = {
        "metadata": {
            "id": workflow,
            "display_name": workflow.replace("-", " ").title(),
            "version": "0.1.0",
            "description": "Native static FP32 affine example; replace the model and acceptance fixtures.",
            "tags": ["native"],
        },
        "ingress": {
            "content_type": content_type,
            "operation": {"default": "run", "allowed": ["run"]},
            (
                "form_schema"
                if content_type == "multipart/form-data"
                else "json_schema"
            ): "schemas/request.json",
        },
        "pipeline": {
            "stages": [
                {
                    "id": "schedule",
                    "phase": "schedule",
                    "handler": "schedule",
                    "queue": "schedule",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "handler": "native_workflow",
                    "queue": "execute." + executor,
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "handler": "persist_results",
                    "queue": "results",
                    "next": None,
                },
            ]
        },
        "runtime": {"kind": "cpp", "entrypoint": adapter, "executor_class": executor},
        "resources": {"defaults": {"gpus_required": 1, "memory_mb": 2048}},
        "configuration": {
            "native": {"tensor_adapter": mapping}
            if template == "tensor"
            else {"custom": {}}
        },
        "outputs": {
            "result_schema": "schemas/result.json",
            "primary_artifact": {
                "name": "native_result",
                "media_type": "application/json",
            },
            "retention_hours": 24,
        },
        "developer": {"readiness": {"recommended_check_phase": "execute"}},
    }
    if content_type == "multipart/form-data":
        manifest["ingress"]["files"] = [
            {
                "name": "features",
                "required": True,
                "media_types": ["application/octet-stream"],
                "max_size_mb": 1,
            }
        ]
    profile = {
        "schema_version": 1,
        "workflow": workflow,
        "model_project": "model/model-build.json",
        "backend": backend,
        "target_profile": TARGET,
        "images": {"builder": image, "server": image},
        "qualification": {
            "cases": "fixtures/cases.json",
            "self_test": "fixtures/self-test.json",
            "required_cases": ["default", "alternate"],
            "policy": "tensor-fp32-v1",
        },
        "startup": {"workflow_cases": ["default"]},
        "output": {"repository": None},
    }
    (root / "plugin.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    _write_json(root / "schemas/request.json", request_schema)
    _write_json(root / "schemas/result.json", {**schema, "additionalProperties": True})
    if content_type == "application/json":
        _write_json(root / "examples/default_request.json", {"values": [0, 1, -1, 4]})
    _write_json(
        root / "fixtures/cases.json",
        {
            "schema_version": 1,
            "cases": [
                {
                    "name": "default",
                    "request": {"values": [0, 1, -1, 4]},
                    "expected": {"values": [1, 3, -1, 9]},
                },
                {
                    "name": "alternate",
                    "request": {"values": [2.5, -8, 0.125, 13]},
                    "expected": {"values": [6, -15, 1.25, 27]},
                },
            ],
        },
    )
    _write_json(
        root / "fixtures/self-test.json",
        {
            "primary": {
                "inputs": {"input": [0, 1, -1, 4]},
                "expected_outputs": {"output": [1, 3, -1, 9]},
                "rtol": 1e-5,
                "atol": 1e-6,
            }
        },
    )
    if content_type == "multipart/form-data":
        cases = read_json(root / "fixtures/cases.json")
        for case in cases["cases"]:
            source = f"fixtures/{case['name']}-input.f32"
            (root / source).write_bytes(struct.pack("<4f", *case["request"]["values"]))
            case["request"] = {"values": copy.deepcopy(raw_descriptor)}
            case["files"] = {
                "features": {"source": source, "media_type": "application/octet-stream"}
            }
        _write_json(root / "fixtures/cases.json", cases)
        _write_json(
            root / "examples/default_request.multipart.json",
            {
                "form_fields": {"values": raw_descriptor},
                "files": {"features": "fixtures/default-input.f32"},
            },
        )
    _write_json(
        root / "model/model-build.json",
        {
            "format_version": 2,
            "name": workflow,
            "version": "0.1.0",
            "adapter": "adapter.py",
            "checkpoint": "checkpoint.pt",
            "source": [],
            "config": {},
            "input_names": ["input"],
            "output_names": ["output"],
            "backends": [backend],
            **({"tensorrt_profile": "baseline"} if backend == "tensorrt" else {}),
            "executor": "local",
            "device": "cuda",
            "required_gpu_arch": "sm90",
            "output_root": "builds",
        },
    )
    (root / "model/adapter.py").write_text(AFFINE_MODEL)
    # This known parameterless example has a genuine, empty state_dict. Real
    # models still require their own checkpoint; Builder's contract is unchanged.
    shutil.copyfile(
        ROOT / "tests/fixtures/native_affine_empty_state.pt",
        root / "model/checkpoint.pt",
    )
    (root / ".gitignore").write_text(".native-release/\nmodel/builds/\n__pycache__/\n")
    if template == "cpp":
        (root / "native").mkdir()
        (root / "native/workflow.cpp").write_text(CUSTOM_WORKFLOW)
        (root / "native/CMakeLists.txt").write_text(CUSTOM_CMAKE)
        profile["native_workflow"] = {
            "source_dir": "native",
            "cmake_target": "my_native_workflow",
        }
    (root / "release.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))
    (root / "README.md").write_text(
        f"# {workflow}\n\nThis example computes `values * 2 + 1` through the native SDK.\nReplace model/adapter.py, schemas, and both acceptance/self-test fixtures with\nyour model's contract. Model code runs only during export/build.\n\nSet output.repository in release.yaml to your registry repository. The pinned\nH100 target and builder identify the qualified numerical stack; other targets\nneed their own qualification.\n\n```sh\npython scripts/plugin_dev.py validate {root}\npython scripts/plugin_dev.py release {root} --release-profile release.yaml --plan\npython scripts/plugin_dev.py release {root} --release-profile release.yaml\n```\n\nThe native path executes C++; no Python workflow.py is required.\n"
    )
    return {
        "status": "scaffolded",
        "workflow_id": workflow,
        "plugin_root": str(root),
        "runtime_profile": "cpp",
        "template": template,
        "backend": backend,
        "content_type": content_type,
        "configuration_required": ["output.repository"],
    }


AFFINE_MODEL = '''"""Build-time model adapter; never imported by native request execution."""
import torch


class Affine(torch.nn.Module):
    def forward(self, value):
        return value * 2.0 + 1.0


def create_model(config, assets):
    return Affine().eval()


def create_cases(config, assets):
    return [(torch.tensor(values, dtype=torch.float32),) for values in
            ([0, 1, -1, 4], [2.5, -8, 0.125, 13], [-0.5, 100, -32, 0.25])]
'''

CUSTOM_CMAKE = """cmake_minimum_required(VERSION 3.24)
project(MyNativeWorkflow LANGUAGES CXX)
add_library(my_native_workflow STATIC workflow.cpp)
target_link_libraries(my_native_workflow PRIVATE PhysicsNeMoWorkflow::authoring nlohmann_json::nlohmann_json)
target_compile_features(my_native_workflow PRIVATE cxx_std_17)
set_target_properties(my_native_workflow PROPERTIES POSITION_INDEPENDENT_CODE ON)
"""

CUSTOM_WORKFLOW = """#include <physicsnemo/workflow/v1/authoring.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace physicsnemo::workflow::v1 {
class ExampleWorkflow final : public Workflow {
 public:
  Outcome run(const Request& request, Context& context) override {
    auto values = nlohmann::json::parse(request.parameters_json).at("values").get<std::vector<float>>();
    if (values.size() != 4) throw std::invalid_argument("values must contain four floats");
    context.check_cancelled();
    physicsnemo::inference::TensorView input{
        "input", physicsnemo::inference::DType::kFloat32,
        {physicsnemo::inference::DeviceType::kCpu, 0}, {4}, values.data(),
        values.size() * sizeof(float)};
    auto result = context.infer("primary", {input});
    const auto output = result.output("output");
    const auto* data = static_cast<const float*>(output.data);
    std::vector<float> predictions(data, data + output.byte_size / sizeof(float));
    return Outcome{nlohmann::json{{"values", predictions}}.dump(), "{}"};
  }
};
std::unique_ptr<Workflow> create_custom_workflow() {
  return std::make_unique<ExampleWorkflow>();
}
}  // namespace physicsnemo::workflow::v1
"""


def release_module():
    if str(DEPLOY) not in sys.path:
        sys.path.insert(0, str(DEPLOY))
    return importlib.import_module("native_release")


def environment_report(resolved):
    """Probe orchestration prerequisites; this never initializes CUDA."""
    checks = []
    try:
        release_module()
        import native_release_lepton

        native_release_lepton.validate_controller_config(resolved)
        detail, configured = "Lepton execution location configured", True
    except ValueError as error:
        detail, configured = str(error), False
    checks.append(
        {
            "name": "lepton_execution_location",
            "ok": configured,
            "required": True,
            "detail": detail,
        }
    )
    for name in ("docker", "lep"):
        present = shutil.which(name) is not None
        checks.append(
            {
                "name": name,
                "ok": present,
                "required": True,
                "detail": f"{name} available"
                if present
                else (
                    "Install Docker on the Linux build/control host"
                    if name == "docker"
                    else "Install the Lepton CLI: uv tool install leptonai"
                ),
            }
        )
    if next(item for item in checks if item["name"] == "docker")["ok"]:
        try:
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            okay = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            okay = False
        checks.append(
            {
                "name": "docker_daemon",
                "ok": okay,
                "required": True,
                "detail": "Docker daemon reachable"
                if okay
                else "Start Docker or run on the configured Linux build/control host",
            }
        )
    if next(item for item in checks if item["name"] == "lep")["ok"]:
        try:
            probe = subprocess.run(
                ["lep", "workspace", "status"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            okay = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            okay = False
        checks.append(
            {
                "name": "lepton_workspace",
                "ok": okay,
                "required": True,
                "detail": "Lepton workspace accessible"
                if okay
                else "Authenticate the Lepton CLI and select the deployment workspace",
            }
        )
    return {
        "workflow_id": resolved["workflow"],
        "status": "ready" if all(item["ok"] for item in checks) else "not_ready",
        "runner": "lepton",
        "target_profile": resolved["target"]["profile"],
        "checks": checks,
        "configuration_required": resolved["configuration_required"],
        "gpu_qualification": "not_run",
        "_exit_code": 0 if all(item["ok"] for item in checks) else 1,
    }


def run_native_service(resolved, *, image, workspace, port=0, dry_run=False):
    require(
        isinstance(image, str) and IMAGE.fullmatch(image),
        "run-local requires --image registry/repository@sha256:<qualified image digest>",
    )
    require(
        type(port) is int and 0 <= port <= 65535, "port must be between 0 and 65535"
    )
    if port == 0:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    workspace = Path(workspace).resolve()
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--platform",
        "linux/amd64",
        "--gpus",
        "device=0",
        "--publish",
        f"127.0.0.1:{port}:8080",
        "--mount",
        f"type=bind,source={artifacts},target=/outputs",
        "--env",
        f"RELEASE_IMAGE_DIGEST={image}",
        "--env",
        f"PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID={resolved['workflow']}",
        "--env",
        "REDIS_STREAM_PREFIX=physicsnemo:",
        "--env",
        "WORKERS=all",
        image,
    ]
    plan = {
        "status": "planned",
        "workflow_id": resolved["workflow"],
        "image": image,
        "command": command,
        "endpoint": f"http://127.0.0.1:{port}/v1/infer/{resolved['workflow']}",
        "workspace": str(workspace),
        "execution": "not_run",
        "gpu_qualification": "not_run",
    }
    _write_json(workspace / "service-plan.json", plan)
    if dry_run:
        return plan
    require(
        shutil.which("docker"),
        "Docker is required; run this session inside the qualified H100 environment",
    )
    try:
        result = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        return {
            **plan,
            "status": "stopped",
            "execution": "native_service_session",
            "_exit_code": 130,
        }
    return {
        **plan,
        "status": "stopped" if result.returncode == 0 else "failed",
        "execution": "native_service_session",
        "_exit_code": result.returncode,
    }


def run_command(args, plugin_root):
    profile = getattr(args, "release_profile", None) or "release.yaml"
    model_only = getattr(args, "model_only", False)
    resolved = resolve_release(
        plugin_root,
        profile,
        workspace=getattr(args, "workspace", None),
        require_output=args.command in ("release", "check") and not model_only,
        model_only=model_only,
    )
    workspace = Path(resolved["paths"]["workspace"])
    cache = Path(
        getattr(args, "cache_dir", None)
        or Path.home() / ".cache/physicsnemo-serve/native"
    )
    if args.command == "check-env":
        return environment_report(resolved)
    if args.command == "run-local":
        return run_native_service(
            resolved,
            image=args.image,
            workspace=workspace,
            port=args.port,
            dry_run=args.dry_run,
        )
    module = release_module()
    if getattr(args, "plan", False):
        return module.plan_release(resolved, cache)
    if model_only:
        require(args.command == "check", "--model-only requires check")
        return module.check_models(resolved, cache_dir=cache, workspace=workspace)
    # The same exact-image gate is used for check and release. Check publishes
    # no promotion tag; neither command calls Python workflow hooks.
    return module.release(
        resolved,
        cache_dir=cache,
        workspace=workspace,
        promote=getattr(args, "promote", None),
    )
