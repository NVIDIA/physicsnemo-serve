"""Relocatable native application bytes shared by direct execution and HTTP."""

from __future__ import annotations

import math
import copy
import mimetypes
from pathlib import Path
import re
import shutil

from .common import (
    contained,
    digest,
    file_identity,
    inventory,
    legacy_module,
    read_json,
    require,
    verify_inventory,
    write_json,
)


def _array_schema(tensor):
    require(
        tensor.get("dtype") == "float32"
        and all(type(n) is int and n > 0 for n in tensor["shape"]),
        "application tensor adapter requires static FP32 tensors",
    )
    size = math.prod(tensor["shape"])
    return {
        "type": "array",
        "minItems": size,
        "maxItems": size,
        "items": {"type": "number"},
    }


def compare_artifacts(executed, case):
    expected = case.get("expected_artifacts")
    if expected is None:
        return
    actual = executed["result"].get("artifacts", [])
    require(
        isinstance(expected, dict) and isinstance(actual, list),
        "fixture artifact golden identities must be an object",
    )
    require(
        all(
            isinstance(item, dict) and isinstance(item.get("name"), str)
            for item in actual
        ),
        "invalid output artifact descriptor",
    )
    by_name = {item["name"]: item for item in actual}
    require(
        len(by_name) == len(actual) and set(by_name) == set(expected),
        "fixture artifact names differ from golden outputs",
    )
    for name, identity in expected.items():
        require(
            isinstance(identity, dict) and set(identity) == {"sha256", "size_bytes"},
            "fixture artifact requires a complete golden identity",
        )
        path = contained(executed["output_dir"], by_name[name]["path"])
        require(
            path.is_file()
            and not path.is_symlink()
            and file_identity(path) == identity,
            f"fixture artifact {name} identity differs from golden output",
        )


def _runtime_deadlines(settings, *, request_timeout=300):
    require(isinstance(settings, dict), "runtime deadlines must be a mapping")
    deadlines = {
        "request_timeout_seconds": request_timeout,
        "drain_timeout_seconds": 60,
        "startup_timeout_seconds": 300,
    }
    for name, default in deadlines.items():
        value = settings.get(name, default)
        require(
            type(value) is int and 0 < value <= 7200,
            f"runtime.{name} must be an integer from 1 through 7200",
        )
        deadlines[name] = value
    return deadlines


def _contract(project, manifests):
    settings = project.get("workflow", {})
    mapping = copy.deepcopy(settings.get("tensor_adapter"))
    alias = (mapping or {}).get(
        "model_alias",
        settings.get(
            "model_alias",
            "primary" if "primary" in manifests else next(iter(manifests)),
        ),
    )
    require(alias in manifests, "workflow.model_alias is not a fixed model alias")
    manifest = manifests[alias]
    mapping = mapping or {
        "model_alias": alias,
        "inputs": {
            item["name"]: {
                "request_field": "values"
                if len(manifest["inputs"]) == 1
                else item["name"],
                "codec": "json-array-fp32",
            }
            for item in manifest["inputs"]
        },
        "outputs": {
            item["name"]: {
                "response_field": "values"
                if len(manifest["outputs"]) == 1
                else item["name"],
                "codec": "json-array-fp32",
            }
            for item in manifest["outputs"]
        },
    }
    mapping.setdefault("model_alias", alias)
    schemas = {}
    for tensors, field, category in (
        (manifest["inputs"], "request_field", "inputs"),
        (manifest["outputs"], "response_field", "outputs"),
    ):
        require(
            set(mapping[category]) == {item["name"] for item in tensors},
            f"tensor_adapter.{category} must map every model tensor",
        )
        properties = {}
        for item in tensors:
            rule = mapping[category][item["name"]]
            require(
                rule.get("codec") in ("json-array-fp32", "raw-fp32-le"),
                "unsupported tensor codec",
            )
            properties[rule[field]] = (
                _array_schema(item)
                if rule["codec"] == "json-array-fp32"
                else {"type": "object"}
            )
        schemas[category] = {
            "type": "object",
            "required": list(properties),
            "properties": properties,
            "additionalProperties": category == "outputs",
        }
    for name, key in (("request", "inputs"), ("result", "outputs")):
        explicit = project.get("schemas", {}).get(name)
        if explicit:
            schemas[key] = (
                copy.deepcopy(explicit)
                if isinstance(explicit, dict)
                else read_json(Path(project["root"]) / explicit)
            )
    return mapping, schemas["inputs"], schemas["outputs"]


def _ingress_contract(project, request_schema, mapping, repository):
    authored = project.get("ingress")
    ingress = (
        copy.deepcopy(authored)
        if authored is not None
        else {
            "content_type": "application/json",
            "operation": {"default": "run", "allowed": ["run"]},
            "json_schema": "request.schema.json",
        }
    )
    normalized = copy.deepcopy(ingress)
    legacy_module(repository, "plugin_runtime")._normalize_ingress_aliases(normalized)
    types = normalized["content_types"]
    require(
        types
        and set(types) <= {"application/json", "multipart/form-data"}
        and normalized["default_content_type"] in types,
        "native ingress requires application/json or multipart/form-data",
    )
    schema_keys = [
        "form_schema" if kind == "multipart/form-data" else "json_schema"
        for kind in types
    ]
    schemas = []
    if authored is not None:
        validate_schema = legacy_module(repository, "native_workflow")._schema
        for key in schema_keys:
            if key in normalized or key + "_inline" in normalized:
                schemas.append(validate_schema(Path(project["root"]), normalized, key))
        require(
            not schemas
            or all(item == schemas[0] for item in schemas)
            and (
                not project.get("schemas", {}).get("request")
                or schemas[0] == request_schema
            ),
            "native JSON and multipart ingress must share the same request schema",
        )
        if schemas:
            request_schema = schemas[0]
        ingress = normalized
        for key in ("request_schema", "json_schema", "form_schema"):
            ingress.pop(key, None)
            ingress.pop(key + "_inline", None)
        for key in schema_keys:
            ingress[key] = "request.schema.json"
    files = ingress.get("files", [])
    require(
        isinstance(files, list) and len(files) <= 64,
        "ingress.files must be a bounded list",
    )
    rules = {}
    for item in files:
        require(isinstance(item, dict), "ingress file must be an object")
        name, size, media = (
            item.get("name"),
            item.get("max_size_mb"),
            item.get("media_types"),
        )
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            and name not in rules
            and type(item.get("required")) is bool
            and type(size) is int
            and 0 < size < 2**40
            and isinstance(media, list)
            and media
            and all(
                isinstance(value, str)
                and value
                and "\r" not in value
                and "\n" not in value
                for value in media
            ),
            "ingress files require unique names, required, media_types and positive max_size_mb",
        )
        rules[name] = {
            "required": item["required"],
            "media_types": media,
            "max_size_bytes": size * 1024 * 1024,
        }
    require(
        not rules or "multipart/form-data" in types,
        "request files require multipart/form-data ingress",
    )
    raw_inputs = any(
        item["codec"] == "raw-fp32-le" for item in mapping["inputs"].values()
    )
    require(
        not raw_inputs or rules,
        "raw tensor inputs require declared multipart ingress.files and fixture inputs",
    )
    return ingress, request_schema, rules


def _capture_fixture_inputs(root, output, cases, rules):
    for index, case in enumerate(cases["cases"]):
        inputs, files = case.get("inputs", {}), case.get("files", {})
        require(
            isinstance(inputs, dict)
            and isinstance(files, dict)
            and not set(inputs) & set(files),
            "fixture inputs/files must be non-conflicting alias maps",
        )
        supplied = {
            **{name: {"source": source} for name, source in inputs.items()},
            **files,
        }
        require(
            set(supplied) <= set(rules),
            "fixture inputs must name declared ingress files",
        )
        require(
            all(
                not rule["required"] or name in supplied for name, rule in rules.items()
            ),
            "fixture is missing a required ingress file",
        )
        sealed_inputs, sealed_files = {}, {}
        for name, item in supplied.items():
            require(
                isinstance(item, dict) and isinstance(item.get("source"), str),
                "fixture files require an authored relative source path",
            )
            source = contained(root, item["source"])
            require(
                source.is_file() and not source.is_symlink(),
                "fixture input must be a regular non-symlink file",
            )
            identity = file_identity(source)
            media = (
                item.get("media_type")
                or mimetypes.guess_type(source.name)[0]
                or "application/octet-stream"
            )
            require(
                media in rules[name]["media_types"]
                and identity["size_bytes"] <= rules[name]["max_size_bytes"],
                "fixture file violates ingress size or media type policy",
            )
            require(
                all(item.get(key, value) == value for key, value in identity.items()),
                "fixture file differs from its declared identity",
            )
            target = output / "fixtures/inputs" / str(index) / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            require(
                file_identity(target) == identity,
                "fixture input changed during capture",
            )
            relative = target.relative_to(output).as_posix()
            sealed_inputs[name] = relative
            sealed_files[name] = {"source": relative, "media_type": media, **identity}
        if supplied:
            case["inputs"], case["files"] = sealed_inputs, sealed_files


def assemble(project, models, workflow, output, *, repository, environment, source_sha):
    from .config import validate_asset_alias

    root, output = Path(project["root"]), Path(output)
    require(not output.exists(), f"bundle output already exists: {output}")
    assets = project.get("assets", {})
    require(isinstance(assets, dict), "assets must be an alias map")
    for alias in assets:
        validate_asset_alias(alias)
    output.mkdir(parents=True)
    self_tests = read_json(
        root / project.get("fixtures", {}).get("self_test", "fixtures/self-test.json")
    )
    require(
        set(self_tests) == set(models),
        "self-test fixtures must cover every fixed model alias",
    )
    configuration = {
        "schema_version": 2,
        "adapter": "tensor.v1"
        if project.get("workflow", {}).get("template", "tensor") == "tensor"
        else "custom.v1",
        "models": {},
        "assets": {},
    }
    manifests = {}
    model_build_sources = {}
    for alias, info in models.items():
        source = info.get("compilation_source_sha")
        require(
            isinstance(source, str) and re.fullmatch(r"[a-f0-9]{40}", source),
            f"Model {alias} lacks a verified compilation source; rebuild the model",
        )
        model_build_sources[alias] = source
        model_root = Path(info["output_root"]) / "model"
        package = Path(info["package"])
        relative = package.relative_to(model_root)
        destination = output / "models" / alias
        shutil.copytree(model_root, destination)
        require(
            inventory(model_root) == inventory(destination),
            "compiled model bytes changed during bundling",
        )
        manifests[alias] = read_json(destination / relative / "model.json")
        configuration["models"][alias] = {
            "package_path": (Path("models") / alias / relative).as_posix(),
            "backend": project["backend"],
            "device": "cuda",
            "device_index": 0,
            "self_test": self_tests[alias],
        }
    shutil.copytree(workflow["install_root"], output / "workflow", symlinks=True)
    for alias, relative in assets.items():
        source = root / relative
        require(
            source.is_file() and not source.is_symlink(),
            f"asset must be a regular file: {source}",
        )
        destination = contained(output / "assets", alias, exists=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        configuration["assets"][alias] = destination.relative_to(
            output.resolve()
        ).as_posix()
    mapping, request_schema, result_schema = _contract(project, manifests)
    ingress, request_schema, input_files = _ingress_contract(
        project, request_schema, mapping, repository
    )
    if configuration["adapter"] == "tensor.v1":
        configuration["tensor_adapter"] = mapping
    else:
        configuration["custom"] = project.get("workflow", {}).get("custom", {})
    write_json(output / "configuration/workflow.json", configuration)
    write_json(output / "configuration/request.schema.json", request_schema)
    write_json(output / "configuration/result.schema.json", result_schema)
    cases_file = root / project.get("fixtures", {}).get("cases", "fixtures/cases.json")
    cases = read_json(cases_file)
    require(
        cases.get("schema_version") == 1 and cases.get("cases"),
        "independent nonempty workflow fixtures required",
    )
    _capture_fixture_inputs(root, output, cases, input_files)
    write_json(output / "fixtures/cases.json", cases)
    write_json(output / "fixtures/self-test.json", self_tests)
    # The projection uses the existing public plugin contract and pipeline.
    executor = "native." + project["name"]
    manifest = {
        "metadata": {
            "id": project["name"],
            "display_name": project["name"],
            "version": "0.1.0",
            "description": "Compiled native workflow",
        },
        "runtime": {
            "kind": "cpp",
            "entrypoint": configuration["adapter"],
            "executor_class": executor,
        },
        "ingress": ingress,
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
        "resources": {
            "defaults": {
                "gpus_required": 1,
                "memory_mb": project.get("serve", {}).get("memory_mb", 2048),
            }
        },
        "configuration": {
            "native": {"tensor_adapter": mapping}
            if configuration["adapter"] == "tensor.v1"
            else {"custom": configuration.get("custom", {})}
        },
        "outputs": {
            "result_schema": "result.schema.json",
            "primary_artifact": {
                "name": "native_result",
                "media_type": "application/json",
            },
            "retention_hours": 24,
        },
    }
    if project.get("_legacy_native"):
        # Preserve the original public plugin contract. Paths are projected
        # into the bundle, while authored metadata, queues and resource policy
        # remain authoritative and the original files stay untouched.
        import yaml

        from .config import UniqueLoader
        from .legacy import validate_worker_pipeline

        authored = yaml.load(
            contained(root, project["_legacy_native"]["manifest"]).read_text(),
            Loader=UniqueLoader,
        )
        validate_worker_pipeline(authored)
        executor = authored["runtime"]["executor_class"]
        authored["ingress"] = manifest["ingress"]
        authored["outputs"].pop("result_schema_inline", None)
        authored["outputs"]["result_schema"] = "result.schema.json"
        manifest = authored
    plugin = output / "configuration/plugin" / project["name"]
    write_json(plugin / "plugin.yaml", manifest)
    write_json(plugin / "request.schema.json", request_schema)
    write_json(plugin / "result.schema.json", result_schema)
    # Validate through the same schema expansion as the HTTP server.
    validated = legacy_module(repository, "plugin_runtime").load_plugin_manifest(
        plugin / "plugin.yaml"
    )
    schedule_queue = next(
        stage["queue"]
        for stage in validated["pipeline"]["stages"]
        if stage["phase"] == "schedule"
    )
    worker = read_json(
        Path(repository) / "native-inference/deploy/worker_runtime_config.json"
    )
    worker["streams"] = [
        schedule_queue if stream == "schedule" else stream
        for stream in worker["streams"]
    ]
    for item in worker["roles"]["scheduler"]["inputs"]:
        if item["stream"] == "schedule":
            item["stream"] = schedule_queue
    write_json(output / "configuration/worker_runtime_config.json", worker)
    descriptor = {
        "schema_version": 1,
        "name": project["name"],
        "runtime": "cpp",
        "backend": project["backend"],
        "source_sha": source_sha,
        "model_build_sources": model_build_sources,
        "workflow": {
            "config": "configuration/workflow.json",
            "runner": (
                Path("workflow")
                / Path(workflow["runner"]).relative_to(workflow["install_root"])
            ).as_posix(),
        },
        "environment": {
            key: value
            for key, value in environment.items()
            if key not in ("process_env",)
        },
        "models": {
            alias: configuration["models"][alias]["package_path"] for alias in models
        },
        "executor_class": executor,
        "runtime_deadlines": _runtime_deadlines(project.get("serve", {})),
    }
    if input_files:
        descriptor.update(
            input_files=input_files, manifest_version=manifest["metadata"]["version"]
        )
    write_json(output / "bundle.json", descriptor)
    files = inventory(output)
    seal = {"schema_version": 1, "files": files, "bundle_digest": digest(files)}
    write_json(output / "inventory.json", seal)
    verify(output)
    return seal


def verify(output, expected=None):
    output = Path(output)
    seal = read_json(contained(output, "inventory.json"))
    require(
        seal.get("schema_version") == 1
        and digest(seal["files"]) == seal.get("bundle_digest"),
        "bundle inventory digest differs",
    )
    require(
        expected is None or seal["bundle_digest"] == expected,
        "bundle digest differs from build",
    )
    verify_inventory(output, seal["files"], exclude=("inventory.json",))
    return seal


def compare(actual, expected, *, rtol=1e-5, atol=1e-6, where="fixture"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(expected) <= set(actual),
            f"{where}: expected object fields differ",
        )
        for key, value in expected.items():
            compare(actual[key], value, rtol=rtol, atol=atol, where=f"{where}.{key}")
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            f"{where}: expected array length differs",
        )
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            compare(a, b, rtol=rtol, atol=atol, where=f"{where}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"{where}: expected {expected}, got {actual}",
        )
    else:
        require(actual == expected, f"{where}: expected {expected}, got {actual}")


def materialize_deployment(bundle, output, *, artifact_root, input_cache):
    """Resolve portable references into a disposable host/image launch config."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    seal = verify(bundle)
    descriptor = read_json(bundle / "bundle.json")
    if descriptor.get("adapter") == "transolver_surface":
        from .cfd import materialize_deployment as materialize_cfd

        return materialize_cfd(
            bundle, output, artifact_root=artifact_root, input_cache=input_cache
        )
    workflow = read_json(bundle / descriptor["workflow"]["config"])
    sources = descriptor.get("model_build_sources")
    require(
        isinstance(sources, dict)
        and set(sources) == set(workflow["models"])
        and all(
            isinstance(source, str) and re.fullmatch(r"[a-f0-9]{40}", source)
            for source in sources.values()
        ),
        "Bundle lacks verified per-model compilation sources; rebuild the application",
    )
    for model in workflow["models"].values():
        model["package_path"] = str(contained(bundle, model["package_path"]))
    for alias, relative in workflow["assets"].items():
        workflow["assets"][alias] = str(contained(bundle, relative))
    gpu = descriptor["environment"]["gpu"]
    target = {
        "profile": "build-product-v1",
        "compilation": {
            "platform": "linux/amd64",
            "gpu_product": gpu["name"],
            "compute_capability": [gpu["major"], gpu["minor"]],
            "cuda_architecture": gpu.get(
                "architecture", f"sm{gpu['major']}{gpu['minor']}"
            ),
            "precision": "fp32",
            "mig": "disabled",
            "hardware_compatibility": "build-product-only",
        },
        "deployment": {
            "minimum_available_memory_mb": 256,
            "mig": "disabled",
            "driver_requirements": {
                "minimum_driver_version": gpu.get("driver_version", "0")
            },
        },
    }
    config = {
        "schema_version": 2,
        "workflow_id": descriptor["name"],
        "executor_class": descriptor["executor_class"],
        "device_index": 0,
        "target": target,
        "workflow": workflow,
        "model_build_sources": sources,
        "request_schema": read_json(bundle / "configuration/request.schema.json"),
        "response_schema": read_json(bundle / "configuration/result.schema.json"),
        "model_assets": [
            {
                "path": str(contained(bundle, item["path"])),
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in seal["files"]
            if item["path"].startswith(("models/", "assets/")) and "sha256" in item
        ],
        "input_artifacts": {},
        "artifact_root": str(Path(artifact_root).resolve()),
        "input_cache": str(Path(input_cache).resolve()),
        **_runtime_deadlines(descriptor.get("runtime_deadlines", {})),
    }
    if descriptor.get("input_files"):
        config.update(
            input_files=descriptor["input_files"],
            ingress_root=str(Path(artifact_root).resolve()),
            manifest_version=descriptor["manifest_version"],
        )
    write_json(output / "deployment.json", config)
    write_json(
        output / "release-inventory.json", {"bundle_digest": seal["bundle_digest"]}
    )
    return config
