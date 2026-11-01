"""Read existing native authoring inputs without invoking managed release tools."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .common import contained, legacy_module, require


def validate_worker_pipeline(manifest):
    """Admit only authored pipelines honored by the native worker's wire format."""
    stages = manifest["pipeline"]["stages"]
    require(
        [stage["phase"] for stage in stages] == ["schedule", "execute", "results"],
        "native local adoption requires the schedule/execute/results pipeline",
    )
    # NativeRole publishes a fixed results stream/stage and never advances
    # through an authored successor after executing the native workflow.
    results = stages[2]
    require(
        results["id"] == "results",
        "native worker requires the results stage id to be 'results'",
    )
    require(
        results["queue"] == "results",
        "native worker requires the results queue to be 'results'",
    )
    require(
        results.get("next") is None,
        "native worker requires a terminal results stage (next: null)",
    )
    schedule_queue = stages[0]["queue"]
    require(
        schedule_queue not in {"release", "results", "dlq"},
        f"native schedule queue conflicts with reserved stream {schedule_queue!r}",
    )
    require(
        not schedule_queue.endswith(":grp"),
        "native schedule queue cannot use the reserved :grp consumer-group suffix",
    )


def native_document(root, repository, manifest, *, profile="release.yaml"):
    """Project the original plugin/profile into the common local build contract."""
    root = Path(root)
    from .config import UniqueLoader, validate_asset_alias

    release = yaml.load(contained(root, profile).read_text(), Loader=UniqueLoader)
    require(
        isinstance(release, dict) and release.get("schema_version") == 1,
        "native release profile requires schema_version: 1",
    )
    name = manifest["metadata"]["id"]
    require(
        release.get("workflow") == name,
        "release workflow differs from plugin metadata.id",
    )
    backend = release.get("backend")
    require(
        backend in ("aoti", "tensorrt"),
        "native release backend must be aoti or tensorrt",
    )
    adapter = manifest["runtime"]["entrypoint"]
    if adapter == "transolver_surface":
        from .cfd import native_document as cfd_document

        return cfd_document(root, repository, manifest, release, profile=profile)
    from .bundle import _runtime_deadlines

    deadlines = _runtime_deadlines(release.get("runtime", {}), request_timeout=600)
    require(
        adapter in ("tensor.v1", "custom.v1"),
        f"native adapter {adapter} requires its domain application adapter",
    )
    # The original three native stages are authoritative; do not silently drop
    # preprocessing, fanout or publication phases from an authored pipeline.
    validate_worker_pipeline(manifest)
    has_models, has_project = "models" in release, "model_project" in release
    require(
        has_models != has_project,
        "native release requires exactly one of models or model_project",
    )
    selections = (
        release["models"]
        if has_models
        else {"primary": {"project": release["model_project"]}}
    )
    require(
        isinstance(selections, dict) and selections,
        "native models must be a nonempty alias map",
    )
    for item in selections.values():
        require(
            isinstance(item, dict) and isinstance(item.get("project"), str),
            "native model selection requires project",
        )
        contained(root, item["project"])
    qualification = release.get("qualification", {})
    for key in ("cases", "self_test"):
        require(
            isinstance(qualification.get(key), str),
            "native qualification requires " + key,
        )
        contained(root, qualification[key])
    helper = legacy_module(repository, "native_workflow")
    ingress = copy.deepcopy(manifest["ingress"])
    request_schemas = [
        helper._schema(
            root,
            ingress,
            "form_schema" if kind == "multipart/form-data" else "json_schema",
        )
        for kind in ingress["content_types"]
    ]
    require(
        request_schemas
        and all(value == request_schemas[0] for value in request_schemas),
        "native ingress requires one canonical parameter schema",
    )
    result_schema = helper._schema(root, manifest["outputs"], "result_schema")
    settings = copy.deepcopy(manifest.get("configuration", {}).get("native", {}))
    workflow = {"template": "tensor" if adapter == "tensor.v1" else "custom"}
    if adapter == "tensor.v1":
        mapping = settings.get("tensor_adapter")
        workflow["tensor_adapter"] = (
            {"model_alias": "primary", **mapping}
            if isinstance(mapping, dict)
            else mapping
        )
    else:
        selected = release.get("native_workflow", {})
        workflow.update(
            source_dir=selected.get("source_dir"),
            target=selected.get("cmake_target"),
            custom=settings.get("custom", {}),
        )
    assets = {}
    for asset in release.get("assets", []):
        require(isinstance(asset, dict), "native asset must be an object")
        source = asset.get("source")
        contained(root, source)
        destination = asset.get("destination", "")
        alias = asset.get("name", Path(destination).stem)
        require(
            isinstance(alias, str) and alias and alias not in assets,
            "native assets require unique names",
        )
        validate_asset_alias(alias)
        assets[alias] = source
    require(
        not release.get("input_artifacts"),
        "fixed native inputs require a supported domain adapter; use declared assets or request uploads",
    )
    images = release.get("images", {})
    return {
        "schema_version": 1,
        "name": name,
        "runtime": "cpp",
        "models": copy.deepcopy(selections),
        "build": {
            "backend": backend,
            "environment": "local",
            "target": "current-gpu",
            **({"builder_image": images["builder"]} if images.get("builder") else {}),
        },
        "workflow": workflow,
        "assets": assets,
        "schemas": {"request": request_schemas[0], "result": result_schema},
        "ingress": ingress,
        "fixtures": {key: qualification[key] for key in ("cases", "self_test")},
        "serve": {
            **deadlines,
            "memory_mb": manifest.get("resources", {})
            .get("defaults", {})
            .get("memory_mb", 2048),
            **({"image": images["server"]} if images.get("server") else {}),
        },
        "_legacy_native": {"manifest": "plugin.yaml", "profile": profile},
    }
