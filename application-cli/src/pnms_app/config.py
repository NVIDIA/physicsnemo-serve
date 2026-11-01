"""Resolve authored runtime contracts without importing models or providers."""

from pathlib import Path, PureWindowsPath
import re

import yaml

from .common import contained, file_identity, legacy_module, read_json, require


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key, value in loader.construct_pairs(node, deep=deep):
        require(key not in result, f"duplicate YAML field: {key}")
        result[key] = value
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_asset_alias(alias):
    require(
        isinstance(alias, str)
        and bool(PureWindowsPath(alias).parts)
        and not PureWindowsPath(alias).anchor
        and ".." not in PureWindowsPath(alias).parts,
        "asset alias must be a relative contained path",
    )


def load_project(path, repository, *, require_sources=True):
    path = Path(path).resolve()
    root = path.parent if path.is_file() else path
    descriptor = root / "pnms.yaml"
    if descriptor.is_file():
        doc = yaml.load(descriptor.read_text(), Loader=UniqueLoader)
        require(
            isinstance(doc, dict) and doc.get("schema_version") == 1,
            "pnms.yaml requires schema_version: 1",
        )
        require(
            not set(doc)
            - {
                "schema_version",
                "name",
                "runtime",
                "models",
                "plugin",
                "build",
                "workflow",
                "serve",
                "assets",
                "fixtures",
                "schemas",
                "ingress",
            },
            "unknown pnms.yaml field",
        )
        kind, name = doc.get("runtime"), doc.get("name")
        require(kind in ("cpp", "python"), "runtime must be cpp or python")
    elif not require_sources and (root / ".pnms/last-build.json").is_file():
        state = root / ".pnms"
        selected = read_json(state / "last-build.json")
        receipt = contained(state, selected["receipt"])
        require(
            file_identity(receipt) == selected["identity"],
            "successful build receipt changed",
        )
        built = read_json(receipt)
        require(
            built.get("status") == "complete"
            and built.get("runtime") in ("python", "cpp"),
            "application requires a completed build; run pnms build",
        )
        kind, name = built["runtime"], built["name"]
        if kind == "python":
            doc = {"schema_version": 1, "name": name, "runtime": kind, "plugin": "."}
        else:
            from .bundle import verify

            bundle = contained(receipt.parent, built["bundle"])
            verify(bundle, built["bundle_digest"])
            captured = read_json(bundle / "bundle.json")
            configuration = read_json(contained(bundle, captured["workflow"]["config"]))
            template = (
                "transolver_surface"
                if captured.get("adapter") == "transolver_surface"
                else "tensor"
                if configuration["adapter"] == "tensor.v1"
                else "custom"
            )
            doc = {
                **built.get("application", {}),
                "schema_version": 1,
                "name": name,
                "runtime": kind,
                "models": {}
                if template == "transolver_surface"
                else {alias: {"project": "."} for alias in captured["models"]},
                "workflow": {"template": template},
                "build": {"backend": captured["backend"], "environment": "local"},
            }
    else:
        require(
            (root / "plugin.yaml").is_file(),
            f"No pnms.yaml or plugin.yaml found in {root}; use pnms init",
        )
        manifest = legacy_module(repository, "plugin_runtime").load_plugin_manifest(
            root / "plugin.yaml"
        )
        kind, name = manifest["runtime"]["kind"], manifest["metadata"]["id"]
        if kind == "cpp":
            from .legacy import native_document

            profile = (
                path.name
                if path.is_file() and path.name != "plugin.yaml"
                else "release.yaml"
            )
            doc = native_document(root, repository, manifest, profile=profile)
        else:
            doc = {"schema_version": 1, "name": name, "runtime": kind, "plugin": "."}
    require(
        isinstance(name, str)
        and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", name),
        "name must be a simple application identifier",
    )
    project = {**doc, "root": root, "name": name, "runtime": kind, "document": doc}
    require(isinstance(doc.get("build", {}), dict), "build must be a mapping")
    build = dict(doc.get("build", {}))
    for field in ("python", "runtime", "sdk_prefix", "venv", "tensorrt_root"):
        if build.get(field):
            build[field] = str((root / build[field]).absolute())
    project["build"] = build
    require(isinstance(doc.get("serve", {}), dict), "serve must be a mapping")
    require(isinstance(doc.get("ingress", {}), dict), "ingress must be a mapping")
    if (
        kind == "cpp"
        and isinstance(doc.get("workflow"), dict)
        and doc["workflow"].get("template") == "transolver_surface"
    ):
        from .cfd import resolve_project

        return resolve_project(project, repository, require_sources=require_sources)
    if kind == "python":
        require(
            build.get("environment", "local") == "local",
            "Python source builds require build.environment: local; container execution is native-only. Use --image for Python Docker release packaging",
        )
        native = set(build) & {
            "backend",
            "target",
            "sdk_prefix",
            "runtime",
            "tensorrt_root",
            "gpu",
            "builder_image",
        }
        require(
            not native and not set(doc) & {"models", "workflow", "ingress"},
            f"python project cannot use native options: {', '.join(sorted(native or (set(doc) & {'models', 'workflow'})))}",
        )
        plugin = (root / doc.get("plugin", ".")).resolve()
        if require_sources:
            manifest = legacy_module(repository, "plugin_runtime").load_plugin_manifest(
                plugin / "plugin.yaml"
            )
            require(
                manifest["runtime"]["kind"] == "python",
                "runtime python conflicts with plugin runtime",
            )
            project["manifest"] = manifest
        project["plugin"] = plugin
        return project
    require(
        "plugin" not in doc,
        "cpp project uses a generated serving plugin; remove plugin field",
    )
    require(isinstance(doc.get("assets", {}), dict), "assets must be an alias map")
    for alias in doc.get("assets", {}):
        validate_asset_alias(alias)
    require(
        build.get("environment", "local") in ("local", "container"),
        "build.environment must be local or container",
    )
    require(
        build.get("target", "current-gpu") == "current-gpu",
        "build.target must be current-gpu; build on the selected allocated GPU",
    )
    backend = build.get("backend", "aoti")
    require(backend in ("aoti", "tensorrt"), "build.backend must be aoti or tensorrt")
    models = doc.get("models")
    require(
        isinstance(models, dict) and models,
        "cpp project requires nonempty models alias map",
    )
    resolved = {}
    for alias, selection in models.items():
        require(
            isinstance(alias, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", alias),
            "model alias must be a simple identifier",
        )
        require(
            isinstance(selection, dict) and isinstance(selection.get("project"), str),
            f"models.{alias}.project required",
        )
        model = (root / selection["project"]).resolve()
        require(
            not require_sources or model.is_file(),
            f"models.{alias}.project missing: {model}",
        )
        resolved[alias] = {**selection, "project": model}
    workflow = doc.get("workflow", {"template": "tensor"})
    require(
        isinstance(workflow, dict)
        and workflow.get("template", "tensor") in ("tensor", "custom"),
        "workflow.template must be tensor or custom",
    )
    if require_sources and workflow.get("template", "tensor") == "custom":
        for field in ("source_dir", "target"):
            require(
                isinstance(workflow.get(field), str) and workflow[field].strip(),
                f"custom workflow requires nonempty workflow.{field}",
            )
        source = (root / workflow["source_dir"]).resolve()
        require(source.is_dir(), f"workflow.source_dir is missing: {source}")
        require(
            (source / "CMakeLists.txt").is_file(),
            f"custom workflow CMakeLists.txt is missing: {source / 'CMakeLists.txt'}",
        )
    project.update(
        models=resolved,
        backend=backend,
        build={"environment": "local", **build},
        workflow=workflow,
    )
    return project


def plan(project):
    return {
        "schema_version": 1,
        "status": "planned",
        "runtime": project["runtime"],
        "name": project["name"],
        "environment": project.get("build", {}).get("environment", "local"),
        "backend": project.get("backend"),
        "device_verification": "pending"
        if project["runtime"] == "cpp"
        else "not_required",
        "outputs": ["model_packages", "bundle"]
        if project["runtime"] == "cpp"
        else ["plugin_snapshot"],
        "checks": {"configuration": "pending", "native": "not_run", "http": "not_run"},
    }
