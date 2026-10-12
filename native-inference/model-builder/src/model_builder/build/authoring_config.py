"""Framework-free schema and input resolution for format-2 authoring projects."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from model_builder.export.aoti_profiles import validate_aoti_profile
from model_builder.export.aoti_options import validate_aoti_options
from model_builder.export.tensorrt_profiles import validate_tensorrt_profile

from . import projects


_PATH_FIELDS = {"adapter", "checkpoint", "runtime", "toolchain_lock", "output_root"}
_NULLABLE = {"checkpoint", "builder_image", "runtime", "toolchain_lock"}
_PROFILE_FIELDS = projects._EXECUTION_FIELDS | {
    "aoti_profile",
    "aoti_options",
    "tensorrt_profile",
}
_FIELDS = projects._EXECUTION_FIELDS | {
    "format_version",
    "name",
    "version",
    "adapter",
    "source",
    "checkpoint",
    "checkpoint_sha256",
    "config",
    "assets",
    "input_names",
    "output_names",
    "aoti_profile",
    "aoti_options",
    "tensorrt_profile",
    "profiles",
    "default_profile",
}


def _relative(value, field):
    projects._text(value, field)
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or (not path.parts and field != "source")
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{field} must be a contained relative project path.")
    return path


def _contained(root, value, field):
    relative = _relative(value, field)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{field} must not include a symlink: {value}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{field} must stay inside the project: {value}")
    return path


def validate_source(source, root):
    """Validate declared relative source roots without traversing their contents."""
    if not isinstance(source, list):
        raise ValueError(
            "source must be a list of relative project files or directories."
        )
    seen = set()
    for value in source:
        path = _contained(root, value, "source")
        normalized = str(path.relative_to(root))
        if normalized in seen:
            raise ValueError(f"duplicate source path: {value}")
        seen.add(normalized)


def _settings(document, prefix="", *, effective=False):
    if "aoti_profile" in document:
        validate_aoti_profile(document["aoti_profile"])
    if "tensorrt_profile" in document:
        validate_tensorrt_profile(document["tensorrt_profile"])
    if "aoti_options" in document:
        # Named profiles replace the complete map. Check precision-policy
        # compatibility only after inherited and selected settings are resolved.
        profile = document.get("aoti_profile", "baseline") if effective else "baseline"
        validate_aoti_options(document["aoti_options"], profile)
    projects._settings(
        {
            name: value
            for name, value in document.items()
            if name != "config" and not (name in _NULLABLE and value is None)
        },
        prefix,
    )


def _validate(document, root):
    if not isinstance(document, dict):
        raise ValueError("Authoring project configuration must be a JSON object.")
    if (
        type(document.get("format_version")) is not int
        or document["format_version"] != 2
    ):
        raise ValueError("Authoring project format_version must be integer 2.")
    unknown = document.keys() - _FIELDS
    if unknown:
        raise ValueError(
            f"Unknown authoring project field: {', '.join(sorted(unknown))}."
        )
    for name in ("name", "version"):
        value = document.get(name)
        if not isinstance(value, str) or not projects._IDENTIFIER.fullmatch(value):
            raise ValueError(f"{name} must be a nonempty simple identifier.")
    _contained(root, document.get("adapter"), "adapter")
    validate_source(document.get("source", []), root)
    config = document.get("config", {})
    if not isinstance(config, dict):
        projects._text(config, "config")
    for field in ("input_names", "output_names"):
        if field in document:
            values = document[field]
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(v, str) or not projects._IDENTIFIER.fullmatch(v)
                    for v in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"{field} must be a nonempty unique list of tensor names."
                )
    _settings(document)
    profiles = document.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be an object.")
    for name, profile in profiles.items():
        if not projects._IDENTIFIER.fullmatch(name) or not isinstance(profile, dict):
            raise ValueError(f"profiles.{name} must be a named settings object.")
        unknown = profile.keys() - _PROFILE_FIELDS
        if unknown:
            raise ValueError(
                f"Unsupported profiles.{name} field: {', '.join(sorted(unknown))}; "
                "profiles may only override execution and compiler settings."
            )
        _settings(profile, f"profiles.{name}.")
    default = document.get("default_profile")
    if "default_profile" in document and (
        not isinstance(default, str) or default not in profiles
    ):
        raise ValueError("default_profile must name an existing project profile.")


def load(path: Path, args=None) -> dict:
    """Resolve project < profile < CLI, retaining the untouched source identity.

    An unfinished checkpoint or environment is valid authoring configuration.
    Readiness is checked by ``missing`` and the execution planner, respectively.
    """
    path = projects._absolute(path)
    if path.is_dir():
        path = path / "model-build.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Project configuration must be a regular file: {path}")
    try:
        source = path.read_bytes()
        document = json.loads(
            source,
            object_pairs_hook=projects._object,
            parse_constant=projects._nonfinite,
            parse_float=projects._finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read project JSON {path}: {exc}") from exc
    _validate(document, path.parent)
    if args is not None and getattr(args, "recipe", None) is not None:
        raise ValueError("Choose an authoring project or --recipe, not both.")
    profile = getattr(args, "profile", None) or document.get("default_profile")
    profiles = document.get("profiles", {})
    if profile is not None and profile not in profiles:
        raise ValueError(f"Unknown project profile: {profile}")
    effective = copy.deepcopy(
        {
            key: value
            for key, value in document.items()
            if key not in ("profiles", "default_profile")
        }
    )
    effective.update(copy.deepcopy(profiles.get(profile, {})))
    for field, value in {
        "config": {},
        "source": [],
        "assets": {},
        "checkpoint": None,
        "backends": ["aoti"],
        "executor": "container",
        "device": "cuda",
        "output_root": "builds",
    }.items():
        effective.setdefault(field, value)
    for field in _PATH_FIELDS & effective.keys():
        if effective[field] is not None:
            effective[field] = str(projects._absolute(effective[field], path.parent))
    if isinstance(effective["config"], str):
        effective["config"] = str(projects._absolute(effective["config"], path.parent))
    effective["assets"] = {
        name: str(projects._absolute(value, path.parent))
        for name, value in effective["assets"].items()
    }
    if args is not None:
        for field, attribute in projects._ARGUMENTS.items():
            value = getattr(args, attribute, None)
            if value is not None:
                if field not in _FIELDS:
                    raise ValueError(
                        f"--{attribute.replace('_', '-')} is not supported for authoring projects."
                    )
                effective[field] = (
                    str(projects._absolute(value))
                    if field in _PATH_FIELDS or field == "config"
                    else value
                )
        cli_assets = {}
        for selection in getattr(args, "asset", []) or []:
            name, separator, value = selection.partition("=")
            if not separator or not name or not value:
                raise ValueError(
                    "Asset overrides must use NAME=FILE with a nonempty name and file."
                )
            if name in cli_assets:
                raise ValueError(f"duplicate asset override: {name}")
            cli_assets[name] = str(projects._absolute(value))
        effective["assets"].update(cli_assets)
        if getattr(args, "backend", None):
            effective["backends"] = list(args.backend)
    _settings(effective, effective=True)
    return {
        "path": path,
        "profile": profile,
        "document": document,
        "effective": effective,
        "source_identity": {
            "sha256": hashlib.sha256(source).hexdigest(),
            "size_bytes": len(source),
        },
    }


def missing(
    effective: dict, project_root: Path, *, require_runtime: bool = True
) -> list[dict]:
    """Describe missing model inputs without importing model code or touching Docker."""
    diagnostics = []

    def report(code, field, message):
        diagnostics.append({"code": code, "field": field, "message": message})

    def regular(value, field):
        if value is None:
            report(
                "AUTHORING_INPUT_REQUIRED",
                field,
                f"Set {field} in model-build.json before checking or building your model.",
            )
            return False
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        if path.is_symlink() or not path.is_file():
            report(
                "AUTHORING_FILE_MISSING",
                field,
                f"{field} must point to an existing regular file: {path}",
            )
            return False
        return True

    regular(effective.get("checkpoint"), "checkpoint")
    config = effective.get("config", {})
    if isinstance(config, str) and regular(config, "config"):
        try:
            value = json.loads(
                Path(config).read_bytes(),
                object_pairs_hook=projects._object,
                parse_constant=projects._nonfinite,
                parse_float=projects._finite_float,
            )
            if not isinstance(value, dict):
                raise ValueError("configuration must contain a JSON object")
        except (OSError, ValueError) as exc:
            report("AUTHORING_CONFIG_INVALID", "config", f"Cannot load config: {exc}")
    for name, path in effective.get("assets", {}).items():
        regular(path, f"assets.{name}")
    for value in effective.get("source", []):
        path = project_root / value
        if not (path.is_file() or path.is_dir()):
            report(
                "AUTHORING_SOURCE_MISSING",
                "source",
                f"source entry does not exist: {value}; select your Python model files.",
            )
    adapter = effective.get("adapter")
    if regular(adapter, "adapter"):
        try:
            module = ast.parse(Path(adapter).read_text(encoding="utf-8"))
            functions = {
                node.name: node
                for node in module.body
                if isinstance(node, ast.FunctionDef)
            }
            for name in ("create_model", "create_cases"):
                function = functions.get(name)
                if function is None:
                    report(
                        "AUTHORING_ADAPTER_INCOMPLETE",
                        "adapter",
                        f"Define {name}(config, assets) in {adapter}.",
                    )
                    continue
                body = [
                    node
                    for node in function.body
                    if not (
                        isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                    )
                ]
                if len(body) == 1 and isinstance(body[0], ast.Raise):
                    exception = body[0].exc
                    target = (
                        exception.func if isinstance(exception, ast.Call) else exception
                    )
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "NotImplementedError"
                    ):
                        report(
                            "AUTHORING_ADAPTER_INCOMPLETE",
                            "adapter",
                            f"Edit {adapter}:{name} to connect your model and validation inputs.",
                        )
        except (OSError, ValueError, SyntaxError) as exc:
            report(
                "AUTHORING_ADAPTER_INVALID", "adapter", f"Cannot read adapter: {exc}"
            )
    if not effective.get("toolchain_lock"):
        if effective.get("executor", "container") == "container":
            if not effective.get("builder_image"):
                report(
                    "MISSING_BUILDER_IMAGE",
                    "builder_image",
                    "Set builder_image in model-build.json to your configured builder image; "
                    "there is no published default image yet.",
                )
        elif require_runtime and not effective.get("runtime"):
            report(
                "MISSING_RUNTIME",
                "runtime",
                "Set runtime in model-build.json to the installed physicsnemo-infer executable.",
            )
    return diagnostics
