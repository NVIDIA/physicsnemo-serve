"""Framework-free customer project configuration resolution."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_EXECUTION_FIELDS = {
    "backends",
    "executor",
    "device",
    "runtime",
    "builder_image",
    "toolchain_lock",
    "output_root",
    "required_gpu_arch",
}
_FIELDS = _EXECUTION_FIELDS | {
    "format_version",
    "model",
    "recipe",
    "checkpoint",
    "checkpoint_sha256",
    "config",
    "assets",
    "profiles",
    "default_profile",
}
_PATH_FIELDS = {
    "recipe",
    "checkpoint",
    "config",
    "runtime",
    "toolchain_lock",
    "output_root",
}
_ARGUMENTS = {
    "checkpoint": "checkpoint",
    "checkpoint_sha256": "checkpoint_sha256",
    "config": "config",
    "executor": "executor",
    "device": "device",
    "runtime": "runtime",
    "builder_image": "builder_image",
    "toolchain_lock": "lock",
    "required_gpu_arch": "required_gpu_arch",
}


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _nonfinite(value):
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _nonfinite(value)
    return number


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{field} must be a nonempty string without NUL characters.")


def _settings(document, prefix=""):
    for name, value in document.items():
        field = prefix + name
        if name in _PATH_FIELDS or name == "builder_image":
            _text(value, field)
        elif name == "checkpoint_sha256":
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
        elif name == "backends":
            if (
                not isinstance(value, list)
                or not value
                or any(item not in ("aoti", "tensorrt") for item in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(
                    f"{field} must be a nonempty unique list of aoti or tensorrt."
                )
        elif name == "executor":
            if value not in ("container", "local"):
                raise ValueError(f"{field} must be container or local.")
        elif name == "device":
            if not isinstance(value, str) or not re.fullmatch(
                r"cpu|cuda(?::[0-9]+)?", value
            ):
                raise ValueError(f"{field} must be cpu, cuda, or cuda:<index>.")
        elif name == "required_gpu_arch":
            if not isinstance(value, str) or not re.fullmatch(r"sm[0-9]{2,3}", value):
                raise ValueError(f"{field} must use a compute capability such as sm90.")
        elif name == "assets":
            if not isinstance(value, dict):
                raise ValueError(
                    f"{field} must be an object mapping names to file paths."
                )
            for asset, path in value.items():
                if not _IDENTIFIER.fullmatch(asset):
                    raise ValueError(f"Invalid asset name: {asset!r}.")
                _text(path, f"{field}.{asset}")


def _validate(document):
    if not isinstance(document, dict):
        raise ValueError("Project configuration must be a JSON object.")
    if (
        type(document.get("format_version")) is not int
        or document["format_version"] != 1
    ):
        raise ValueError("Project format_version must be integer 1.")
    unknown = document.keys() - _FIELDS
    if unknown:
        raise ValueError(f"Unknown project field: {', '.join(sorted(unknown))}.")
    if ("model" in document) == ("recipe" in document):
        raise ValueError("Project requires exactly one of model or recipe.")
    if "model" in document:
        model = document["model"]
        if not isinstance(model, str) or not _IDENTIFIER.fullmatch(model):
            raise ValueError("Project model must be a simple model identifier.")
    _settings(document)
    profiles = document.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("Project profiles must be an object.")
    for name, profile in profiles.items():
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"Invalid profile name: {name!r}.")
        if not isinstance(profile, dict):
            raise ValueError(f"profiles.{name} must be an object.")
        unknown = profile.keys() - _EXECUTION_FIELDS
        if unknown:
            raise ValueError(
                f"Unsupported profiles.{name} field: {', '.join(sorted(unknown))}; "
                "profiles may only override execution settings."
            )
        _settings(profile, f"profiles.{name}.")
    if "default_profile" in document:
        default = document["default_profile"]
        if not isinstance(default, str) or default not in profiles:
            raise ValueError("default_profile must name an existing project profile.")


def _absolute(value, parent=None):
    # Keep symlink leaves visible to the existing input validators.
    path = Path(value).expanduser()
    if parent is not None and not path.is_absolute():
        path = parent / path
    path = path.absolute()
    if path.name in ("", ".", ".."):
        return path.resolve()
    return path.parent.resolve() / path.name


def _project_path(model):
    if not model:
        return None
    candidate = _absolute(model)
    if candidate.is_dir():
        return candidate / "model-build.json"
    if candidate.is_file():
        if candidate.name != "model-build.json":
            raise ValueError(
                "A project file must be named model-build.json; use --recipe for recipes."
            )
        return candidate
    if (
        "/" in str(model)
        or "\\" in str(model)
        or str(model).startswith((".", "~"))
        or str(model).endswith(".json")
    ):
        raise ValueError(f"Cannot find project: {model}")
    return None


def apply_project(args) -> dict | None:
    """Resolve project defaults < profile < explicit CLI without writing files.

    ``executor`` and ``device`` parser defaults must be None so explicit CLI
    selections remain distinguishable. All returned identities describe the
    supplied authoring file; existing recipe and toolchain validators remain
    responsible for execution readiness.
    """
    path = _project_path(args.model)
    if path is None:
        if getattr(args, "profile", None) is not None:
            raise ValueError(
                "--profile requires a project directory or model-build.json."
            )
        return None
    if args.recipe is not None:
        raise ValueError("Choose a project or --recipe, not both.")
    if path.is_symlink():
        raise ValueError(
            "The project configuration must be a regular file, not a symlink."
        )
    try:
        source = path.read_bytes()
        document = json.loads(
            source,
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read project JSON {path}: {exc}") from exc
    _validate(document)
    profile = getattr(args, "profile", None) or document.get("default_profile")
    profiles = document.get("profiles", {})
    if profile is not None and profile not in profiles:
        raise ValueError(f"Unknown project profile: {profile}")
    effective = {
        name: value
        for name, value in document.items()
        if name not in ("profiles", "default_profile")
    }
    effective.update(profiles.get(profile, {}))
    effective.setdefault("executor", "container")
    effective.setdefault("device", "cuda")
    for field in _PATH_FIELDS & effective.keys():
        effective[field] = str(_absolute(effective[field], path.parent))
    assets = {
        name: str(_absolute(value, path.parent))
        for name, value in effective.get("assets", {}).items()
    }
    cli_assets = {}
    for selection in args.asset:
        name, separator, value = selection.partition("=")
        if not separator or not name or not value:
            raise ValueError(
                "Asset overrides must use NAME=FILE with a nonempty name and file."
            )
        if name in cli_assets:
            raise ValueError(f"duplicate asset override: {name}")
        cli_assets[name] = str(_absolute(value))
    assets.update(cli_assets)
    if assets or "assets" in effective:
        effective["assets"] = assets
    for field, attribute in _ARGUMENTS.items():
        value = getattr(args, attribute, None)
        if value is not None:
            effective[field] = str(_absolute(value)) if field in _PATH_FIELDS else value
    cli_backends = args.backend
    if cli_backends:
        effective["backends"] = list(cli_backends)
    backends = list(effective.get("backends", []))
    args.model = effective.get("model")
    args.recipe = Path(effective["recipe"]) if "recipe" in effective else None
    for field, attribute in _ARGUMENTS.items():
        if field in effective:
            value = effective[field]
            setattr(args, attribute, Path(value) if field in _PATH_FIELDS else value)
    args.asset = [f"{name}={value}" for name, value in assets.items()]
    args.backend = backends
    if args.output is not None:
        args.output = _absolute(args.output)
    elif "output_root" in effective:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = (
            Path(effective["output_root"]) / f"{stamp}-{uuid.uuid4().hex[:12]}"
        )
    return {
        "path": path,
        "profile": profile,
        "effective": effective,
        "source_identity": {
            "sha256": hashlib.sha256(source).hexdigest(),
            "size_bytes": len(source),
        },
    }
