"""Framework-free resolution and retention of model build inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil


def _digest(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"{label} SHA-256 must contain 64 lowercase hexadecimal characters"
        )
    return value


def _relative(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} requires a relative file path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} path must remain within the recipe directory")
    return path


def validate_input_spec(recipe):
    """Validate declarations without importing frameworks or opening input files."""
    if type(recipe.get("format_version")) is not int or recipe[
        "format_version"
    ] not in (1, 2):
        raise ValueError("Unsupported recipe format_version; expected 1 or 2")
    if recipe["format_version"] == 1:
        return
    adapter = _relative(recipe.get("adapter"), "adapter")
    if adapter.parts[0] == "model-inputs":
        raise ValueError("adapter cannot use the reserved model-inputs directory")
    assets = recipe.get("assets", {})
    if not isinstance(assets, dict):
        raise ValueError("assets must be an object of named file descriptors")
    for name in assets:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("asset names must be simple identifiers")
    descriptors = [
        ("config", recipe.get("config")),
        ("checkpoint", recipe.get("checkpoint")),
    ]
    descriptors += [
        (f"asset {name}", descriptor) for name, descriptor in assets.items()
    ]
    for label, descriptor in descriptors:
        allowed = (
            {"path", "sha256", "format"}
            if label == "checkpoint"
            else {"path", "sha256"}
        )
        if not isinstance(descriptor, dict) or set(descriptor) - allowed:
            raise ValueError(
                f"{label} must be a file descriptor with fields {sorted(allowed)}"
            )
        if label == "checkpoint" and descriptor.get("format") != "torch-state-dict":
            raise ValueError("checkpoint format must be torch-state-dict")
        if label.startswith("asset ") and "path" not in descriptor:
            raise ValueError(f"{label} requires a default path")
        if "path" in descriptor:
            _relative(descriptor["path"], label)
        if "sha256" in descriptor:
            _digest(descriptor["sha256"], label)


def _regular_file(path, label, *, base=None):
    path = Path(path).expanduser().absolute()
    if base is None:
        # Explicit selections can be outside the recipe, including through OS
        # directory aliases such as macOS /tmp. Reject a symlink as the file.
        if path.is_symlink():
            raise ValueError(f"{label} path is a symlink: {path}")
    else:
        candidate = base
        for part in path.relative_to(base).parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError(f"{label} path contains a symlink: {candidate}")
    if not path.is_file():
        raise ValueError(f"{label} must be an existing regular file: {path}")
    return path.resolve()


def _identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def _canonical(config):
    return (
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _configuration(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite value {value}")

    try:
        config = json.loads(
            path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant
        )
        if not isinstance(config, dict):
            raise ValueError("expected a JSON object")
        canonical = _canonical(
            config
        )  # Also rejects floating-point overflow such as 1e999.
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    return config, {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "size_bytes": len(canonical),
    }


def resolve_inputs(
    recipe,
    recipe_path,
    *,
    config=None,
    checkpoint=None,
    checkpoint_sha256=None,
    assets=None,
):
    """Resolve whole-file overrides and record content identities before ML work."""
    validate_input_spec(recipe)
    if recipe["format_version"] == 1:
        if (
            any(value is not None for value in (config, checkpoint, checkpoint_sha256))
            or assets
        ):
            raise ValueError("Model input overrides require recipe format_version 2")
        return None
    base = Path(recipe_path).parent.resolve()
    declared_assets = recipe.get("assets", {})
    assets = assets or {}
    unknown = set(assets) - set(declared_assets)
    if unknown:
        raise ValueError(f"Unknown asset override: {', '.join(sorted(unknown))}")
    if checkpoint_sha256 is not None:
        _digest(checkpoint_sha256, "checkpoint")

    def select(label, descriptor, override, explicit_pin=None):
        if override is None and "path" not in descriptor:
            raise ValueError(f"Recipe requires --{label} FILE")
        path = _regular_file(
            override if override is not None else base / descriptor["path"],
            label,
            base=None if override is not None else base,
        )
        actual = _identity(path)
        pins = [explicit_pin]
        if override is None:
            pins.append(descriptor.get("sha256"))
        for pin in pins:
            if pin is not None and pin != actual["sha256"]:
                raise ValueError(f"{label} SHA-256 mismatch: {path}")
        return {
            "path": str(path),
            **actual,
            "origin": "cli" if override is not None else "recipe",
        }

    result = {
        "config": select("config", recipe["config"], config),
        "checkpoint": select(
            "checkpoint", recipe["checkpoint"], checkpoint, checkpoint_sha256
        ),
        "assets": {
            name: select(f"asset {name}", descriptor, assets.get(name))
            for name, descriptor in declared_assets.items()
        },
    }
    result["config_data"], result["effective_config"] = _configuration(
        Path(result["config"]["path"])
    )
    # Detect a change between hashing and parsing the configuration.
    if _identity(Path(result["config"]["path"])) != {
        key: result["config"][key] for key in ("sha256", "size_bytes")
    }:
        raise ValueError("config changed while resolving inputs")
    return result


def input_identities(resolved):
    if resolved is None:
        return None

    def identity(descriptor):
        return {key: descriptor[key] for key in ("sha256", "size_bytes")}

    return {
        "config": identity(resolved["config"]),
        "checkpoint": identity(resolved["checkpoint"]),
        "assets": {name: identity(value) for name, value in resolved["assets"].items()},
        "effective_config": identity(resolved["effective_config"]),
    }


def stage_inputs(resolved, destination):
    """Retain verified input bytes, without overwriting a previous staging area."""
    if resolved is None:
        return None
    root = Path(destination).resolve() / "model-inputs"
    root.mkdir(parents=True, exist_ok=False)
    staged = copy.deepcopy(resolved)

    def retain(descriptor, target, label):
        original = _regular_file(descriptor["path"], label)
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        expected = {key: descriptor[key] for key in ("sha256", "size_bytes")}
        if _identity(target) != expected:
            raise ValueError(
                f"{label} changed since resolution: input integrity mismatch"
            )
        descriptor["path"] = str(target)

    retain(staged["config"], root / "config.json", "config")
    retain(staged["checkpoint"], root / "checkpoint.pt", "checkpoint")
    for name, descriptor in staged["assets"].items():
        retain(
            descriptor,
            root / "assets" / name / Path(descriptor["path"]).name,
            f"asset {name}",
        )
    staged["config_data"], canonical_identity = _configuration(
        Path(staged["config"]["path"])
    )
    if canonical_identity != input_identities(resolved)["effective_config"]:
        raise ValueError(
            "config changed since resolution: effective configuration integrity mismatch"
        )
    canonical_path = root / "effective-config.json"
    canonical_path.write_bytes(_canonical(staged["config_data"]))
    staged["effective_config"] = {**canonical_identity, "path": str(canonical_path)}
    return staged


def effective_recipe(recipe, resolved, source_root):
    """Create a portable replay recipe; leave the producer's recipe unchanged."""
    result = copy.deepcopy(recipe)
    if resolved is None:
        return result
    root = Path(source_root).resolve()

    def descriptor(value):
        return {
            "path": Path(value["path"]).relative_to(root).as_posix(),
            "sha256": value["sha256"],
        }

    result["config"] = descriptor(resolved["config"])
    result["checkpoint"] = {
        "format": "torch-state-dict",
        **descriptor(resolved["checkpoint"]),
    }
    result["assets"] = {
        name: descriptor(value) for name, value in resolved["assets"].items()
    }
    return result
