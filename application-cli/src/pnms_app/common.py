"""Provider-independent artifact and source utilities."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


class AppError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AppError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, f"duplicate JSON field: {key}")
            value[key] = item
        return value

    def invalid(value):
        raise AppError(f"nonfinite JSON value: {value}")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def file_identity(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return {"sha256": result.hexdigest(), "size_bytes": Path(path).stat().st_size}


def contained(root, relative, *, exists=True):
    root, relative = Path(root).resolve(), Path(relative)
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        f"bundle reference must be relative and contained: {relative}",
    )
    result = root / relative
    require(
        result.resolve().is_relative_to(root),
        f"bundle reference escapes root: {relative}",
    )
    require(not exists or result.exists(), f"missing artifact: {result}")
    return result


IGNORED = {
    ".git",
    ".pnms",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".native-release",
    "builds",
}


def is_dotenv_path(path):
    return any(
        part.casefold() == ".env" or part.casefold().startswith(".env.")
        for part in Path(path).parts
    )


def inventory(root, *, exclude=(), source=False):
    root = Path(root).resolve()
    result = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.as_posix() in exclude:
            continue
        if source and (
            is_dotenv_path(relative)
            or any(
                part in IGNORED or part.startswith(".env") or part.endswith(".egg-info")
                for part in relative.parts
            )
        ):
            continue
        if path.is_symlink():
            require(
                path.resolve().is_relative_to(root) and path.resolve().is_file(),
                f"artifact symlink must name a file within its root: {path}",
            )
            result.append({"path": relative.as_posix(), "symlink": os.readlink(path)})
        elif path.is_file():
            result.append({"path": relative.as_posix(), **file_identity(path)})
        else:
            require(path.is_dir(), f"artifact must be a regular file: {path}")
    return result


def verify_inventory(root, records, *, exclude=()):
    require(
        isinstance(records, list) and inventory(root, exclude=exclude) == records,
        f"artifact inventory differs: {root}",
    )
    return records


def snapshot(source, output):
    source, output = Path(source).resolve(), Path(output)
    require(not output.exists(), f"snapshot already exists: {output}")
    files = inventory(source, source=True)
    output.mkdir(parents=True)
    for item in files:
        origin, target = source / item["path"], output / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if "symlink" in item:
            target.symlink_to(item["symlink"])
        else:
            shutil.copy2(origin, target)
            require(
                file_identity(target)
                == {key: item[key] for key in ("sha256", "size_bytes")},
                f"source changed during snapshot: {origin}",
            )
    return files


def repository_root(explicit=None):
    selected = explicit or os.environ.get("PNMS_REPOSITORY")
    candidates = (
        [Path(selected)]
        if selected
        else [Path(__file__).resolve().parents[3], *Path.cwd().parents, Path.cwd()]
    )
    for candidate in candidates:
        if (candidate / "scripts/plugin_dev.py").is_file() and (
            candidate / "native-inference"
        ).is_dir():
            return candidate.resolve()
    raise AppError(
        "PhysicsNeMo Serve source checkout required; set PNMS_REPOSITORY or --repository"
    )


def legacy_module(repository, name):
    for directory in (Path(repository) / "scripts", Path(repository) / "python"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    return importlib.import_module(name)
