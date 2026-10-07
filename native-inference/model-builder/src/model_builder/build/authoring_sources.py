"""Capture and import explicitly selected customer model sources."""

from __future__ import annotations

import copy
import hashlib
import importlib
from importlib.machinery import PathFinder
import os
from pathlib import Path
import sys
import tempfile

from .inputs import _identity


_EXCLUDED = {"__pycache__", "build", "builds", "dist", "node_modules", "venv", "env"}
# The worker runs one model in an isolated process. Retain imports and temporary
# source files until process exit because forward/export can import lazily.
_TREES = {}
_IMPORT_ROOTS = set()


def _relative(value, *, allow_root=False):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Model source requires a project-relative Python path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or (not path.parts and not allow_root):
        raise ValueError("Model source path must remain within the project directory")
    return path


def _selected(root, relative):
    selected = root
    for part in relative.parts:
        selected /= part
        if selected.is_symlink():
            raise ValueError(f"Model source path contains a symlink: {selected}")
    return selected


def _source_file(path):
    if path.is_symlink():
        raise ValueError(f"Model source path is a symlink: {path}")
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(
            f"Model source must be an existing regular Python file: {path}"
        )
    return path


def capture(root, selections, adapter, *, exclude=()):
    """Return sorted project-relative sources with whole-file content identities.

    Selection is explicit. Directories contribute Python files only, excluding
    hidden/build/cache directories and optional output directories in ``exclude``.
    Exclusions can be absolute or project-relative; the adapter is always required.
    This function does not import customer code or any ML framework.
    """
    root = Path(root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            "Model source root must be an existing directory, not a symlink"
        )
    root = root.resolve()
    if not isinstance(selections, list) or not all(
        isinstance(item, str) for item in selections
    ):
        raise ValueError(
            "Model sources must be a list of project-relative files or directories"
        )
    excluded = [
        Path(os.path.abspath(root / Path(value).expanduser())) for value in exclude
    ]

    def is_excluded(path):
        return any(path.is_relative_to(directory) for directory in excluded)

    def select(value, *, allow_root=False):
        relative = _relative(value, allow_root=allow_root)
        if is_excluded(root / relative):
            raise ValueError(
                f"Model source selects an excluded output directory: {value}"
            )
        return _selected(root, relative)

    adapter_path = select(adapter)
    selected = {_source_file(adapter_path)}

    def visit(directory):
        for child in sorted(directory.iterdir()):
            if (
                child.name.startswith(".")
                or child.name in _EXCLUDED
                or is_excluded(child)
            ):
                continue
            if child.is_symlink():
                raise ValueError(f"Model source path contains a symlink: {child}")
            if child.is_dir():
                visit(child)
            elif child.suffix == ".py":
                selected.add(_source_file(child))

    for value in selections:
        path = select(value, allow_root=True)
        if path.is_dir():
            visit(path)
        else:
            selected.add(_source_file(path))
    return {
        path.relative_to(root).as_posix(): {"path": str(path), **_identity(path)}
        for path in sorted(selected)
    }


def _source_data(config, assets):
    if not isinstance(config, dict) or not isinstance(assets, dict):
        raise ValueError("Captured model configuration and assets must be objects")
    files = config.get("_source_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Captured model requires a nonempty _source_files mapping")
    names = list(files.values())
    if not all(isinstance(name, str) and name for name in names) or len(
        set(names)
    ) != len(names):
        raise ValueError("Captured model source asset names must be unique strings")
    paths = [_relative(name) for name in files]
    if len(set(paths)) != len(paths):
        raise ValueError("Captured model source paths must be unique")
    directories = {parent for path in paths for parent in path.parents if parent.parts}
    modules = {path.with_suffix("") for path in paths if path.name != "__init__.py"}
    conflicts = (set(paths) | modules) & directories
    if conflicts:
        raise ValueError(
            "Captured model module/package path conflict: "
            + ", ".join(str(path) for path in sorted(conflicts))
        )
    adapter = _relative(config.get("_adapter"))
    if adapter not in paths or adapter.suffix != ".py":
        raise ValueError("Captured model adapter must name a captured Python source")
    model_config = config.get("_model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Captured model _model_config must be an object")
    user_assets = config.get("_user_assets")
    if (
        not isinstance(user_assets, list)
        or not all(isinstance(name, str) and name for name in user_assets)
        or len(set(user_assets)) != len(user_assets)
        or set(user_assets) & set(names)
    ):
        raise ValueError(
            "Captured model _user_assets must be unique names separate from source assets"
        )
    missing = (set(names) | set(user_assets)) - set(assets)
    if missing:
        raise ValueError(
            f"Captured model is missing assets: {', '.join(sorted(missing))}"
        )
    data = {}
    for relative, asset in zip(paths, names):
        if relative.suffix != ".py":
            raise ValueError("Captured model sources must be Python files")
        try:
            source = _source_file(Path(assets[asset]))
        except TypeError as exc:
            raise ValueError(
                f"Captured model source asset {asset!r} must be a file path"
            ) from exc
        data[relative] = source.read_bytes()
    forwarded_assets = {}
    for name in user_assets:
        try:
            path = Path(assets[name])
        except TypeError as exc:
            raise ValueError(f"Model asset {name!r} must be a file path") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Model asset {name!r} must be an existing regular file")
        forwarded_assets[name] = path
    return adapter, data, model_config, forwarded_assets


def _module_name(adapter):
    parts = list(adapter.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(part.isidentifier() for part in parts):
        raise ValueError("Model adapter path must form a valid Python module name")
    return ".".join(parts)


def _inside(module, directory):
    namespace = getattr(module, "__dict__", {})
    locations = [namespace.get("__file__")]
    locations.extend(namespace.get("__path__", ()) or ())
    return any(
        value and Path(value).resolve().is_relative_to(directory) for value in locations
    )


def _verify_origins(directory, data):
    directory = directory.resolve()
    modules = {}
    for relative in data:
        path = (
            relative.parent
            if relative.name == "__init__.py"
            else relative.with_suffix("")
        )
        if not all(part.isidentifier() for part in path.parts):
            continue
        for parent in (path, *path.parents):
            if parent.parts:
                modules[".".join(parent.parts)] = parent
    search_paths = {"": None}
    for name, relative in sorted(modules.items()):
        expected = None
        for candidate in (relative / "__init__.py", relative.with_suffix(".py")):
            if candidate in data:
                expected = directory / candidate
                break
        if name in sys.modules:
            module = sys.modules[name]
            origin = getattr(module, "__file__", None)
            locations = getattr(module, "__path__", None)
        else:
            # Search each component without importing its parents. The short
            # name also keeps namespace specs independent of sys.modules.
            spec = PathFinder.find_spec(
                relative.name, search_paths[name.rpartition(".")[0]]
            )
            origin = spec.origin if spec else None
            locations = spec.submodule_search_locations if spec else None
        locations = list(locations or ())
        if expected is not None:
            valid = origin is not None and Path(origin).resolve() == expected
        else:
            valid = origin is None and directory / relative in {
                Path(location).resolve() for location in locations
            }
        if not valid:
            raise ValueError(
                f"Captured model source import conflict for {name}: "
                "module must resolve to its captured source. "
                "Rename the conflicting package or use a clean environment."
            )
        search_paths[name] = locations


def _load(adapter, data):
    key = (
        str(adapter),
        tuple(
            (str(path), hashlib.sha256(content).hexdigest())
            for path, content in sorted(data.items())
        ),
    )
    if key in _TREES:
        return _TREES[key][1]
    module_name = _module_name(adapter)
    roots = {path.parts[0].removesuffix(".py") for path in data}
    reserved = set(sys.stdlib_module_names) | set(sys.builtin_module_names)
    conflicts = roots & (set(sys.modules) | reserved | _IMPORT_ROOTS)
    if conflicts:
        raise ValueError(
            "Captured model source import conflict with existing modules: "
            + ", ".join(sorted(conflicts))
            + ". Rename the conflicting module or use a fresh worker process."
        )
    temporary = tempfile.TemporaryDirectory(prefix="physicsnemo-model-source-")
    directory = Path(temporary.name).resolve()
    try:
        for relative, content in data.items():
            destination = directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        sys.path.insert(0, str(directory))
        importlib.invalidate_caches()
        _verify_origins(directory, data)
        module = importlib.import_module(module_name)
        _verify_origins(directory, data)
    except BaseException:
        if str(directory) in sys.path:
            sys.path.remove(str(directory))
        for name, module in list(sys.modules.items()):
            if name.partition(".")[0] in roots or _inside(module, directory):
                del sys.modules[name]
        temporary.cleanup()
        raise
    _TREES[key] = (temporary, module, data)
    _IMPORT_ROOTS.update(roots)
    return module


def _call(name, config, assets):
    adapter, data, model_config, user_assets = _source_data(config, assets)
    module = _load(adapter, data)
    verify_imports()
    callback = getattr(module, name, None)
    if not callable(callback):
        raise ValueError(f"Model adapter {adapter} must define {name}(config, assets)")
    try:
        return callback(copy.deepcopy(model_config), dict(user_assets))
    finally:
        verify_imports()


def create_model(config, assets):
    """Construct the customer model from retained source and original inputs."""
    return _call("create_model", config, assets)


def create_cases(config, assets):
    """Construct customer verification cases using the same captured imports."""
    return _call("create_cases", config, assets)


def export_options(config, assets, context):
    """Forward the optional hook from the same retained customer source tree."""
    from model_builder.export.options import ExportOptions

    adapter, data, _, _ = _source_data(config, assets)
    module = _load(adapter, data)
    verify_imports()
    callback = getattr(module, "export_options", None)
    if callback is None:
        return ExportOptions()
    if not callable(callback):
        raise ValueError(f"Model adapter {adapter} export_options must be callable")
    try:
        return callback(context)
    finally:
        verify_imports()


def verify_imports():
    """Fail if live imported Python source differs from its captured identity.

    Callers also run this after eager/export execution and before release because
    model forward methods can modify source after the adapter callbacks return.
    Python's generated bytecode caches are excluded from the source comparison.
    """
    for temporary, _, expected in _TREES.values():
        directory = Path(temporary.name)
        actual = {}

        def visit(current):
            if current.is_symlink() or not current.is_dir():
                raise ValueError(
                    f"Captured model source changed during execution: {current}"
                )
            for child in sorted(current.iterdir()):
                if child.name == "__pycache__":
                    continue
                if child.is_symlink():
                    raise ValueError(
                        f"Captured model source changed during execution: {child}"
                    )
                if child.is_dir():
                    visit(child)
                elif child.suffix == ".py":
                    actual[child.relative_to(directory)] = child.read_bytes()

        try:
            visit(directory)
        except OSError as exc:
            raise ValueError(
                f"Captured model source changed during execution: {exc}"
            ) from exc
        changed = {
            path
            for path in expected.keys() | actual.keys()
            if expected.get(path) != actual.get(path)
        }
        if changed:
            raise ValueError(
                "Captured model source changed during execution: "
                + ", ".join(str(path) for path in sorted(changed))
            )
        _verify_origins(directory, expected)
