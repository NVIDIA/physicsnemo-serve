"""Content-addressed, immutable installations of the native workflow compiler."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import uuid

from .common import (
    contained,
    digest,
    file_identity,
    inventory,
    read_json,
    require,
    write_json,
)


_BUILD_VARIABLES = (
    "CC",
    "CXX",
    "CUDACXX",
    "CUDAHOSTCXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "CUDAFLAGS",
    "LDFLAGS",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_OSX_ARCHITECTURES",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "CUDA_HOME",
    "CUDAToolkit_ROOT",
    "TENSORRT_ROOT",
)


def _headers(root, identities=None):
    """Include files can have any name and link outside their include tree."""
    root = Path(root).resolve()
    if not root.is_dir():
        return None
    identities = {} if identities is None else identities
    visited, records = set(), []
    for directory, children, files in os.walk(root, followlinks=True):
        directory = Path(directory)
        resolved = directory.resolve()
        if resolved in visited:
            children[:] = []
            continue
        visited.add(resolved)
        children.sort()
        for name in sorted(files):
            path = directory / name
            if not path.is_file():
                continue
            target = path.resolve()
            if target not in identities:
                identities[target] = file_identity(target)
            records.append(
                {"path": path.relative_to(root).as_posix(), **identities[target]}
            )
    return sorted(records, key=lambda item: item["path"])


def prepare_configuration(build):
    query = Path(build) / ".cmake/api/v1/query"
    query.mkdir(parents=True, exist_ok=True)
    for name in ("codemodel-v2", "cmakeFiles-v1", "toolchains-v1"):
        (query / name).touch()


def _command_tokens(command):
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    if os.name == "nt":
        # CMake's Windows fragments use native paths, not POSIX escapes.
        lexer.escape = ""
        lexer.quotes = '"'
    return list(lexer)


def _link_inputs(target, toolchains, build):
    """Hash external objects and archives, which runtime loader probes cannot see."""
    tokens = []
    for item in target.get("link", {}).get("commandFragments", []):
        for token in _command_tokens(item["fragment"]):
            tokens.extend(token[4:].split(",") if token.startswith("-Wl,") else [token])
    directories = [
        Path(directory)
        for chain in toolchains
        for directory in chain.get("compiler", {})
        .get("implicit", {})
        .get("linkDirectories", [])
    ]
    files, responses = {}, set()
    offset = 0
    while offset < len(tokens):
        token = tokens[offset]
        offset += 1
        if token.startswith("@"):
            path = Path(token[1:])
            path = path if path.is_absolute() else build / path
            require(path.is_file(), f"unresolved workflow link response file: {path}")
            if path not in responses:
                responses.add(path)
                files[str(path)] = file_identity(path)
                tokens[offset:offset] = _command_tokens(path.read_text())
        elif token == "-L" and offset < len(tokens):
            directories.insert(0, Path(tokens[offset]))
            offset += 1
        elif token.startswith("-L"):
            directories.insert(0, Path(token[2:]))
    for token in tokens:
        if token.startswith("-l") and len(token) > 2:
            name = token[2:]
            names = (
                [name[1:]]
                if name.startswith(":")
                else [
                    f"lib{name}{suffix}" for suffix in (".so", ".dylib", ".a", ".tbd")
                ]
            )
            candidates = [
                (directory if directory.is_absolute() else build / directory) / name
                for directory in directories
                for name in names
            ]
            present = [path for path in candidates if path.is_file()]
            require(
                present,
                f"cannot identify workflow link dependency {token}; use an absolute CMake library path",
            )
            # Hash every viable candidate so static/dynamic selection flags cannot
            # silently hide a changed archive or an alternate linker provider.
            files.update({str(path): file_identity(path) for path in present})
        elif not token.startswith(("-", "@")):
            path = Path(token)
            path = path if path.is_absolute() else build / path
            if path.is_file():
                files[str(path)] = file_identity(path)
    return files


def configured_inputs(build, baseline=None):
    """Read actual CMake discovery, including implicit compiler include paths."""
    build = Path(build).resolve()
    reply = build / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    require(indexes, "CMake did not provide workflow dependency metadata")
    objects = {
        item["kind"]: read_json(reply / item["jsonFile"])
        for item in read_json(indexes[-1])["objects"]
    }
    require(
        {"codemodel", "cmakeFiles", "toolchains"} <= set(objects),
        "CMake workflow dependency metadata is incomplete",
    )
    source = Path(objects["codemodel"]["paths"]["source"])
    roots, identities, files, targets, generated = set(), {}, {}, [], set()
    for chain in objects["toolchains"]["toolchains"]:
        roots.update(
            chain.get("compiler", {}).get("implicit", {}).get("includeDirectories", [])
        )
    for config in objects["codemodel"]["configurations"]:
        for reference in config.get("targets", []):
            target = read_json(reply / reference["jsonFile"])
            groups = target.get("compileGroups", [])
            for group in groups:
                roots.update(item["path"] for item in group.get("includes", []))
            targets.append(
                {
                    "compile": groups,
                    "link": target.get("link", {}),
                    "archive": target.get("archive", {}),
                }
            )
            for item in target.get("sources", []):
                path = Path(item["path"])
                if not path.is_absolute():
                    path = (build if item.get("isGenerated") else source) / path
                if item.get("isGenerated"):
                    generated.add(path.resolve())
                    continue
                if path.is_file():
                    files[str(path)] = file_identity(path)
            files.update(
                _link_inputs(target, objects["toolchains"]["toolchains"], build)
            )
    for item in objects["cmakeFiles"]["inputs"]:
        if item.get("isGenerated"):
            continue
        path = Path(item["path"])
        if not path.is_absolute():
            path = source / path
        require(path.is_file(), f"CMake input disappeared: {path}")
        files[str(path)] = file_identity(path)
    headers = {}
    for root in sorted(roots):
        records = _headers(root, identities) or []
        records = [
            item
            for item in records
            if (Path(root) / item["path"]).resolve() not in generated
        ]
        if baseline is not None and Path(root).resolve().is_relative_to(build):
            # Fresh configure-time fetched sources remain inputs. Outputs first
            # created by compilation are sealed artifacts, not mutable inputs.
            names = {item["path"] for item in baseline["headers"].get(str(root), [])}
            records = [item for item in records if item["path"] in names]
        headers[str(root)] = records
    if baseline is not None:
        files = {
            path: value
            for path, value in files.items()
            if not Path(path).resolve().is_relative_to(build)
            or path in baseline["files"]
        }
    return {
        "targets": targets,
        "toolchains": objects["toolchains"]["toolchains"],
        "files": files,
        "headers": headers,
    }


def _files(path):
    path = Path(path)
    require(path.exists(), f"workflow cache dependency is missing: {path}")
    require(
        not path.is_symlink(),
        f"workflow cache dependency must not be a symlink: {path}",
    )
    return inventory(path, source=True) if path.is_dir() else [file_identity(path)]


def _source(root):
    return {
        name: _files(root / name)
        for name in ("CMakeLists.txt", "include", "src", "cmake")
        if (root / name).exists()
    }


def _toolchain(environment, toolchains):
    executable = shutil.which("cmake", path=environment.get("PATH"))
    require(executable, "workflow compiler tool is unavailable: cmake")
    completed = subprocess.run(
        ["cmake", "--version"],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=True,
    )
    result = {
        "cmake": {
            "executable": file_identity(Path(executable).resolve()),
            "version": completed.stdout.strip(),
        },
        "compilers": {},
        "commands": {},
    }
    for chain in toolchains:
        compiler = chain["compiler"]
        path = Path(compiler["path"]).resolve()
        result["compilers"][chain["language"]] = {
            "path": str(path),
            "executable": file_identity(path),
            "id": compiler.get("id"),
            "version": compiler.get("version"),
        }
    for name in ("CC", "CXX", "CUDACXX", "CUDAHOSTCXX"):
        command = environment.get(name)
        if not command:
            continue
        arguments = [command] if Path(command).is_file() else _command_tokens(command)
        executable = shutil.which(arguments[0], path=environment.get("PATH"))
        require(executable, f"workflow compiler tool is unavailable: {command}")
        result["commands"][name] = {
            "path": str(Path(executable).resolve()),
            "executable": file_identity(Path(executable).resolve()),
            "arguments": arguments[1:],
        }
    return result


def environment_identity(python, sdk_prefix, environment, build_environment):
    """Reuse the selected model-stage probe, or fingerprint a direct SDK caller."""
    if build_environment is not None:
        require(
            isinstance(build_environment.get("stack"), dict),
            "workflow build environment needs captured stack provenance",
        )
        return {
            "stack": build_environment["stack"],
            "architecture": build_environment.get("gpu", {}).get("architecture"),
        }
    from .compatibility import PROBE

    runtime = (
        Path(sdk_prefix)
        / "bin"
        / ("physicsnemo-infer.exe" if os.name == "nt" else "physicsnemo-infer")
    )
    result = subprocess.run(
        [str(python), "-c", PROBE, str(runtime), "--preserve-loader-order"],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=True,
    )
    providers = json.loads(result.stdout)
    require(
        providers.get("schema_version") == 1,
        "workflow provider probe did not return a fingerprint",
    )
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util,json,pathlib; "
            "s=importlib.util.find_spec('torch'); "
            "print(json.dumps(str(pathlib.Path(s.origin).parent) if s and s.origin else None))",
        ],
        env=dict(environment),
        capture_output=True,
        text=True,
        check=True,
    )
    torch_root = json.loads(result.stdout)
    return {
        "providers": providers,
        "torch_development": {
            name: _files(Path(torch_root) / name)
            for name in ("include", "share/cmake")
            if torch_root and (Path(torch_root) / name).exists()
        },
    }


def _normalize(value, substitutions):
    if isinstance(value, str):
        for path, label in sorted(
            substitutions, key=lambda item: len(item[0]), reverse=True
        ):
            value = value.replace(path, label)
            if os.name == "nt":
                value = value.replace(path.replace("\\", "/"), label)
        return value
    if isinstance(value, dict):
        result = {
            _normalize(key, substitutions): _normalize(item, substitutions)
            for key, item in value.items()
        }
        require(
            len(result) == len(value),
            "workflow dependency path normalization conflicts",
        )
        return result
    if isinstance(value, list):
        return [_normalize(item, substitutions) for item in value]
    return value


def fingerprint(
    project,
    *,
    repository,
    sdk_prefix,
    python,
    backend,
    custom_source,
    custom_target,
    tensorrt_root,
    environment,
    configured,
    configured_snapshot=None,
    environment_overrides=None,
    build_environment=None,
):
    project_root = Path(project.get("root", ".")).resolve()
    workflow = project.get("workflow", {})
    dependencies = workflow.get("dependencies", [])
    require(
        isinstance(dependencies, list)
        and all(isinstance(name, str) for name in dependencies),
        "workflow.dependencies must list project-relative files or directories",
    )
    configured_snapshot = (
        configured_snapshot
        if configured_snapshot is not None
        else configured_inputs(configured)
    )
    compiler = _toolchain(environment, configured_snapshot["toolchains"])
    variables = {
        key: environment[key] for key in _BUILD_VARIABLES if key in environment
    }
    variables.update(environment_overrides or {})
    toolchain_file = environment.get("CMAKE_TOOLCHAIN_FILE")
    if toolchain_file and not Path(toolchain_file).expanduser().is_absolute():
        # CMake searches the build and source trees, then records its selection.
        toolchain_file = next(
            (
                line.split("=", 1)[1]
                for line in (Path(configured) / "CMakeCache.txt")
                .read_text()
                .splitlines()
                if line.startswith("CMAKE_TOOLCHAIN_FILE:FILEPATH=")
            ),
            None,
        )
        require(
            toolchain_file and Path(toolchain_file).is_absolute(),
            "CMake did not record the workflow toolchain path",
        )
    toolkit = environment.get("CUDAToolkit_ROOT") or environment.get("CUDA_HOME")
    sdk = Path(sdk_prefix).resolve()
    trt = Path(tensorrt_root or environment.get("TENSORRT_ROOT", "/nonexistent"))
    inputs = {
        "schema_version": 1,
        "backend": backend,
        "system": platform.system(),
        "machine": platform.machine(),
        "python_executable": file_identity(Path(python).resolve()),
        "compiler_adapter": {
            name: file_identity(Path(__file__).with_name(name))
            for name in ("native.py", "workflow_cache.py")
        },
        "workflow_source": _source(
            Path(repository) / "native-inference/workflows/runtime"
        ),
        "sdk_source": _source(Path(repository) / "native-inference/cpp-runtime"),
        "sdk_install": inventory(sdk),
        "custom": {"target": custom_target, "source": _files(custom_source)}
        if custom_source
        else None,
        "dependencies": {
            name: _files(contained(project_root, name)) for name in dependencies
        },
        "toolchain": compiler,
        "configured": configured_snapshot,
        "environment": variables,
        "toolchain_file": _files(Path(toolchain_file).expanduser())
        if toolchain_file
        else None,
        "cuda_headers": _headers(Path(toolkit) / "include")
        if toolkit and (Path(toolkit) / "include").is_dir()
        else None,
        "tensorrt_headers": {
            str(path.relative_to(trt)): file_identity(path)
            for path in sorted((trt / "include").rglob("Nv*.h*"))
            if path.is_file()
        }
        if backend == "tensorrt"
        else None,
        "providers": environment_identity(python, sdk, environment, build_environment),
    }
    # Captured application inputs have a new root for every attempt. Their bytes
    # and logical relative names, rather than that temporary path, determine reuse.
    substitutions = [
        (str(path.resolve()), label)
        for path, label in (
            (Path(configured), "<build>"),
            (project_root, "<project>"),
            (Path(project.get("_state_root", project_root)), "<project>"),
            (sdk, "<sdk>"),
            (Path(repository), "<repository>"),
        )
    ]

    inputs = _normalize(inputs, substitutions)
    return {"key": digest(inputs), "inputs": inputs}


def _installed_inventory(root):
    records = inventory(root)
    for record in records:
        path = Path(root) / record["path"]
        record["mode"] = stat.S_IMODE(path.lstat().st_mode)
    return records


def _verified(entry, identity):
    try:
        require(not entry.is_symlink(), "entry is a symlink")
        seal = read_json(entry / "seal.json")
        require(isinstance(seal, dict), "seal is not an object")
        require(
            seal.get("schema_version") == 1
            and seal.get("key") == identity["key"]
            and seal.get("inputs") == identity["inputs"]
            and digest(seal["inputs"]) == seal["key"],
            "identity differs",
        )
        require(
            set(path.name for path in entry.iterdir()) == {"install", "seal.json"},
            "entry is incomplete or has extra files",
        )
        require(
            seal.get("inventory")
            and not (entry / "install").is_symlink()
            and _installed_inventory(entry / "install") == seal["inventory"],
            "installed inventory differs",
        )
        require(
            isinstance(seal.get("compatibility"), dict),
            "provider provenance is missing",
        )
        return seal
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"workflow cache is corrupt or incomplete: {entry}; remove this cache entry and rebuild ({error})"
        ) from error


def restore(cache_root, identity, install):
    entry = Path(cache_root) / identity["key"]
    if not entry.exists() and not entry.is_symlink():
        return None
    seal = _verified(entry, identity)
    shutil.copytree(entry / "install", install, symlinks=True)
    require(
        _installed_inventory(install) == seal["inventory"],
        "workflow cache changed during restore",
    )
    return seal


def publish(cache_root, identity, install, compatibility):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / identity["key"]
    if entry.exists() or entry.is_symlink():
        return _verified(entry, identity)
    staged = cache_root / (".pending-" + uuid.uuid4().hex)
    staged.mkdir()
    try:
        records = _installed_inventory(install)
        require(records, "cannot cache an empty workflow installation")
        shutil.copytree(install, staged / "install", symlinks=True)
        require(
            _installed_inventory(staged / "install") == records,
            "workflow install changed during publication",
        )
        seal = {
            "schema_version": 1,
            **identity,
            "inventory": records,
            "compatibility": compatibility,
        }
        write_json(staged / "seal.json", seal)
        try:
            staged.rename(entry)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # A concurrent completed publisher wins. Never mutate its contents.
            return _verified(entry, identity)
        return seal
    finally:
        if staged.exists():
            shutil.rmtree(staged)
