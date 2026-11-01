"""Generated application stack identity; independent of Model Builder locks.

The lock observes the selected environment. It is not a dependency installer.
Only successful application publication may replace the authored pnms.lock.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .common import digest, read_json, require, write_json

_PYTHON_PROBE = r"""
import hashlib, importlib.metadata as metadata, json, platform, re, sys
packages = []
for distribution in metadata.distributions():
    name = distribution.metadata.get('Name')
    if not name:
        continue
    contents = distribution.read_text('METADATA') or distribution.read_text('PKG-INFO') or ''
    packages.append(dict(name=re.sub(r'[-_.]+', '-', name).lower(), version=distribution.version,
                         metadata_sha256=hashlib.sha256(contents.encode()).hexdigest()))
print(json.dumps(dict(version=platform.python_version(), implementation=platform.python_implementation(),
                     cache_tag=sys.implementation.cache_tag, system=platform.system(), machine=platform.machine(),
                     distributions=sorted(packages, key=lambda package: (package['name'], package['version'], package['metadata_sha256'])))))
"""


def python_identity(project):
    """Probe only the selected interpreter and distribution metadata; no ML imports."""
    executable = (
        project.get("python_executable")
        or project.get("build", {}).get("python")
        or sys.executable
    )
    completed = subprocess.run(
        [str(executable), "-c", _PYTHON_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        "Cannot inspect selected Python environment for pnms.lock: "
        + (completed.stderr or "interpreter failed"),
    )
    observed = json.loads(completed.stdout)
    require(
        isinstance(observed, dict)
        and observed.get("version")
        and isinstance(observed.get("distributions"), list),
        "Selected Python returned invalid environment identity for pnms.lock",
    )
    return observed


def _providers(value):
    # The application workflow binary changes when user C++ source changes. It
    # does not belong to the environment lock; providers and SDK bytes do.
    return {
        key: value[key]
        for key in (
            "schema_version",
            "system",
            "machine",
            "libc",
            "torch",
            "cuda",
            "tensorrt",
            "cxx11_abi",
            "torch_libraries",
            "system_libraries",
            "driver_libraries",
        )
        if key in value
    }


def snapshot(project, environment, *, python=None):
    selected = {"kind": "local"}
    image = project.get("_container_image")
    if image:
        selected.update(kind="container", image=image)
    document = {
        "schema_version": 1,
        "runtime": project["runtime"],
        "environment": selected,
    }
    if project["runtime"] == "python":
        selected["python"] = python if python is not None else python_identity(project)
    else:
        stack = environment.get("stack", {})
        normalized = {
            key: stack[key]
            for key in (
                "python",
                "torch",
                "cuda",
                "tensorrt",
                "libc",
                "machine",
                "system",
                "cxx11_abi",
                "dependencies",
                "driver_version",
                "cuda_compiler_version",
                "compiler",
                "runtime_sha256",
            )
            if key in stack
        }
        if "compiler" in normalized:
            normalized["compiler"] = {
                name: "\n".join(
                    line
                    for line in report.splitlines()
                    if not line.strip().startswith("InstalledDir:")
                )
                for name, report in normalized["compiler"].items()
            }
        if "sdk_inventory" in stack:
            # CMake/pkg-config configuration may embed the installation prefix.
            # Headers and actual runtime/library bytes carry the SDK identity.
            normalized["sdk_files"] = {
                name: {
                    key: record[key]
                    for key in ("sha256", "size_bytes")
                    if key in record
                }
                for name, record in stack["sdk_inventory"].items()
                if name.startswith(("bin/", "include/", "lib/", "lib64/"))
                and "/cmake/" not in name
                and "/pkgconfig/" not in name
            }
        if "native_providers" in stack:
            normalized["native_providers"] = _providers(stack["native_providers"])
        gpu = environment.get("gpu", {})
        document["native"] = {
            "backend": project["backend"],
            "stack": normalized,
            "gpu": {
                "policy": "build-product-only",
                **{
                    key: gpu[key]
                    for key in (
                        "name",
                        "major",
                        "minor",
                        "architecture",
                        "mig_mode",
                        "driver_version",
                    )
                    if key in gpu
                },
            },
        }
    # Convert tuple-valued probes to their stable JSON representation before
    # comparison with a lock read back from disk.
    return json.loads(json.dumps(document, sort_keys=True, allow_nan=False))


def capture(project):
    path = Path(project.get("_state_root", project["root"])) / "pnms.lock"
    return path.read_bytes() if path.exists() else None


def _differences(expected, actual, prefix=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                result.append(name)
            else:
                result.extend(_differences(expected[key], actual[key], name))
        return result
    return [] if expected == actual else [prefix]


def preflight(project, environment, *, refresh=False):
    """Reject already observed drift before creating a missing SDK.

    The same probe supplies the final check. SDK/provider fields unavailable
    until preparation finishes are checked by check(), before model compilation.
    """
    path = Path(project.get("_state_root", project["root"])) / "pnms.lock"
    if not path.exists() or refresh:
        return
    candidate = snapshot(project, environment)
    try:
        existing = read_json(path)
    except (ValueError, OSError) as error:
        raise ValueError(
            "Cannot read pnms.lock; inspect it and run pnms build --refresh-lock to replace it after a successful build"
        ) from error

    def selected(expected, observed):
        if isinstance(expected, dict) and isinstance(observed, dict):
            return {
                key: selected(expected[key], value)
                for key, value in observed.items()
                if key in expected
            }
        return expected

    differences = _differences(selected(existing, candidate), candidate)
    require(
        not differences,
        "Selected environment differs from pnms.lock at "
        + ", ".join(differences[:8])
        + "; restore the locked environment or run pnms build --refresh-lock to accept this stack after a successful build",
    )


def check(project, environment, *, refresh=False, python=None):
    candidate = snapshot(project, environment, python=python)
    path = Path(project.get("_state_root", project["root"])) / "pnms.lock"
    if path.exists() and not refresh:
        try:
            existing = read_json(path)
        except (ValueError, OSError) as error:
            raise ValueError(
                "Cannot read pnms.lock; inspect it and run pnms build --refresh-lock to replace it after a successful build"
            ) from error
        differences = _differences(existing, candidate)
        require(
            not differences,
            "Selected environment differs from pnms.lock at "
            + ", ".join(differences[:8])
            + "; restore the locked environment or run pnms build --refresh-lock to accept this stack after a successful build",
        )
    return {"schema_version": 1, "digest": digest(candidate), "snapshot": candidate}


def commit(project, record, *, previous):
    require(
        capture(project) == previous,
        "pnms.lock changed during this build; rerun pnms build against the current lock",
    )
    require(
        record.get("digest") == digest(record["snapshot"]),
        "application environment lock identity differs from its snapshot",
    )
    path = Path(project.get("_state_root", project["root"])) / "pnms.lock"
    try:
        existing = read_json(path) if previous is not None else None
    except ValueError:
        existing = None
    if existing != record["snapshot"]:
        write_json(path, record["snapshot"])
