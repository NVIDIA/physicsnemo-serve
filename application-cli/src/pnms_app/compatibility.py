"""Portable provider fingerprints for already compiled native workflows.

The same standard-library probe runs in the selected build environment and the
release image. No model execution, compilation or GPU access occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .common import require


PROBE = r"""
import hashlib, importlib.metadata, importlib.util, json, os, pathlib, platform, re, subprocess, sys

def require(value, message):
    if not value:
        raise RuntimeError(message)

def identity(path):
    path = pathlib.Path(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return dict(sha256=result.hexdigest(), size_bytes=path.stat().st_size)

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
trt = None
try:
    trt = importlib.metadata.version("tensorrt")
except importlib.metadata.PackageNotFoundError:
    pass
result = dict(schema_version=1, system=platform.system(), machine=platform.machine(),
              libc=platform.libc_ver(), torch=torch.__version__ if torch else None,
              cuda=torch.version.cuda if torch else None,
              cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI) if torch else None,
              tensorrt=trt, torch_libraries={}, system_libraries={}, driver_libraries=[])
environment = os.environ.copy()
preserve_loader_order = len(sys.argv) > 2 and sys.argv[2] == "--preserve-loader-order"
if torch:
    directory = pathlib.Path(torch.__file__).resolve().parent
    library_root = directory / "lib"
    identities = {}
    for path in sorted(library_root.rglob("*")):
        if path.is_file() and (".so" in path.name or path.suffix in (".dll", ".dylib", ".pyd")):
            target = path.resolve()
            if target not in identities:
                identities[target] = identity(target)
            result["torch_libraries"][path.relative_to(directory).as_posix()] = identities[target]
    if not preserve_loader_order:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(library_root), environment.get("LD_LIBRARY_PATH", "")])
library = pathlib.Path(sys.argv[1]).resolve()
require(library.is_file(), "compiled workflow library is missing")
result["workflow_binary"] = identity(library)
if platform.system() == "Linux":
    if not preserve_loader_order:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(library.parent), environment.get("LD_LIBRARY_PATH", "")])
    linked = subprocess.run(["ldd", str(library)], env=environment, capture_output=True, text=True)
    require(linked.returncode == 0, "cannot inspect native workflow dependencies: " + linked.stderr)
    for line in linked.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("linux-vdso"):
            continue
        if "=>" in line:
            name, target = (part.strip() for part in line.split("=>", 1))
            target = re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", target)
        else:
            target = re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line)
            name = pathlib.Path(target).name
        # NVIDIA driver libraries come from the deployment node, and may be
        # absent in an ordinary CPU Docker build. Driver admission is separate.
        if name.startswith(("libcuda.so", "libnvidia-")):
            result["driver_libraries"].append(name)
            continue
        require(target != "not found" and pathlib.Path(target).is_file(), "native provider is unavailable: " + name)
        result["system_libraries"][name] = identity(target)
    require(result["system_libraries"], "native provider closure is empty")
result["driver_libraries"] = sorted(set(result["driver_libraries"]))
print(json.dumps(result, sort_keys=True))
"""


def capture(python: Path, library: Path, *, environment: dict, output: Path) -> dict:
    completed = subprocess.run(
        [str(python), "-c", PROBE, str(library), "--preserve-loader-order"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".log").write_text(completed.stderr or "")
    if completed.returncode:
        raise RuntimeError(
            f"native provider fingerprint failed; diagnostics: {output.with_suffix('.log')}"
        )
    value = json.loads(completed.stdout)
    output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return value


def verify(expected, *, python, library, environment, output):
    require(
        isinstance(expected, dict)
        and expected.get("schema_version") == 1
        and expected.get("workflow_binary")
        and isinstance(expected.get("torch_libraries"), dict)
        and isinstance(expected.get("system_libraries"), dict),
        "native provider fingerprint is missing; run pnms build before host execution",
    )
    try:
        observed = capture(python, library, environment=environment, output=output)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            "Cannot verify native runtime providers; restore the matched venv and "
            f"native libraries, or rebuild with pnms build: {error}"
        ) from error
    differences = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    require(
        not differences,
        "native runtime provider mismatch at "
        + ", ".join(differences)
        + "; restore the matched venv and native libraries, or rebuild with pnms build; "
        + f"observed fingerprint: {output}",
    )
    return observed
