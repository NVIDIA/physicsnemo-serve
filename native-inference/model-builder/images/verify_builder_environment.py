"""Check the explicit Geo overlay without allowing replacement of NGC's stack."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def verify_environment(before, after, contract):
    keys = ("python", "system", "machine", "torch_cuda")
    if "torch_module_version" in contract:
        keys += ("torch_module_version",)
    for key in keys:
        if before.get(key) != contract[key] or after.get(key) != contract[key]:
            raise ValueError(f"builder {key} differs from the qualified target")
    if not before.get("protected") or before["protected"] != after.get("protected"):
        raise ValueError("protected NGC stack changed during dependency installation")
    versions = contract["versions"]
    packages = after["packages"]
    for name, version in versions.items():
        if packages.get(name, {}).get("version") != version:
            raise ValueError(f"pinned dependency version differs: {name}=={version}")
    pending = [(name, set()) for name in versions]
    seen = {}
    exceptions = []
    while pending:
        name, extras = pending.pop()
        previous = seen.get(name)
        if previous is not None and extras <= previous:
            continue
        seen[name] = extras | (previous or set())
        for value in packages[name]["requires"]:
            requirement = Requirement(value)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra})
                for extra in seen[name] | {""}
            ):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency not in versions:
                raise ValueError(f"unpinned active dependency: {name} -> {value}")
            actual = packages[dependency]["version"]
            if requirement.specifier and actual not in requirement.specifier:
                exception = {"consumer": name, "requirement": value, "actual": actual}
                if exception not in contract["metadata_exceptions"]:
                    raise ValueError(
                        f"unsatisfied dependency: {name} -> {value}; got {actual}"
                    )
                if exception not in exceptions:
                    exceptions.append(exception)
            pending.append((dependency, set(requirement.extras)))
    return {
        "status": "passed",
        "dependency_versions": dict(versions),
        "qualified_metadata_exceptions": exceptions,
        "protected": after["protected"],
        "target": {key: after[key] for key in keys},
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_environment():
    import torch

    names = sorted(
        {canonicalize_name(d.metadata["Name"]) for d in metadata.distributions()}
    )
    packages = {}
    protected = {}
    for name in names:
        distribution = metadata.distribution(name)
        packages[name] = {
            "version": distribution.version,
            "requires": distribution.requires or [],
        }
        if not (
            name in ("torch", "torchvision", "torchaudio", "triton")
            or name.startswith("tensorrt")
            or (name.startswith("nvidia-") and name != "nvidia-physicsnemo")
        ):
            continue
        files = []
        for member in sorted(distribution.files or [], key=str):
            value = str(member)
            if not (
                value.endswith(("METADATA", "WHEEL", "version.py"))
                or ".so" in Path(value).name
            ):
                continue
            path = Path(distribution.locate_file(member))
            if path.is_file():
                files.append(
                    {
                        "path": value,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        protected[name] = {"version": distribution.version, "files": files}
    return {
        "python": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "system": platform.system(),
        "machine": platform.machine(),
        "torch_cuda": torch.version.cuda,
        "torch_module_version": str(torch.__version__),
        "protected": protected,
        "packages": packages,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text())
    snapshot = capture_environment()
    if args.capture:
        args.capture.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        return 0
    if args.output is None:
        parser.error("--verify requires --output")
    report = verify_environment(json.loads(args.verify.read_text()), snapshot, contract)
    report.update(
        contract_sha256=_sha256(args.contract), guard_sha256=_sha256(Path(__file__))
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Verified {len(report['dependency_versions'])} pinned dependencies; NGC stack unchanged."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
