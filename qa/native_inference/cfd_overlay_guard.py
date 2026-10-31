"""Extend the native NGC guard for exactly one pinned CFD provider addition."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


PROVIDER = "nvidia-physicsnemo-cfd"
PROVIDER_COMMIT = "921f14dc2ac14c04aabffaba3290deb792379dd8"
OVERLAY_VERSIONS = {
    PROVIDER: "0.0.2",
    "pyvista": "0.45.2",
    "torch-geometric": "2.8.0.post1",
    "trimesh": "4.6.9",
    "fast-simplification": "0.1.12",
    "scooby": "0.12.0",
}


def load_native_guard(path: Path):
    spec = importlib.util.spec_from_file_location(
        "native_builder_environment_guard", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_overlay(before, after, contract, *, native_guard):
    snapshot = copy.deepcopy(after)
    if PROVIDER in before.get("protected", {}):
        raise ValueError("CFD provider must be a new explicitly qualified overlay")
    added = snapshot["protected"].pop(PROVIDER, None)
    if added is None or added.get("version") != OVERLAY_VERSIONS[PROVIDER]:
        raise ValueError("pinned CFD provider protected identity is missing or changed")
    # Reuse the original guard without relaxing a single base package/hash or
    # allowing another NVIDIA distribution to appear in the protected set.
    report = native_guard.verify_environment(before, snapshot, contract)
    packages = after["packages"]
    for name, version in OVERLAY_VERSIONS.items():
        if packages.get(name, {}).get("version") != version:
            raise ValueError(f"pinned provider overlay differs: {name}=={version}")
    report["provider_overlay_versions"] = OVERLAY_VERSIONS
    report["provider_protected_identity"] = added
    return report


def verify_provider_source():
    module = importlib.import_module("physicsnemo.cfd")
    distribution = metadata.distribution(PROVIDER)
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (
        getattr(module, "__version__", None) != "0.0.2"
        or direct.get("vcs_info", {}).get("commit_id") != PROVIDER_COMMIT
    ):
        raise ValueError(
            "installed CFD source version/commit differs from the pinned provider"
        )


def verify_overlay_dependencies(snapshot):
    import vtkmodules.all as vtk

    for name in OVERLAY_VERSIONS:
        for text in snapshot["packages"][name]["requires"]:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(requirement.name)
            # VTK is provided by the same distro Python/C++9.1 pair; installing
            # a wheel here would silently change the numerical geometry oracle.
            actual = (
                vtk.vtkVersion.GetVTKVersion()
                if dependency == "vtk"
                else snapshot["packages"].get(dependency, {}).get("version")
            )
            if actual is None or not requirement.specifier.contains(
                actual, prereleases=True
            ):
                raise ValueError(
                    f"unsatisfied provider overlay dependency: {name} -> {text}; got {actual}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-guard",
        type=Path,
        default=Path("/opt/physicsnemo/image/verify_builder_environment.py"),
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    guard = load_native_guard(args.native_guard)
    after = guard.capture_environment()
    report = verify_overlay(
        json.loads(args.verify.read_text()),
        after,
        json.loads(args.contract.read_text()),
        native_guard=guard,
    )
    verify_provider_source()
    verify_overlay_dependencies(after)
    report["provider_commit"] = PROVIDER_COMMIT
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Verified pinned CFD overlay; every protected base stack file is unchanged.")


if __name__ == "__main__":
    main()
