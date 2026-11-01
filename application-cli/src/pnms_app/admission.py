"""Admit direct inference on the deployment's current GPU allocation."""

import os
from pathlib import Path
import subprocess
import sys

from .common import contained, require
from .service import _gpu_binding


def local_environment(project, built):
    binding = _gpu_binding(project, built)
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + binding,
            "--query-gpu=name,compute_cap,driver_version,mig.mode.current,uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = probe.stdout.strip().splitlines()
    require(
        probe.returncode == 0 and len(rows) == 1,
        "Cannot verify the current GPU target; check nvidia-smi and the allocation",
    )
    fields = [item.strip() for item in rows[0].split(",")]
    require(len(fields) == 5, "Invalid GPU target probe response")
    name, capability, driver, mig, _ = fields
    target = built["environment"]["gpu"]
    require(
        name == target["name"]
        and "sm" + capability.replace(".", "") == target["architecture"],
        f"GPU target differs from the compiled build ({target['name']} / {target['architecture']}); rebuild on this target",
    )
    require(
        driver == target.get("driver_version"),
        "GPU driver differs from the compiled build; rebuild and validate on this target",
    )
    inactive = {"disabled", "n/a", "[n/a]", "not supported", "[not supported]"}
    require(
        mig.lower() in inactive
        and target.get("mig_mode", "disabled").lower() in inactive,
        "MIG GPU targets are not supported by this build",
    )
    return {**built.get("loader_environment", {}), "CUDA_VISIBLE_DEVICES": binding}


def verify_providers(built, *, environment, output):
    from . import compatibility, native

    location = (Path(built["root"]) / built["bundle"]).resolve()
    require(
        not Path(output).resolve().is_relative_to(location),
        "runtime provider evidence must be outside the sealed bundle",
    )
    selected = native.loader_environment(
        extra=[location / "workflow/lib", location / "workflow/lib64"]
    )
    selected.update(environment)
    loader = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    selected[loader] = os.pathsep.join(
        filter(
            None,
            [
                str(location / "workflow/lib"),
                str(location / "workflow/lib64"),
                selected.get(loader),
            ],
        )
    )
    return compatibility.verify(
        built.get("workflow", {}).get("compatibility"),
        python=Path(built["environment"]["python"]),
        library=contained(
            location,
            "workflow/lib/"
            + built.get("workflow", {}).get(
                "library_name", "libphysicsnemo_workflow.so"
            ),
        ),
        environment=selected,
        output=output,
    )
