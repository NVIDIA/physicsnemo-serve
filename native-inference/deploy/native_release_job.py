"""Finite Lepton GPU compilation driver baked into a content-bound input image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from native_release import (
    compilation_key,
    content_id,
    STACK_PROBE,
    file_identity,
    model_input_identity,
    read_json,
    require,
    verify_completed_build,
    verify_inventory,
    write_json,
)
from native_release_artifacts import seal

BEGIN = "NATIVE_RELEASE_BUILD_SUMMARY_BEGIN"
END = "NATIVE_RELEASE_BUILD_SUMMARY_END"


def relocate_projects(resolved, input_root):
    for alias, model in resolved["model_aliases"].items():
        model["project"] = str(
            Path(input_root) / "projects" / alias / Path(model["project"]).name
        )


def qualify_stack(request, backend="aoti"):
    probe = subprocess.run(
        [sys.executable, "-c", STACK_PROBE],
        capture_output=True,
        text=True,
        check=True,
        timeout=1200,
        env={**os.environ, "PNMS_RELEASE_BACKEND": backend},
    )
    values = [line for line in probe.stdout.splitlines() if line.startswith("{")]
    require(
        values and content_id(json.loads(values[-1])) == request["stack_fingerprint"],
        "compiler image native stack differs from the selected immutable stack",
    )


def environment(target, backend="aoti"):
    import torch

    require(torch.cuda.is_available(), "compilation requires a CUDA GPU")
    props = torch.cuda.get_device_properties(0)
    actual = {
        "gpu_name": props.name,
        "compute_capability": [props.major, props.minor],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if backend == "tensorrt":
        import tensorrt

        actual["tensorrt_version"] = tensorrt.__version__
    require(
        actual["gpu_name"] == target["compilation"]["gpu_product"]
        and actual["compute_capability"] == target["compilation"]["compute_capability"],
        "builder GPU product or architecture differs",
    )
    query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=driver_version,mig.mode.current,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        .stdout.strip()
        .splitlines()
    )
    require(len(query) == 1, "builder must select exactly one GPU")
    driver, mig, free = [value.strip() for value in query[0].split(",")]
    require(mig in {"Disabled", "[N/A]", "N/A"}, "MIG is not admitted for compilation")

    def version(value):
        return tuple(int(part) for part in value.split("."))

    requirements = target["deployment"]["driver_requirements"]
    require(
        version(driver) >= version(requirements["minimum_driver_version"])
        and (
            not requirements.get("maximum_driver_version")
            or version(driver) <= version(requirements["maximum_driver_version"])
        ),
        "builder driver is outside qualified stack bounds",
    )
    actual.update(driver_version=driver, mig=mig, free_memory_mb=int(free))
    require(
        actual["free_memory_mb"] >= target["deployment"]["minimum_available_memory_mb"],
        "builder GPU has insufficient available memory",
    )
    return actual


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    request = read_json(args.input_root / "request.json")
    aliases = request["build_aliases"]
    expected = ["preflight", *["compile." + name for name in sorted(aliases)], "seal"]
    summary = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "source_sha": request["resolved"]["source_sha"],
        "image_digest": args.image,
        "expected_cases": expected,
        "cases": [],
    }
    resolved = request["resolved"]
    resolved["paths"]["repository_root"] = "/opt/native-release/source"
    relocate_projects(resolved, args.input_root)
    handoff = root / "handoff"
    handoff.mkdir()

    def case(name, function):
        start = time.monotonic()
        record = {"name": name, "status": "running"}
        summary["cases"].append(record)
        write_json(root / "summary.json", summary)
        try:
            value = function()
            record["status"] = "passed"
            return value
        except BaseException as error:
            record.update(
                status="failed", error=type(error).__name__ + ": " + str(error)
            )
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            write_json(root / "summary.json", summary)

    try:

        def preflight():
            require(
                os.environ.get("SOURCE_SHA") == resolved["source_sha"],
                "compiler image source differs",
            )
            qualify_stack(request, backend=resolved["backend"])
            return environment(resolved["target"], backend=resolved["backend"])

        actual = case("preflight", preflight)
        require(
            request["builder_image"] == resolved["images"]["builder"],
            "selected builder differs from input provenance",
        )
        keys = {}
        for alias in sorted(aliases):
            model = resolved["model_aliases"][alias]
            verify_inventory(Path(model["project"]).parent, model["project_files"])
            key = compilation_key(resolved, alias)
            require(
                key == aliases[alias]["compilation_key"],
                "captured compilation inputs changed",
            )
            keys[alias] = key
            candidate = handoff / "aliases" / alias
            candidate.parent.mkdir(exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "model_builder.build.cli",
                "build",
                model["project"],
                "--backend",
                resolved["backend"],
                "--executor",
                "local",
                "--device",
                "cuda",
                "--runtime",
                args.runtime,
                "--required-gpu-arch",
                resolved["target"]["compilation"]["cuda_architecture"],
                "--output",
                str(candidate),
                "--json",
                "--update-lock",
            ]
            write_json(root / ("command-" + alias + ".json"), {"argv": command})

            def build():
                with (root / ("build-" + alias + ".log")).open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.stage_timeout,
                        env={
                            **os.environ,
                            "PYTHONPATH": "/opt/native-release/source/native-inference/model-builder/src",
                            "PYTHONDONTWRITEBYTECODE": "1",
                        },
                    )
                write_json(
                    candidate / "release-provenance.json",
                    {
                        "schema_version": 1,
                        "source_sha": resolved["source_sha"],
                        "job_image": args.image,
                        "builder_stack_image": request["builder_image"],
                        "compilation_key": key,
                    },
                )
                return verify_completed_build(
                    candidate,
                    backend=resolved["backend"],
                    expected_identity=model_input_identity(resolved, alias),
                    target=resolved["target"]["compilation"],
                )

            case("compile." + alias, build)
        write_json(
            handoff / "build-provenance.json",
            {
                "schema_version": 1,
                "source_sha": resolved["source_sha"],
                "builder_image": request["builder_image"],
                "job_image": args.image,
                "compilation_keys": keys,
                "environment": actual,
                "native_stack_fingerprint": request["stack_fingerprint"],
                "request": file_identity(args.input_root / "request.json"),
            },
        )
        manifest = case(
            "seal",
            lambda: seal(
                handoff,
                root / "bundle.tar",
                run_id=args.run_id,
                image_digest=args.image,
                source_sha=resolved["source_sha"],
                builder_image=request["builder_image"],
                compilation_keys=keys,
            ),
        )
        write_json(root / "manifest.json", manifest)
        summary.update(
            status="passed",
            artifact_manifest=file_identity(root / "manifest.json"),
            compilation_keys=keys,
        )
    except BaseException as error:
        summary.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(root / "summary.json", summary)
        print(BEGIN, flush=True)
        print(json.dumps(summary, sort_keys=True), flush=True)
        print(END, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=Path("/opt/native-release/input")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--runtime", default="/opt/physicsnemo-inference/bin/physicsnemo-infer"
    )
    parser.add_argument("--stage-timeout", type=int, default=7000)
    args = parser.parse_args()
    require(0 < args.stage_timeout <= 7140, "bounded compilation timeout required")
    run(args)


if __name__ == "__main__":
    main()
