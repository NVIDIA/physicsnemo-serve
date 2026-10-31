#!/usr/bin/env python3
"""Full pinned run_1 reference/build/persistent-runner qualification on Lepton H100.

Builds static AOTI packages for 60000 and 8095 points, runs the exact provider
benchmark once, runs the persistent native fixture twice, and uses the original
provider evaluator on native physical fields. This does not exercise Serve HTTP.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from qa.native_inference.assets import prepare_assets
from qa.native_inference.cfd_contract import (
    ASSET_PINS,
    CASE_POINT_COUNTS,
    FieldContractError,
    block_plan,
    compare_evidence,
    inference_seed,
    validate_reference_request,
    verify_file,
)
from qa.native_inference.cfd_reference import verify_provider
from qa.native_inference.cfd_startup import (
    stage_provider_reference as stage_startup_fixture,
)
from qa.native_inference.contract import verify_build
from qa.native_inference.transolver import load_assets
from qa.scripts.cfd_parity_contract import (
    compare_reports,
    read_json_object,
    sha256_file,
    write_json_atomic,
)


SUMMARY_BEGIN = "NATIVE_CFD_QA_SUMMARY_BEGIN"
SUMMARY_END = "NATIVE_CFD_QA_SUMMARY_END"
CASES = [
    "preflight",
    "assets",
    "python_reference",
    "model_import",
    "compile60000",
    "compile8095",
    "native_runner",
    "native_fields",
    "shared_evaluator",
    "parity",
]


def resource_envelope(
    *,
    proc_root=Path("/proc"),
    cgroup_root=Path("/sys/fs/cgroup"),
    shm_path=Path("/dev/shm"),
) -> dict:
    """Record actual container limits as well as host-wide memory observations."""
    memory = {}
    path = proc_root / "meminfo"
    if path.is_file():
        for line in path.read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = int(raw.strip().split()[0]) * 1024
    limit, current, version = None, None, "unavailable"
    if (cgroup_root / "memory.max").is_file():
        raw = (cgroup_root / "memory.max").read_text().strip()
        limit = None if raw == "max" else int(raw)
        current_path, version = cgroup_root / "memory.current", "v2"
    else:
        base = cgroup_root / "memory"
        if (base / "memory.limit_in_bytes").is_file():
            limit = int((base / "memory.limit_in_bytes").read_text().strip())
            if limit >= (1 << 60):
                limit = None
            current_path, version = base / "memory.usage_in_bytes", "v1"
        else:
            current_path = None
    if current_path is not None and current_path.is_file():
        current = int(current_path.read_text().strip())
    shm = os.statvfs(shm_path)
    return {
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "host_memory_total_bytes": memory.get("MemTotal"),
        "host_memory_available_bytes": memory.get("MemAvailable"),
        "cgroup_version": version,
        "cgroup_memory_limit_bytes": limit,
        "cgroup_memory_current_bytes": current,
        "shm_total_bytes": shm.f_blocks * shm.f_frsize,
        "shm_free_bytes": shm.f_bavail * shm.f_frsize,
    }


def stage_project(
    template: Path,
    destination: Path,
    points: int,
    *,
    backend="aoti",
    plugin_root=Path("/opt/physicsnemo-inference/lib"),
) -> None:
    if backend not in {"aoti", "tensorrt"}:
        raise FieldContractError("unsupported native CFD backend")
    shutil.copytree(template, destination)
    recipe = read_json_object(destination / "model-build.json")
    recipe["backends"] = [backend]
    if backend == "aoti":
        recipe.pop("tensorrt_profile", None)
        recipe["assets"] = {}
    else:
        if recipe.get("tensorrt_profile") != "layout-order-exact-v2":
            raise FieldContractError(
                "Transolver TensorRT requires layout-order-exact-v2"
            )
        # CFD executes under the independently captured provider policy. The
        # standalone example intentionally retains its IEEE export profile.
        recipe["tensorrt_profile"] = "provider-default-exact-v1"
        if len(recipe.get("assets", {})) != 9:
            raise FieldContractError(
                "Transolver TensorRT requires all nine exact plugins"
            )
        for relative in recipe["assets"].values():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise FieldContractError(
                    "exact plugin asset must remain inside project"
                )
            source = Path(plugin_root) / path.name
            if not source.is_file() or source.stat().st_size == 0:
                raise FieldContractError(
                    f"required TensorRT exact plugin is missing: {source}"
                )
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256_file(source) != sha256_file(target):
                raise FieldContractError("TensorRT exact plugin changed while staging")
    write_json_atomic(destination / "model-build.json", recipe)
    adapter = destination / "adapter.py"
    source, matches = re.subn(
        r"(?m)^POINT_COUNT = 75$", f"POINT_COUNT = {points}", adapter.read_text()
    )
    if matches != 1:
        raise FieldContractError(
            "Transolver template point count no longer matches expected source"
        )
    adapter.write_text(source)
    (destination / "model-build.lock.json").unlink(missing_ok=True)


def resolve_runtime_executable(value: str) -> str:
    """Builder's runtime argument is a file path, rather than a PATH command."""
    found = shutil.which(value)
    if found is None:
        raise FieldContractError(f"native SDK runtime executable is missing: {value}")
    return str(Path(found).resolve(strict=True))


def package_files(package: Path) -> list[dict]:
    records = []
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise FieldContractError("compiled package cannot contain symlinks")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not records or not (package / "model.json").is_file():
        raise FieldContractError("compiled package is missing its manifest")
    return records


class Job:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.summary = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.source_sha,
            "image_digest": args.image,
            "backend": args.backend,
            "profile_id": "physicsnemo-cfd-surface-transolver-run1-v1",
            "status": "running",
            "qualification": "standalone-persistent-runner",
            "serve_http_tested": False,
            "expected_cases": CASES,
            "cases": [],
        }

    def save(self):
        write_json_atomic(self.root / "summary.json", self.summary)

    def case(self, name, function):
        if name not in CASES or any(
            record["name"] == name for record in self.summary["cases"]
        ):
            raise FieldContractError(f"unknown or duplicate case {name}")
        record = {"name": name, "status": "running"}
        self.summary["cases"].append(record)
        self.save()
        start = time.monotonic()
        print(f"native CFD: starting {name}", flush=True)
        try:
            value = function()
            record["status"] = "passed"
            return value
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            self.save()
            print(
                f"native CFD: {name} {record['status']} ({record['seconds']:.1f}s)",
                flush=True,
            )

    def command(self, name, argv):
        log_dir = self.root / "logs"
        log_dir.mkdir(exist_ok=True)
        argv = [str(item) for item in argv]
        write_json_atomic(log_dir / f"{name}.command.json", {"argv": argv})
        stdout, stderr = log_dir / f"{name}.stdout.log", log_dir / f"{name}.stderr.log"
        # GNU time captures this process tree's CPU maximum RSS independently
        # for each phase. These instrumented correctness runs are not benchmarks.
        measured = [
            "/usr/bin/time",
            "--verbose",
            "--output",
            str(log_dir / f"{name}.resources.log"),
            *argv,
        ]
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.Popen(
                measured, cwd=self.root, stdout=out, stderr=err, start_new_session=True
            )
            try:
                code = process.wait(timeout=self.args.command_timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        if code:
            tail = "\n".join(
                path.read_text(errors="replace")[-12000:] for path in (stdout, stderr)
            )
            raise RuntimeError(
                f"{name} exited {code}; logs: {stdout}, {stderr}\n{tail}"
            )
        return stdout.read_text()


def preflight(job, profile, request):
    import torch
    import vtkmodules.all as vtk

    validate_reference_request(profile, request)
    verify_provider(profile)
    identity = read_json_object(job.args.source_metadata)
    if identity.get("source_sha") != job.args.source_sha:
        raise FieldContractError(
            "GPU job image source metadata differs from requested revision"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", job.args.source_sha) or not re.fullmatch(
        r".+@sha256:[0-9a-f]{64}", job.args.image
    ):
        raise FieldContractError("source SHA and immutable job image are required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise FieldContractError("qualify on one Lepton GPU at a time")
    gpu, capability = (
        torch.cuda.get_device_name(0),
        list(torch.cuda.get_device_capability(0)),
    )
    if "H100" not in gpu or capability != [9, 0]:
        raise FieldContractError(
            f"full run_1 requires H100/sm90, got {gpu}/{capability}"
        )
    for executable in (job.args.builder, job.args.runner, "/usr/bin/time"):
        if not shutil.which(executable):
            raise FieldContractError(f"required executable is missing: {executable}")
    job.args.runtime = resolve_runtime_executable(job.args.runtime)
    env = {
        "gpu_name": gpu,
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "resources": resource_envelope(),
    }
    if job.args.backend == "tensorrt":
        import tensorrt

        env["tensorrt_version"] = tensorrt.__version__
    write_json_atomic(job.root / "environment.json", env)
    return env


def assets_and_layout(job):
    import vtkmodules.all as vtk

    assets = load_assets(prepare_assets(job.args.asset_cache))
    for key, source in (("checkpoint", "checkpoint"), ("statistics", "stats")):
        verify_file(assets[source]["path"], ASSET_PINS[key])
    root = job.root / "inputs" / "drivaerml"
    case = root / "run_1"
    case.mkdir(parents=True)
    for source, name in (("vtp", "boundary_1.vtp"), ("stl", "drivaer_1.stl")):
        try:
            os.link(assets[source]["path"], case / name)
        except OSError:
            shutil.copyfile(assets[source]["path"], case / name)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(case / "boundary_1.vtp"))
    reader.Update()
    count = int(reader.GetOutput().GetNumberOfCells())
    if count != CASE_POINT_COUNTS["run_1"]:
        raise FieldContractError(f"pinned case cell count mismatch: {count}")
    write_json_atomic(
        job.root / "resolved-case.json",
        {"point_count": count, "block_points": block_plan(count), "assets": assets},
    )
    del reader
    gc.collect()
    return assets, root


def capture_repeatable_reference(job, profile, common):
    first, second = job.root / "reference", job.root / "reference-repeat"
    for name, output in (
        ("python-reference", first),
        ("python-reference-repeat", second),
    ):
        job.command(
            name,
            [
                sys.executable,
                "-m",
                "qa.scripts.run_native_cfd_reference",
                "capture",
                *common,
                "--output",
                output,
            ],
        )
    result = compare_evidence(profile, first / "fields.json", second / "fields.json")
    reports = compare_reports(
        rest_report=json.loads(
            (first / "benchmark-output/benchmark_results.json").read_text()
        ),
        direct_report=json.loads(
            (second / "benchmark-output/benchmark_results.json").read_text()
        ),
        comparison=profile["comparison"],
    )
    result.update(scientific_reports_compared=True, reports=reports)
    result["passed"] = result["passed"] and reports["status"] == "passed"
    write_json_atomic(job.root / "reference-repeatability.json", result)
    if not result["passed"]:
        raise FieldContractError("independent pinned provider captures did not repeat")
    return first


def build_or_reuse(reuse, output, build, *, verify):
    """Reuse a byte-copied, reverified complete Builder result without compiling."""
    if reuse is None:
        return build()
    verify(Path(reuse))
    shutil.copytree(reuse, output, symlinks=True)
    return verify(Path(output))


def run(args) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    job = Job(args)
    profile, request = read_json_object(args.profile), read_json_object(args.request)
    try:
        env = job.case("preflight", lambda: preflight(job, profile, request))
        assets, input_root = job.case("assets", lambda: assets_and_layout(job))
        common = [
            "--profile",
            args.profile,
            "--request",
            args.request,
            "--input-root",
            input_root,
            "--checkpoint",
            assets["checkpoint"]["path"],
            "--stats",
            assets["stats"]["path"],
            "--image",
            args.image,
            "--source-sha",
            args.source_sha,
        ]
        reference_root = job.case(
            "python_reference",
            lambda: capture_repeatable_reference(job, profile, common),
        )
        template = args.native_root / "examples" / "transolver-surface"
        projects = job.root / "projects"
        projects.mkdir()
        first = projects / "transolver60000"
        stage_project(
            template, first, 60000, backend=args.backend, plugin_root=args.plugin_root
        )
        job.case(
            "model_import",
            lambda: job.command(
                "model-import",
                [
                    args.builder,
                    "import-checkpoint",
                    assets["checkpoint"]["path"],
                    "--project",
                    first,
                    "--output",
                    first / "weights",
                    "--json",
                ],
            ),
        )
        reused = {}
        for item in getattr(args, "reuse_build", []):
            points, separator, path = item.partition("=")
            if (
                not separator
                or points not in {"60000", "8095"}
                or int(points) in reused
            ):
                raise FieldContractError(
                    "reuse-build requires unique60000/8095=PATH entries"
                )
            reused[int(points)] = Path(path)
        packages, build_outputs = [], {}
        for points in (60000, 8095):
            project = projects / f"transolver{points}"
            if points != 60000:
                stage_project(
                    template,
                    project,
                    points,
                    backend=args.backend,
                    plugin_root=args.plugin_root,
                )
                shutil.copytree(first / "weights", project / "weights")
            output = job.root / "build" / f"transolver{points}"

            def build(project=project, output=output):
                job.command(
                    f"build-{points}",
                    [
                        args.builder,
                        "build",
                        project,
                        "--executor",
                        "local",
                        "--device",
                        "cuda:0",
                        "--runtime",
                        args.runtime,
                        "--backend",
                        args.backend,
                        "--output",
                        output,
                        "--json",
                    ],
                )
                return verify_build(output, [args.backend])[args.backend]

            package = Path(
                job.case(
                    f"compile{points}",
                    lambda points=points, output=output, build=build: build_or_reuse(
                        reused.get(points),
                        output,
                        build,
                        verify=lambda path: verify_build(path, [args.backend])[
                            args.backend
                        ],
                    ),
                )
            )
            job.summary["cases"][-1]["model_recompiled"] = points not in reused
            packages.append(package)
            build_outputs[points] = output
        startup_fixture = stage_startup_fixture(
            build_outputs,
            job.root / "startup",
            profile=profile,
            reference_path=reference_root / "fields.json",
            backend=args.backend,
        )
        model_sources = {
            f"points-{points}": (
                read_json_object(output / "release-provenance.json")["source_sha"]
                if (output / "release-provenance.json").is_file()
                else args.source_sha
            )
            for points, output in build_outputs.items()
        }
        receipt = {
            "schema_version": 1,
            "backend": args.backend,
            "model_build_sources": model_sources,
            "qualification_source_sha": args.source_sha,
            "status": "provisional",
            "source_sha": args.source_sha,
            "builder_image": args.image,
            "environment": env,
            "profile_id": profile["profile_id"],
            "point_count": CASE_POINT_COUNTS["run_1"],
            "block_points": block_plan(CASE_POINT_COUNTS["run_1"]),
            "checkpoint": {
                key: ASSET_PINS["checkpoint"][key] for key in ("sha256", "size_bytes")
            },
            "checkpoint_sha256": ASSET_PINS["checkpoint"]["sha256"],
            "stats_sha256": ASSET_PINS["statistics"]["sha256"],
            "packages": [
                {
                    "manifest_sha256": sha256_file(path / "model.json"),
                    "files": package_files(path),
                }
                for path in packages
            ],
            "startup_fixture": {
                "path": "startup/" + startup_fixture["path"],
                "files": [
                    {**item, "path": "startup/" + item["path"]}
                    for item in startup_fixture["files"]
                ],
            },
        }
        provisional = job.root / "provisional-build-receipt.json"
        write_json_atomic(provisional, receipt)
        native = job.root / "native"
        native.mkdir()
        stats = native / "global_stats.json"
        shutil.copyfile(assets["stats"]["path"], stats)
        fixture = job.root / "runner-fixture.json"
        write_json_atomic(
            fixture,
            {
                "backend": args.backend,
                "packages": [str(path) for path in packages],
                "stats_path": str(stats),
                "startup_fixture_path": str(job.root / "startup/startup.json"),
                "block_size": 60000,
                "math_profile": "provider-default-v1",
                "output_dir": str(native),
                "mesh_path": assets["vtp"]["path"],
                "stl_path": assets["stl"]["path"],
                "point_limit": 0,
                "permutation_seed": inference_seed(42, "run_1"),
            },
        )
        job.case(
            "native_runner",
            lambda: job.command("native-runner", [args.runner, fixture]),
        )
        job.case(
            "native_fields",
            lambda: job.command(
                "native-fields",
                [
                    sys.executable,
                    "-m",
                    "qa.scripts.run_native_cfd_reference",
                    "native",
                    "--profile",
                    args.profile,
                    "--metadata",
                    native / "metadata.json",
                    "--receipt",
                    provisional,
                    "--input-dumps",
                    native / "inputs",
                    "--output",
                    native / "fields.json",
                    "--image",
                    args.image,
                    "--source-sha",
                    args.source_sha,
                ],
            ),
        )
        evaluation = job.root / "native-evaluation"
        job.case(
            "shared_evaluator",
            lambda: job.command(
                "shared-evaluator",
                [
                    sys.executable,
                    "-m",
                    "qa.scripts.run_native_cfd_reference",
                    "evaluate",
                    *common,
                    "--fields",
                    native / "fields.json",
                    "--output",
                    evaluation,
                ],
            ),
        )
        job.case(
            "parity",
            lambda: job.command(
                "parity",
                [
                    sys.executable,
                    "-m",
                    "qa.scripts.run_native_cfd_reference",
                    "compare",
                    "--profile",
                    args.profile,
                    "--reference",
                    reference_root / "fields.json",
                    "--candidate",
                    native / "fields.json",
                    "--candidate-evaluation",
                    evaluation / "evaluation.json",
                    "--output",
                    job.root / "comparison.json",
                ],
            ),
        )
        if [item["name"] for item in job.summary["cases"]] != CASES or any(
            item["status"] != "passed" for item in job.summary["cases"]
        ):
            raise FieldContractError("full qualification inventory did not pass")
        job.summary["status"] = "passed"
        job.save()
        receipt.update(
            status="passed",
            qualification="full-run1-fields-and-metrics-v1",
            summary_sha256=sha256_file(job.root / "summary.json"),
        )
        write_json_atomic(job.root / "build-receipt.json", receipt)
        return 0
    except BaseException as error:
        job.summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        job.save()
        raise
    finally:
        print(SUMMARY_BEGIN, flush=True)
        print(json.dumps(job.summary, sort_keys=True, allow_nan=False), flush=True)
        print(SUMMARY_END, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "profile", "request"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--source-sha", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument(
        "--asset-cache", type=Path, default=Path("/outputs/assets/transolver")
    )
    result.add_argument(
        "--native-root", type=Path, default=Path("/opt/physicsnemo/native-inference")
    )
    result.add_argument("--builder", default="pnms-model-builder")
    result.add_argument("--runtime", default="physicsnemo-infer")
    result.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    result.add_argument(
        "--plugin-root", type=Path, default=Path("/opt/physicsnemo-inference/lib")
    )
    result.add_argument("--runner", default="physicsnemo-transolver-runner")
    result.add_argument("--command-timeout", type=int, default=21600)
    result.add_argument("--reuse-build", action="append", default=[])
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/native-cfd-qa/source.json")
    )
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
