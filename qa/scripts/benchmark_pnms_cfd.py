#!/usr/bin/env python3
"""Matched client/full-report CFD benchmark sessions and guarded comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qa.native_inference.cfd_contract import (  # noqa: E402
    ASSET_PINS,
    CASE_PINS,
    CASE_POINT_COUNTS,
    MATH_KEYS,
    MODEL_REVISION,
    METRIC_OUTPUTS,
    FieldContractError,
    block_plan,
    inference_seed,
    validate_evidence,
    validate_reference_request,
    validate_report,
    verify_file,
)
from qa.scripts.cfd_parity_contract import (  # noqa: E402
    compare_reports,
    normalize_report,
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from qa.scripts.run_native_cfd_http_job import HTTP as NativeHTTP  # noqa: E402

BOUNDARY = "client-http-to-full-canonical-report-v1"
RUNTIMES = ("python", "aoti", "tensorrt")
SESSION_ORDER = (*RUNTIMES, *reversed(RUNTIMES))
NATIVE_WORKFLOW = "physicsnemo-cfd-transolver-native"
PYTHON_WORKFLOW = "physicsnemo-cfd-surface-benchmark"
SUMMARY_BEGIN = "PNMS_CFD_BENCHMARK_SUMMARY_BEGIN"
SUMMARY_END = "PNMS_CFD_BENCHMARK_SUMMARY_END"
MAX_PYTHON_ARTIFACT_BYTES = 16 * 1024 * 1024
SERVICE_OWNER_ENV = "PNMS_CFD_BENCHMARK_OWNER"


def require(condition, message):
    if not condition:
        raise FieldContractError(message)


class HTTP(NativeHTTP):
    def download_python(self, path, destination):
        """Hash bounded legacy output bytes whose descriptors have no identities."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.base_url + path, timeout=120) as response:
            length = response.headers.get("Content-Length")
            require(
                length is None
                or (length.isdecimal() and int(length) <= MAX_PYTHON_ARTIFACT_BYTES),
                "Python HTTP artifact Content-Length is invalid or exceeds size bound",
            )
            digest, count = hashlib.sha256(), 0
            with destination.open("xb") as output:
                while chunk := response.read(
                    min(1024 * 1024, MAX_PYTHON_ARTIFACT_BYTES - count + 1)
                ):
                    count += len(chunk)
                    require(
                        count <= MAX_PYTHON_ARTIFACT_BYTES,
                        "Python HTTP artifact exceeds size bound",
                    )
                    digest.update(chunk)
                    output.write(chunk)
        require(
            length is None or count == int(length),
            "Python HTTP artifact bytes differ from Content-Length",
        )
        return {"sha256": digest.hexdigest(), "size_bytes": count}


def canonical_bytes(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def identity_bytes(value):
    return {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}


def canonical_report(profile, report):
    """Keep scientific values and their contract, excluding incidental timestamps."""
    validate_report(profile, report)
    structure, metrics = normalize_report(report)
    require(
        len(metrics) == 22, "full run_1 requires eleven case and eleven summary metrics"
    )
    return {
        "structure": structure,
        "metrics": [
            {"key": list(key), "value": value} for key, value in sorted(metrics.items())
        ],
    }


def distribution(values):
    require(
        values
        and all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in values
        ),
        "timings must be finite positive observed values",
    )
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def compare_sessions(sessions):
    """Only matched, qualified sessions may produce client-boundary ratios."""
    require(
        tuple(row.get("runtime") for row in sessions) == SESSION_ORDER,
        "comparison requires six counterbalanced sessions in Python, AOTI, TensorRT, TensorRT, AOTI, Python order",
    )
    first = sessions[0]
    for row in sessions:
        require(
            row.get("schema_version") == 1
            and row.get("status") == "passed"
            and row.get("boundary") == BOUNDARY,
            "only passed sessions with the matched full-report boundary are comparable",
        )
        require(
            re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", row.get("image_digest", ""))
            and re.fullmatch(r"[a-f0-9]{40}", row.get("source_sha", "")),
            "session image/source identity is invalid",
        )
        for section in (
            "workload",
            "hardware",
            "environment",
            "evaluator_environment",
            "numerical_libraries",
        ):
            require(
                isinstance(row.get(section), dict)
                and row[section]
                and row[section] == first.get(section),
                f"sessions do not match: {section}",
            )
        require(
            {"libtorch_cpu.so", "libtorch_cuda.so"}
            <= row["numerical_libraries"].keys(),
            "Torch library observations are incomplete",
        )
        for section in ("environment", "evaluator_environment"):
            require(
                all(
                    isinstance(row[section].get(key), str) and row[section][key]
                    for key in ("python", "torch", "cuda", "vtk")
                )
                and isinstance(row[section].get("math"), dict)
                and set(MATH_KEYS) <= row[section]["math"].keys(),
                "numerical stack/math observations are incomplete",
            )
        require(
            all(
                key in row["hardware"] and row["hardware"][key] is not None
                for key in (
                    "uuid",
                    "name",
                    "driver_version",
                    "compute_capability",
                    "cpu_model",
                    "cpu_affinity_count",
                    "memory_limit_bytes",
                )
            ),
            "hardware observations are incomplete",
        )
        require(
            row["environment"] == row["evaluator_environment"],
            "runtime and evaluator numerical stack/math do not match",
        )
        require(
            "H100" in row["hardware"].get("name", "")
            and row["hardware"].get("compute_capability") == [9, 0]
            and row["hardware"].get("uuid", "").startswith("GPU-"),
            "comparison requires the same observed H100 GPU",
        )
        gates = [
            row.get("prerequisite", {}),
            *row.get("samples", []),
            row.get("postrun", {}),
        ]
        require(
            all(
                gate.get("parity") == "passed"
                and gate.get("artifacts_verified") is True
                for gate in gates
            ),
            "field/report artifact qualification and metric parity must pass before and after timing",
        )
        require(
            [sample.get("label") for sample in row.get("samples", [])]
            == ["first", "warm-1", "warm-2", "warm-3"],
            "each session requires first plus three warm requests",
        )
        hashes = {gate.get("canonical_report_sha256") for gate in gates}
        require(
            len(hashes) == 1
            and re.fullmatch(r"[a-f0-9]{64}", next(iter(hashes), "") or ""),
            "canonical output changed between prerequisite, timing and postrun",
        )
    statistics_by_runtime = {}
    for runtime in RUNTIMES:
        rows = [row for row in sessions if row["runtime"] == runtime]
        statistics_by_runtime[runtime] = {
            "cold_startup_seconds": distribution(
                [row["cold_startup_seconds"] for row in rows]
            ),
            "first_seconds": distribution(
                [row["samples"][0]["seconds"] for row in rows]
            ),
            "warm_seconds": distribution(
                [sample["seconds"] for row in rows for sample in row["samples"][1:]]
            ),
        }
    baseline = statistics_by_runtime["python"]["warm_seconds"]["median"]
    return {
        "schema_version": 1,
        "status": "passed",
        "boundary": BOUNDARY,
        "runtime_statistics": statistics_by_runtime,
        "warm_speedup_vs_python": {
            runtime: baseline / statistics_by_runtime[runtime]["warm_seconds"]["median"]
            for runtime in RUNTIMES[1:]
        },
        "execution_order": [row["runtime"] for row in sessions],
        "hardware": first["hardware"],
        "workload": first["workload"],
        "environment": first["environment"],
        "kernel_throughput_measured": False,
        "tail_latency_estimated": False,
        "limitations": [
            "Full client/report boundary; includes native external shared evaluator and all required artifact downloads.",
            "OS/storage caches and GPU clocks are not reset or controlled; GPU clock telemetry is not recorded.",
            "Warm labels refer to service requests; the unchanged Python provider may load its model in a fresh subprocess for each request.",
        ],
    }


def file_identity(path):
    return {"sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size}


def relative_file(path, root):
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    require(
        path.is_relative_to(root) and path.is_file(),
        "benchmark artifact escaped its session directory",
    )
    return path.relative_to(root).as_posix()


def report_parity(profile, report, reference):
    canonical_report(profile, report)
    comparison = compare_reports(
        rest_report=report, direct_report=reference, comparison=profile["comparison"]
    )
    require(
        comparison["status"] == "passed", "full run_1 scientific metric parity failed"
    )
    return comparison


def hardware():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [row.split(",") for row in result.stdout.strip().splitlines()]
    require(
        len(rows) == 1 and len(rows[0]) == 6,
        "benchmark requires exactly one allocated Lepton H100",
    )
    name, gpu_uuid, driver, bus, memory, architecture = (
        item.strip() for item in rows[0]
    )
    require(
        "H100" in name and architecture == "9.0" and gpu_uuid.startswith("GPU-"),
        "benchmark requires one observed H100 sm90 GPU",
    )
    cpuinfo = Path("/proc/cpuinfo").read_text()
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        "",
    )
    require(cpu, "CPU model observation is unavailable")
    memory_limit = Path("/sys/fs/cgroup/memory.max")
    limit = memory_limit.read_text().strip() if memory_limit.exists() else "unavailable"
    quota = Path("/sys/fs/cgroup/cpu.max")
    return {
        "name": name,
        "uuid": gpu_uuid,
        "driver_version": driver,
        "pci_bus_id": bus,
        "memory_total_mib": int(memory),
        "compute_capability": [9, 0],
        "cpu_model": cpu,
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cpu_quota": quota.read_text().strip() if quota.exists() else "unavailable",
        "memory_limit_bytes": int(limit) if limit.isdecimal() else limit,
    }


def environment_observation(profile):
    from qa.native_inference.cfd_reference import environment, verify_provider
    import torch

    directory = Path(torch.__file__).parent / "lib"
    libraries = {
        path.name: {"path": str(path.resolve()), **file_identity(path)}
        for pattern in ("libtorch*.so", "libc10*.so")
        for path in directory.glob(pattern)
        if path.is_file()
    }
    require(
        {"libtorch_cpu.so", "libtorch_cuda.so"} <= libraries.keys(),
        "selected Python Torch shared libraries are unavailable",
    )
    return {
        "environment": environment(),
        "provider": verify_provider(profile),
        "interpreter": sys.executable,
        "torch_libraries": libraries,
    }


def probe_environment(args):
    write_json_atomic(
        args.output, environment_observation(read_json_object(args.profile))
    )


def prepare(args):
    from qa.native_inference.cfd_reference import reference_model_asset_arguments
    from qa.scripts.run_native_cfd_reference import _verify_inputs
    from qa.native_inference.cfd_benchmark_reference import qualified_reference

    require(
        re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", args.expected_image),
        "benchmark image must be immutable",
    )
    require(
        re.fullmatch(r"[a-f0-9]{40}", args.expected_source_sha),
        "benchmark source must be a full Git SHA",
    )
    require(
        math.isfinite(args.poll_interval) and args.poll_interval > 0,
        "poll interval must be finite and positive",
    )
    selected_artifacts = getattr(args, "service_artifact_root", None)
    artifact_root = selected_artifacts or args.output / "service-artifacts"
    require(
        not artifact_root.exists() and not artifact_root.is_symlink(),
        "service artifact observation root must be fresh and initially absent",
    )
    artifact_root = artifact_root.resolve()
    profile, request = read_json_object(args.profile), read_json_object(args.request)
    validate_reference_request(profile, request)
    reference_report, qualification = qualified_reference(args.reference_fields)
    evidence = read_json_object(args.reference_fields)
    fields = validate_evidence(profile, evidence, args.reference_fields.parent)
    canonical_report(profile, reference_report)
    _verify_inputs(request, args.input_root)
    require(
        (args.checkpoint is None) == (args.stats is None),
        "supply checkpoint and statistics together",
    )
    if args.checkpoint is None:
        assets = reference_model_asset_arguments(args.reference_fields)
        require(len(assets) == 4, "supply local pinned --checkpoint and --stats")
        checkpoint, stats = Path(assets[1]), Path(assets[3])
    else:
        checkpoint, stats = args.checkpoint, args.stats
    for key, path in (("checkpoint", checkpoint), ("statistics", stats)):
        verify_file(path, ASSET_PINS[key])
    command = json.loads(args.service_command_json.read_text())
    require(
        isinstance(command, list)
        and command
        and all(isinstance(value, str) and value for value in command),
        "service command JSON must be a nonempty argv array",
    )
    require(
        shutil.which(command[0]) or Path(command[0]).is_file(),
        "service executable is unavailable",
    )
    source_files = read_json_object(args.service_files_json)
    require(source_files, "service source/file identity inventory must be supplied")
    for path, identity in source_files.items():
        require(Path(path).is_absolute(), "service identity paths must be absolute")
        verify_file(path, identity)
    probe = args.output / "environment.json"
    subprocess.run(
        [
            str(args.environment_python),
            str(Path(__file__).resolve()),
            "environment",
            "--profile",
            str(args.profile.resolve()),
            "--output",
            str(probe),
        ],
        check=True,
        timeout=120,
    )
    observed = read_json_object(probe)
    evaluator = environment_observation(profile)
    require(
        Path(evaluator["interpreter"]).resolve()
        == Path(observed["interpreter"]).resolve(),
        "in-process evaluator interpreter differs from the selected interpreter; launch the benchmark with --environment-python's interpreter",
    )
    for key in ("environment", "provider", "torch_libraries"):
        require(
            evaluator[key] == observed[key],
            "in-process evaluator differs from the selected stack: " + key,
        )
    environment = {
        key: observed["environment"][key]
        for key in ("python", "torch", "cuda", "vtk", "math")
    }
    for key, value in environment.items():
        require(
            value == evidence["environment"].get(key),
            "selected interpreter does not match qualified reference environment: "
            + key,
        )
    build_sources = None
    bundle_identity = None
    if args.runtime != "python":
        require(
            args.native_bundle is not None,
            "native benchmark requires its sealed pnms bundle",
        )
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[2] / "application-cli/src")
        )
        from pnms_app.bundle import verify

        bundle_identity = verify(args.native_bundle)["bundle_digest"]
        descriptor = read_json_object(args.native_bundle / "bundle.json")
        require(
            descriptor.get("adapter") == "transolver_surface"
            and descriptor.get("backend") == args.runtime
            and descriptor.get("source_sha") == args.expected_source_sha,
            "native bundle adapter/backend/source differs",
        )
        build_sources = descriptor["model_build_sources"]
    libraries = observed["torch_libraries"]
    require(
        {"libtorch_cpu.so", "libtorch_cuda.so"} <= libraries.keys(),
        "selected interpreter Torch library observations are incomplete",
    )
    return {
        "profile": profile,
        "request": request,
        "reference_report": reference_report,
        "reference_fields": fields["run_1"],
        "reference_environment": evidence["environment"],
        "environment": environment,
        "interpreter": observed["interpreter"],
        "torch_libraries": libraries,
        "provider": observed["provider"],
        "hardware": hardware(),
        "command": command,
        "service_files": source_files,
        "qualification": qualification,
        "checkpoint": checkpoint,
        "stats": stats,
        "model_build_sources": build_sources,
        "bundle_digest": bundle_identity,
        "runtime": args.runtime,
        "image": args.expected_image,
        "source_sha": args.expected_source_sha,
        "workflow": PYTHON_WORKFLOW if args.runtime == "python" else NATIVE_WORKFLOW,
        "timeout": args.request_timeout,
        "poll_interval": args.poll_interval,
        "artifact_root": artifact_root,
        "input_root": args.input_root,
        "cache_configs": set(),
    }


def verify_native_stack(process, context, *, proc_root=Path("/proc")):
    """Bind the resident C++ process to the selected distribution's actual libraries."""
    from qa.native_inference.pnms_http_job import _descendants

    workers = []
    for pid, start in _descendants(process.pid, proc_root).items():
        executable = proc_root / str(pid) / "exe"
        if (
            executable.exists()
            and Path(os.readlink(executable)).name == "native-worker"
        ):
            workers.append((pid, start))
    require(len(workers) == 1, "expected one owned resident native worker")
    pid, start = workers[0]
    loaded = {}
    for line in (proc_root / str(pid) / "maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith("/"):
            continue
        path = Path(parts[5])
        if (
            re.fullmatch(r"lib(?:torch|c10).*\.so", path.name)
            and path.name not in loaded
        ):
            require(
                path.is_file(),
                "native worker mapped a deleted or missing Torch library",
            )
            expected = context["torch_libraries"].get(path.name)
            actual = file_identity(path)
            require(
                expected
                and actual == {key: expected[key] for key in ("sha256", "size_bytes")},
                "resident native worker Torch library bytes differ from the selected Python stack",
            )
            loaded[path.name] = actual
    require(
        {"libtorch_cpu.so", "libtorch_cuda.so"} <= loaded.keys(),
        "resident native Torch library mappings are incomplete",
    )
    return {"pid": pid, "start": start, "libraries": loaded}


def native_metadata(result, context, run_id):
    execution = result.get("execution", {})
    native = result.get("payload", {}).get("native") or execution.get("native", {})
    require(
        execution.get("run_id") == run_id
        and execution.get("workflow") == context["workflow"]
        and execution.get("status") == "succeeded",
        "native HTTP canonical execution differs",
    )
    for key, value in {
        "runtime": "cpp",
        "backend": context["runtime"],
        "case_id": "run_1",
        "run_seed": 42,
        "permutation_seed": inference_seed(42, "run_1"),
        "source_sha": context["source_sha"],
        "mesh_sha256": CASE_PINS["run_1"]["sha256"],
        "geometry_sha256": CASE_PINS["run_1"]["geometry_sha256"],
    }.items():
        require(native.get(key) == value, "native HTTP provenance differs: " + key)
    identity = native.get("runtime_identity", {})
    require(
        native.get("release_image") == context["image"]
        or identity == {"kind": "registry-image", "value": context["image"]},
        "native HTTP image identity differs",
    )
    require(
        native.get("model_build_sources") == context["model_build_sources"],
        "native model compilation sources differ",
    )
    metadata = native["metadata"]
    points = CASE_POINT_COUNTS["run_1"]
    for key, value in {
        "point_count": points,
        "point_limit": 0,
        "block_size": 60000,
        "output_dtype": "float32",
        "math_profile": "provider-default-v1",
        "permutation": "torch.cuda.randperm",
        "permutation_seed": inference_seed(42, "run_1"),
    }.items():
        require(
            metadata.get(key) == value, "native full-case execution differs: " + key
        )
    require(
        [block["points"] for block in metadata["block_plan"]] == block_plan(points),
        "native full/tail block work differs",
    )
    deployment = metadata["deployment"]
    require(
        deployment["compute_capability"] == [9, 0], "native target architecture differs"
    )
    cuda = deployment["cuda_runtime_version"]
    header = deployment["torch_version"]
    selected = context["environment"]["torch"]
    if header != selected:
        require(
            context.get("native_stack_verified")
            and re.match(r"\d+\.\d+\.\d+", header)
            and re.match(r"\d+\.\d+\.\d+", selected)
            and re.match(r"\d+\.\d+\.\d+", header)[0]
            == re.match(r"\d+\.\d+\.\d+", selected)[0],
            "native Torch header version differs without exact loaded-library proof",
        )
    environment = {
        "python": context["environment"]["python"],
        "torch": selected,
        "cuda": f"{cuda // 1000}.{(cuda % 1000) // 10}",
        "vtk": metadata["vtk_version"],
        "math": metadata["math_flags"],
    }
    require(
        environment == context["environment"],
        "native numerical stack/math does not match the selected evaluator/reference",
    )
    return metadata, environment


def evaluate_native(context, physical, output):
    from qa.native_inference.cfd_evaluate import physical_field_predictions
    from qa.scripts.run_native_cfd_reference import _run_benchmark

    output.mkdir()
    with physical_field_predictions({"run_1": {"physical": physical}}):
        report = _run_benchmark(
            context["profile"],
            context["request"],
            output,
            context["input_root"],
            checkpoint=context["checkpoint"],
            stats=context["stats"],
        )
    path = output / "benchmark-output/benchmark_results.json"
    require(
        json.loads(path.read_text()) == report,
        "shared evaluator report bytes differ from its returned report",
    )
    return report, path


def collect_native(http, context, result, run_id, output):
    import numpy as np

    metadata, environment = native_metadata(result, context, run_id)
    outputs = result["execution"]["outputs"]
    names = [item.get("name") for item in outputs]
    require(
        set(names) == {"physical_fields", "standardized_fields", "native_metadata"}
        and len(names) == 3,
        "native artifact inventory differs",
    )
    downloaded = {}
    for item in outputs:
        name = item["name"]
        destination = output / name
        http.download(
            f"/v1/infer/{context['workflow']}/{run_id}/results?"
            + urllib.parse.urlencode({"artifact": name}),
            destination,
            item,
        )
        downloaded[name] = destination
    require(
        read_json_object(downloaded["native_metadata"]) == metadata,
        "downloaded native metadata differs from HTTP metadata",
    )
    fields = {}
    for name in ("physical", "standardized"):
        path = downloaded[name + "_fields"]
        require(
            path.stat().st_size == CASE_POINT_COUNTS["run_1"] * 16,
            "native field bytes do not cover full run_1",
        )
        fields[name] = np.memmap(
            path, mode="r", dtype="<f4", shape=(CASE_POINT_COUNTS["run_1"], 4)
        )
    report, path = evaluate_native(
        context, fields["physical"], output / "shared-evaluator"
    )
    return (
        report,
        path,
        fields,
        environment,
        {"model_build_sources": context["model_build_sources"], "metadata": metadata},
    )


def collect_python(http, context, result, run_id, output):
    execution, payload = result["execution"], result["payload"]
    require(
        execution.get("workflow") == context["workflow"]
        and execution.get("run_id") == run_id
        and execution.get("status") == "succeeded",
        "Python canonical execution differs",
    )
    require(
        payload.get("model_names") == ["transolver_surface"]
        and payload.get("case_ids") == ["run_1"]
        and payload.get("selected_metrics") == context["request"]["metrics"],
        "Python report work selection differs",
    )
    expected_provider = {
        key: context["profile"]["provider"][key]
        for key in (
            "repository",
            "tag",
            "version",
            "commit",
            "physicsnemo_version",
            "python_version",
        )
    }
    require(
        payload.get("provider") == expected_provider,
        "Python HTTP provider identity differs",
    )
    expected_inputs = [
        {
            key: item[key]
            for key in (
                "case_id",
                "sha256",
                "size_bytes",
                "geometry_sha256",
                "geometry_size_bytes",
            )
        }
        for item in context["request"]["cases"]
    ]
    require(
        payload.get("case_digests") == expected_inputs,
        "Python HTTP input identity differs",
    )
    outputs = {item["name"]: item for item in execution.get("outputs", [])}
    for name in (
        "benchmark_results.json",
        "benchmark_results.csv",
        "benchmark_results.html",
        "resolved_config.json",
        "benchmark_diagnostics.json",
        "benchmark.log",
    ):
        require(name in outputs, "Python report/audit artifact is missing: " + name)
    require(
        len(outputs) == len(execution["outputs"]),
        "Python artifact inventory has duplicate names",
    )
    for name in outputs:
        require(
            re.fullmatch(r"[A-Za-z0-9_.-]+", name), "Python artifact name is unsafe"
        )
        http.download_python(
            f"/v1/infer/{context['workflow']}/{run_id}/results?"
            + urllib.parse.urlencode({"artifact": name}),
            output / name,
        )
    config = read_json_object(output / "resolved_config.json")
    model = config["benchmark"]["models"]
    expected_model = context["profile"]["config"]["models"][0]
    kwargs = (
        {
            **model[0].get("kwargs", {}),
            "cuda_bf16_autocast": model[0]
            .get("kwargs", {})
            .get("cuda_bf16_autocast", False),
        }
        if model
        else {}
    )
    require(
        len(model) == 1
        and model[0]["name"] == "transolver_surface"
        and model[0]["package"] == expected_model["package"]
        and kwargs == expected_model["kwargs"]
        and config["run"]["seed"] == 42,
        "Python effective pinned model/config differs",
    )
    require(
        config["metrics"] == context["request"]["metrics"]
        and config["run"].get("save_inference_mesh") is False,
        "Python effective output work differs",
    )
    diagnostics = read_json_object(output / "benchmark_diagnostics.json")
    require(
        diagnostics.get("returncode") == 0
        and diagnostics.get("timed_out") is False
        and diagnostics.get("cancelled") is False
        and diagnostics.get("duration_seconds", 0) > 0
        and "physicsnemo.cfd.evaluation.benchmarks.run" in diagnostics.get("argv", []),
        "Python provider subprocess execution was not proven",
    )
    require(
        Path(diagnostics["argv"][0]).absolute()
        == Path(context["interpreter"]).absolute(),
        "Python provider interpreter differs from the verified numerical stack",
    )
    report = json.loads((output / "benchmark_results.json").read_text())
    _, metrics = normalize_report(report)
    require(
        {key[-1] for key in metrics} == set().union(*METRIC_OUTPUTS.values()),
        "Python scientific output metric inventory differs",
    )
    return (
        report,
        output / "benchmark_results.json",
        {},
        context["environment"],
        {
            "diagnostics": diagnostics,
            "math_observation": "qualified-reference, pinned provider/config and selected-interpreter probe",
        },
    )


def observe_python_cache(context, *, final=False):
    """Observe each new provider attempt before its metric cache can be reused."""
    if context["runtime"] != "python":
        return
    from qa.native_inference.cfd_benchmark_reference import cold_cache_observation

    for path in context["artifact_root"].rglob("resolved_config.json"):
        if path not in context["cache_configs"]:
            try:
                config = read_json_object(path)
            except json.JSONDecodeError:
                # The shipped writer creates the path before writing/flushing JSON.
                if final:
                    raise
                continue
            observation = cold_cache_observation(config, context["artifact_root"])
            require(
                observation["initially_absent"],
                "Python provider metric cache was not initially absent",
            )
            context["cache_configs"].add(path)


def request_sample(http, context, output, *, clock=time.perf_counter, collect=None):
    output.mkdir(parents=True)
    request = (
        context["request"]
        if context["runtime"] == "python"
        else {"case_id": "run_1", "seed": 42, "capture_inputs": False}
    )
    before_configs = len(context["cache_configs"])
    started = clock()
    submitted = http.json(
        f"/v1/infer/{context['workflow']}/run",
        body={"parameters": request},
        expected=202,
    )
    run_id = submitted.get("run_id")
    require(isinstance(run_id, str) and run_id, "HTTP submission has no run identity")
    deadline = time.monotonic() + context["timeout"]
    while time.monotonic() < deadline:
        observe_python_cache(context)
        status = http.json(f"/v1/infer/{context['workflow']}/{run_id}/status")
        require(
            status.get("status") not in ("failed", "cancelled"),
            "timed HTTP request failed or was cancelled",
        )
        if status.get("status") == "succeeded":
            try:
                result = http.json(f"/v1/infer/{context['workflow']}/{run_id}/results")
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            else:
                execution = result.get("execution", {})
                if (
                    execution.get("status") == "succeeded"
                    and execution.get("run_id") == run_id
                    and isinstance(result.get("payload"), dict)
                ):
                    if context["runtime"] == "python":
                        observe_python_cache(context, final=True)
                        require(
                            len(context["cache_configs"]) == before_configs + 1,
                            "exactly one fresh metric-cache attempt must be observed",
                        )
                    write_json_atomic(output / "http-results.json", result)
                    collector = collect or (
                        collect_python
                        if context["runtime"] == "python"
                        else collect_native
                    )
                    report, report_path, fields, environment, details = collector(
                        http, context, result, run_id, output
                    )
                    canonical = canonical_report(context["profile"], report)
                    canonical_path = output / "canonical-report.json"
                    canonical_path.write_bytes(canonical_bytes(canonical))
                    canonical_identity = file_identity(canonical_path)
                    files = {
                        relative_file(path, output): file_identity(path)
                        for path in output.rglob("*")
                        if path.is_file()
                    }
                    seconds = clock() - started
                    # Scientific/field comparisons are gates, outside the timed
                    # client/report work. All output downloads and hashes above
                    # are part of the wall interval for every runtime.
                    report_parity(
                        context["profile"], report, context["reference_report"]
                    )
                    check_fields(context, fields)
                    return {
                        "run_id": run_id,
                        "seconds": seconds,
                        "parity": "passed",
                        "artifacts_verified": True,
                        "canonical_report_sha256": canonical_identity["sha256"],
                        "report_path": relative_file(report_path, output),
                        "files": files,
                        "environment": environment,
                        "details": details,
                    }
        time.sleep(context["poll_interval"])
    raise FieldContractError("HTTP full-report request exceeded its deadline")


def check_fields(context, fields):
    import numpy as np

    for name, value in fields.items():
        tolerance = context["profile"]["native_reference"]["fields"][name]
        require(
            np.allclose(
                value, context["reference_fields"][name], **tolerance, equal_nan=False
            ),
            "native output field parity failed: " + name,
        )


def _process_snapshot(proc=Path("/proc")):
    """Observe parents, groups and start identities; zombies hold no service resources."""
    rows = {}
    if proc.is_dir():
        for path in proc.glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z":
                    rows[int(path.parent.name)] = (
                        int(fields[1]),
                        int(fields[2]),
                        fields[19],
                    )
            except (OSError, ValueError, IndexError):
                pass
        return rows
    # CPU contract checks also run on macOS, which has no Linux /proc.
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid=,state=,lstart="],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    for line in result.stdout.splitlines():
        pid, parent, group, state, started = line.strip().split(None, 4)
        if not state.startswith("Z"):
            rows[int(pid)] = (int(parent), int(group), started)
    return rows


def _process_token(pid, proc=Path("/proc")):
    if proc.is_dir():
        try:
            fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            return fields[19] if fields[0] != "Z" else None
        except (OSError, IndexError):
            return None
    row = _process_snapshot(proc).get(pid)
    return row[2] if row else None


def _signal_owned_processes(owned, signum):
    for pid, token in owned.items():
        if _process_token(pid) == token:
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


LINUX_SERVICE_GUARD = r"""
import ctypes, json, os, select, signal, socket, subprocess, sys, time
from pathlib import Path

control = socket.socket(fileno=int(sys.argv[1]))
configuration = b''
while b'\n' not in configuration:
    chunk = control.recv(65536)
    if not chunk:
        sys.exit(0)
    configuration += chunk
config = json.loads(configuration.split(b'\n', 1)[0])
control.setblocking(False)
read_signal, write_signal = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
signal.set_wakeup_fd(write_signal, warn_on_full_buffer=False)
interrupted = False
def interrupt(signum, frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGCHLD, lambda signum, frame: None)
signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
connected, failed, forced = True, False, False
launcher, launcher_start, deadline = None, None, None
sent_term, buffer = set(), b''

def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None

def emit(kind, **values):
    global connected
    if connected:
        try:
            control.sendall((json.dumps(dict(kind=kind, **values)) + '\n').encode())
        except OSError:
            connected = False
def signal_owned(pid, start, signum):
    if start is not None and identity(pid) == start:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

def stop(timeout):
    global deadline
    requested = time.monotonic() + timeout
    deadline = requested if deadline is None else min(deadline, requested)
    if launcher is not None and launcher.returncode is None:
        target = (launcher.pid, launcher_start)
        if target not in sent_term:
            signal_owned(*target, signal.SIGTERM)
            sent_term.add(target)

def direct_children():
    children = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) == os.getpid() and fields[0] != 'Z':
                children[int(path.parent.name)] = fields[19]
        except (OSError, ValueError, IndexError):
            pass
    return children

try:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'could not enable private service subreaper')
    launched_at = time.perf_counter()
    launcher = subprocess.Popen(config['command'], cwd=config['cwd'], start_new_session=True)
    launcher_start = identity(launcher.pid)
    emit('started', pid=launcher.pid, start=launcher_start, launched_at=launched_at)
except BaseException as error:
    failed = True
    emit('error', error=type(error).__name__)
    stop(0)

while True:
    try:
        empty = False
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                empty = True
                break
            if pid == 0:
                break
            if launcher is not None and pid == launcher.pid:
                launcher.returncode = os.waitstatus_to_exitcode(status)
                emit('exited', returncode=launcher.returncode)
        if interrupted or not connected:
            interrupted = False
            stop(min(config['timeout'], 10))
        if deadline is not None:
            if empty:
                emit('cleaned', forced=forced)
                break
            forced = forced or time.monotonic() >= deadline
            if forced or launcher is None or launcher.returncode is not None:
                # Reaping never occurs inside this batch of private children.
                for pid, start in direct_children().items():
                    target = (pid, start)
                    if forced or target not in sent_term:
                        signal_owned(pid, start, signal.SIGKILL if forced else signal.SIGTERM)
                        sent_term.add(target)
        wait = None if deadline is None else 0.05
        readable, _, _ = select.select(([control] if connected else []) + [read_signal], [], [], wait)
        if read_signal in readable:
            while True:
                try:
                    if not os.read(read_signal, 4096):
                        break
                except BlockingIOError:
                    break
        if connected and control in readable:
            try:
                chunk = control.recv(65536)
            except ConnectionResetError:
                chunk = b''
            if not chunk:
                connected = False
            else:
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    message = json.loads(line)
                    if message['kind'] == 'stop':
                        stop(message['timeout'])
    except BaseException as error:
        failed = True
        emit('error', error=type(error).__name__)
        stop(0)
control.close()
os.close(read_signal)
os.close(write_signal)
sys.exit(1 if failed else 0)
"""


class _LinuxService:
    """Expose CLI status while a private subreaper owns detached descendants."""

    def __init__(self, guard, control, command):
        self._pnms_service_guard, self._control = guard, control
        self.args, self.pid, self.returncode = command, None, None
        self._buffer, self._closed, self._cleanup, self._error = b"", False, None, None

    def _receive(self, timeout=0):
        if self._closed or not select.select([self._control], [], [], timeout)[0]:
            return False
        try:
            chunk = self._control.recv(65536)
        except ConnectionResetError:
            chunk = b""
        if not chunk:
            self._closed = True
            return False
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            message = json.loads(line)
            if message["kind"] == "started":
                self.pid = message["pid"]
                self._pnms_service_start = message["start"]
                self._pnms_service_launched_at = message["launched_at"]
            elif message["kind"] == "exited":
                self.returncode = message["returncode"]
            elif message["kind"] == "cleaned":
                self._cleanup = message
            elif message["kind"] == "error":
                self._error = message["error"]
        return True

    def _update(self):
        while self._receive():
            pass

    def _healthy(self):
        require(self._error is None, "private service guard failed")
        require(
            not self._closed or self._cleanup is not None,
            "private service guard exited before cleanup",
        )

    def poll(self):
        self._update()
        self._healthy()
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout)
            self._receive(remaining)
        return self.returncode

    def send_signal(self, signum):
        _signal_owned_processes({self.pid: self._pnms_service_start}, signum)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def _stop(self, timeout):
        failure = None
        try:
            self._update()
            if self._cleanup is None and not self._closed:
                self._control.sendall(
                    (json.dumps({"kind": "stop", "timeout": timeout}) + "\n").encode()
                )
            deadline = time.monotonic() + timeout + 10
            while self._cleanup is None:
                require(not self._closed, "private service guard exited before cleanup")
                remaining = deadline - time.monotonic()
                require(
                    remaining > 0, "private service guard cleanup exceeded deadline"
                )
                self._receive(remaining)
        except BaseException as error:
            failure = error
            raise
        finally:
            self._control.close()
            self._closed = True
            try:
                self._pnms_service_guard.wait(timeout=20)
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(
                    f"private service guard cleanup failed: {cleanup_error}"
                )
        self._healthy()
        require(
            self._pnms_service_guard.returncode == 0,
            "private service guard exited unexpectedly",
        )
        require(not self._cleanup["forced"], "shipped service required forced shutdown")


def _start_linux_service(command, cwd, environment, stream, timeout):
    require(
        math.isfinite(timeout) and timeout > 0,
        "shipped service shutdown timeout must be finite and positive",
    )
    control, child_control = socket.socketpair()
    guard = None
    try:
        guard = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                LINUX_SERVICE_GUARD,
                str(child_control.fileno()),
            ],
            pass_fds=(child_control.fileno(),),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        child_control.close()
        process = _LinuxService(guard, control, command)
        control.sendall(
            (
                json.dumps(
                    {
                        "command": command,
                        "cwd": None if cwd is None else str(cwd),
                        "timeout": timeout,
                    }
                )
                + "\n"
            ).encode()
        )
        deadline = time.monotonic() + 10
        while process.pid is None:
            process._healthy()
            remaining = deadline - time.monotonic()
            require(remaining > 0, "private service guard startup exceeded deadline")
            process._receive(remaining)
        return process
    except BaseException as error:
        control.close()
        if guard is not None:
            try:
                guard.terminate()
                guard.wait(timeout=20)
            except BaseException as cleanup_error:
                error.add_note(
                    f"private service guard startup cleanup failed: {cleanup_error}"
                )
        raise
    finally:
        child_control.close()


def _marked_service_processes(rows, owner, proc=Path("/proc")):
    """Discover inherited launch ownership even after detached parents have exited."""
    if owner is None:
        return {}
    marker = f"{SERVICE_OWNER_ENV}={owner}"
    candidates = []
    if proc.is_dir():
        for pid in rows:
            if _process_token(pid, proc) != rows[pid][2]:
                continue
            try:
                if marker.encode() in (proc / str(pid) / "environ").read_bytes().split(
                    b"\0"
                ):
                    candidates.append(pid)
            except OSError:
                pass
    else:
        result = subprocess.run(
            ["ps", "eww", "-axo", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
        require(result.returncode == 0, "could not inspect service ownership")
        for line in result.stdout.splitlines():
            fields = line.split(None, 1)
            if len(fields) != 2:
                continue
            pid, command = fields
            if marker in command.split():
                candidates.append(int(pid))
    return {
        pid: rows[pid][2]
        for pid in candidates
        if pid in rows and _process_token(pid, proc) == rows[pid][2]
    }


def _observe_service_processes(process, owned):
    """Retain start-guarded identities and discover marked replacement workers."""
    rows = _process_snapshot()
    live = {
        pid: token
        for pid, token in owned.items()
        if pid in rows and rows[pid][2] == token
    }
    owner = getattr(process, "_pnms_service_owner", None)
    live.update(_marked_service_processes(rows, owner))
    leader = rows.get(process.pid)
    expected = getattr(process, "_pnms_service_start", None)
    if leader is None or (expected is None and owner is None) or leader[2] == expected:
        live.update((pid, row[2]) for pid, row in rows.items() if row[1] == process.pid)
    pending = list(live)
    while pending:
        parent = pending.pop()
        for pid, row in rows.items():
            if row[0] == parent and pid not in live:
                live[pid] = row[2]
                pending.append(pid)
    owned.update(live)
    return live


def start_service(args, context):
    owner = os.urandom(16).hex()
    environment = {
        **os.environ,
        "RELEASE_IMAGE_DIGEST": args.expected_image,
        "ARTIFACT_DIR": str(context["artifact_root"]),
        "DEFAULT_OUTPUT_DIR": str(context["artifact_root"]),
        "PNMS_OUTPUT_ROOT": str(context["artifact_root"]),
        "HEALTH_STUB_ENABLED": "false",
        "PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID": context["workflow"],
        SERVICE_OWNER_ENV: owner,
    }
    # The Lepton job pins the actual registry image. Use that identity rather
    # than an unrelated host-launcher configuration variable inherited by QA.
    environment.pop("PNMS_IMAGE_CONFIG_DIGEST", None)
    if args.runtime == "python":
        environment.pop("NATIVE_DEPLOYMENT_CONFIG", None)
        environment.update(
            WORKERS="all",
            REDIS_STREAM_PREFIX="",
            STREAM_PREFIX="",
            E2S_PREFETCH_ALLOWED_HTTPS_HOSTS=(
                "huggingface.co,us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"
            ),
            E2S_PREFETCH_ALLOWED_SIGNED_REDIRECT_HOSTS=(
                "us.aws.cdn.hf.co,cas-bridge.xethub.hf.co"
            ),
        )
    stream = (args.output / "service.log").open("wb")
    process = None
    try:
        if sys.platform == "linux":
            process = _start_linux_service(
                context["command"],
                args.service_cwd,
                environment,
                stream,
                getattr(args, "shutdown_timeout", 10),
            )
            observed = _process_token(process.pid)
            require(
                observed is None or observed == process._pnms_service_start,
                "shipped service process identity changed during startup",
            )
        else:
            process = subprocess.Popen(
                context["command"],
                cwd=args.service_cwd,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process._pnms_service_owner = owner
            process._pnms_service_owned = {}
            process._pnms_service_start = _process_token(process.pid)
    except BaseException as error:
        if process is not None:
            try:
                stop_service(process, 10)
            except BaseException as cleanup_error:
                error.add_note(f"service startup cleanup failed: {cleanup_error}")
        stream.close()
        raise
    return process, stream


def wait_ready(args, context, http, process):
    deadline = time.monotonic() + args.startup_timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, "shipped service exited before readiness")
        try:
            ready = http.json(f"/v1/infer/{context['workflow']}/readiness")
            if ready.get("readiness", {}).get("ready") is True:
                return ready
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise FieldContractError("shipped service did not become ready before its deadline")


def stop_service(process, timeout):
    require(
        math.isfinite(timeout) and timeout > 0,
        "shipped service shutdown timeout must be finite and positive",
    )
    if process is None:
        return
    if isinstance(process, _LinuxService):
        process._stop(timeout)
        return
    expected = getattr(process, "_pnms_service_start", None)
    initial = _process_snapshot().get(process.pid)
    require(
        initial is None or expected is None or initial[2] == expected,
        "shipped service process identity changed before shutdown",
    )
    owned = getattr(process, "_pnms_service_owned", {})

    def observe():
        return _observe_service_processes(process, owned)

    deadline = time.monotonic() + timeout
    live = observe()  # Preserve detached children's identities before CLI exit.
    if process.pid in live:
        _signal_owned_processes({process.pid: live[process.pid]}, signal.SIGTERM)
    signalled = set()
    while True:
        exited = process.poll() is not None
        live = observe()
        if not live and exited:
            return
        if exited:
            remaining = {
                pid: token
                for pid, token in live.items()
                if (pid, token) not in signalled
            }
            _signal_owned_processes(remaining, signal.SIGTERM)
            signalled.update(remaining.items())
        if time.monotonic() >= deadline:
            break
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))

    _signal_owned_processes(live, signal.SIGKILL)
    process.wait(timeout=10)
    forced_deadline = time.monotonic() + 10
    while live := observe():
        require(
            time.monotonic() < forced_deadline,
            "shipped service descendants survived forced shutdown",
        )
        _signal_owned_processes(live, signal.SIGKILL)
        time.sleep(0.05)
    raise FieldContractError("shipped service required forced shutdown")


def run_session(args):
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    summary = {
        "schema_version": 1,
        "status": "running",
        "boundary": BOUNDARY,
        "runtime": args.runtime,
        "image_digest": args.expected_image,
        "source_sha": args.expected_source_sha,
        "run_id": args.run_id,
        "ordinal": args.ordinal,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "performance_claim": "client/full-report service boundary only",
        "gpu_memory_peak_measured": False,
    }
    process, stream = None, None
    try:
        context = prepare(args)
        summary.update(
            hardware=context["hardware"],
            environment=context["environment"],
            evaluator_environment=context["environment"],
            service_files=context["service_files"],
            model_build_sources=context["model_build_sources"],
            bundle_digest=context["bundle_digest"],
            reference_qualification=context["qualification"],
            service_command=context["command"],
            service_artifact_root=str(context["artifact_root"]),
            selected_interpreter=context["interpreter"],
            numerical_libraries={
                name: {key: item[key] for key in ("sha256", "size_bytes")}
                for name, item in context["torch_libraries"].items()
            },
        )
        summary["workload"] = {
            "profile_sha256": sha256_file(args.profile),
            "request_sha256": sha256_file(args.request),
            "reference_fields_sha256": sha256_file(args.reference_fields),
            "reference_report_sha256": context["qualification"]["report_sha256"],
            "provider": context["profile"]["provider"],
            "model_revision": MODEL_REVISION,
            "checkpoint": ASSET_PINS["checkpoint"],
            "statistics": ASSET_PINS["statistics"],
            "case": "run_1",
            "point_count": CASE_POINT_COUNTS["run_1"],
            "seed": 42,
            "block_size": 60000,
            "precision": "float32",
        }
        http = HTTP(args.url)
        started = time.perf_counter()
        process, stream = start_service(args, context)
        started = getattr(process, "_pnms_service_launched_at", started)
        summary["initial_readiness"] = wait_ready(args, context, http, process)
        summary["cold_startup_seconds"] = time.perf_counter() - started
        if hasattr(process, "_pnms_service_owned"):
            _observe_service_processes(process, process._pnms_service_owned)
        if args.runtime != "python":
            context["native_stack_verified"] = verify_native_stack(process, context)
            summary["native_process"] = context["native_stack_verified"]
        for label in ("prerequisite", "first", "warm-1", "warm-2", "warm-3", "postrun"):
            require(process.poll() is None, "shipped service exited during sampling")
            sample = request_sample(http, context, args.output / label)
            sample.update(label=label, directory=label)
            if label in ("prerequisite", "postrun"):
                summary[label] = sample
            else:
                summary["samples"].append(sample)
            require(
                sample["environment"] == context["environment"],
                "measured runtime environment differs",
            )
            require(
                sample["canonical_report_sha256"]
                == summary["prerequisite"]["canonical_report_sha256"],
                "canonical output changed during the session",
            )
            if args.runtime != "python":
                require(
                    verify_native_stack(process, context)
                    == context["native_stack_verified"],
                    "resident native worker or selected libraries changed during timing",
                )
            write_json_atomic(args.output / "session.json", summary)
        summary["final_readiness"] = wait_ready(args, context, http, process)
        if hasattr(process, "_pnms_service_owned"):
            _observe_service_processes(process, process._pnms_service_owned)
        summary["status"] = "passed"
        summary["methodology"] = {
            "cold_startup": "service process launch through workflow readiness; excludes source/asset verification and exact library-map verification",
            "prerequisite": "untimed qualification request before performance samples; its observed client wall is retained separately",
            "first": "first measured request after prerequisite; not cold-model inference",
            "warm": "three sequential requests through the same running service",
            "native_report": "actual HTTP field downloads followed by unchanged shared evaluator; all output read/hash work included",
            "python_report": "actual HTTP full report, CSV/HTML and every registered audit artifact downloaded and hashed",
            "validation": "metric/field comparisons happen after the measured wall interval",
            "peak_memory": "not measured",
        }
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            stop_service(process, args.shutdown_timeout)
        except BaseException as error:
            summary.update(
                status="failed", shutdown_error=f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            if stream is not None:
                stream.close()
            summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(args.output / "session.json", summary)
            print(
                SUMMARY_BEGIN
                + "\n"
                + json.dumps(summary, sort_keys=True)
                + "\n"
                + SUMMARY_END,
                flush=True,
            )
    return 0


def load_session(path, profile, reference, reference_fields):
    path = Path(path).resolve(strict=True)
    session = read_json_object(path)
    context = {"profile": profile, "reference_fields": reference_fields}
    for gate in [
        session.get("prerequisite", {}),
        *session.get("samples", []),
        session.get("postrun", {}),
    ]:
        require(
            gate.get("environment") == session.get("environment"),
            "sample runtime environment differs from its session",
        )
        directory = gate.get("directory")
        require(
            isinstance(directory, str)
            and directory
            in ("prerequisite", "first", "warm-1", "warm-2", "warm-3", "postrun"),
            "session sample directory is invalid",
        )
        root = path.parent / directory
        require(
            not root.is_symlink() and root.resolve().is_relative_to(path.parent),
            "session sample directory escaped its receipt",
        )
        files = gate.get("files", {})
        require(
            files and "canonical-report.json" in files,
            "session has no hashed output artifacts",
        )
        for relative, identity in files.items():
            selected = root / relative
            require(
                isinstance(relative, str)
                and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts
                and not selected.is_symlink()
                and selected.resolve().is_relative_to(root.resolve()),
                "unsafe session artifact path",
            )
            verify_file(selected, identity)
        report_relative = gate.get("report_path")
        require(report_relative in files, "full scientific report is not hash bound")
        report = json.loads((root / report_relative).read_text())
        report_parity(profile, report, reference)
        canonical = canonical_bytes(canonical_report(profile, report))
        require(
            (root / "canonical-report.json").read_bytes() == canonical
            and hashlib.sha256(canonical).hexdigest()
            == gate.get("canonical_report_sha256"),
            "canonical output hash differs from the actual full report",
        )
        if session["runtime"] != "python":
            import numpy as np

            require(
                all(name + "_fields" in files for name in ("physical", "standardized")),
                "native session lacks its downloaded fields",
            )
            fields = {
                name: np.memmap(
                    root / (name + "_fields"),
                    mode="r",
                    dtype="<f4",
                    shape=(CASE_POINT_COUNTS["run_1"], 4),
                )
                for name in ("physical", "standardized")
            }
            check_fields(context, fields)
        gate["parity"], gate["artifacts_verified"] = "passed", True
    return session


def compare(args):
    from qa.native_inference.cfd_benchmark_reference import qualified_reference

    profile = read_json_object(args.profile)
    reference, qualification = qualified_reference(args.reference_fields)
    fields = validate_evidence(
        profile, read_json_object(args.reference_fields), args.reference_fields.parent
    )["run_1"]
    sessions = [
        (path, load_session(path, profile, reference, fields)) for path in args.sessions
    ]
    for _, row in sessions:
        for key, value in {
            "profile_sha256": sha256_file(args.profile),
            "reference_fields_sha256": sha256_file(args.reference_fields),
            "reference_report_sha256": qualification["report_sha256"],
        }.items():
            require(
                row.get("workload", {}).get(key) == value,
                "comparison reference/profile hash does not match measured workload",
            )
    ordinals = [row.get("ordinal") for _, row in sessions]
    require(
        all(type(value) is int and value >= 0 for value in ordinals)
        and len(set(ordinals)) == len(ordinals),
        "session execution ordinals must be unique nonnegative integers",
    )
    sessions.sort(key=lambda item: item[1]["ordinal"])
    summary = compare_sessions([row for _, row in sessions])
    summary["session_receipts"] = [
        {"path": str(path.resolve()), **file_identity(path)} for path, _ in sessions
    ]
    write_json_atomic(args.output, summary)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    session = commands.add_parser(
        "session",
        help="Run inside the selected immutable Lepton image; never invokes Docker",
    )
    session.add_argument("--runtime", choices=RUNTIMES, required=True)
    for name in (
        "profile",
        "request",
        "reference-fields",
        "input-root",
        "service-command-json",
        "service-files-json",
        "output",
    ):
        session.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-image", "expected-source-sha", "run-id"):
        session.add_argument("--" + name, required=True)
    session.add_argument("--ordinal", type=int, required=True)
    for name in ("checkpoint", "stats", "native-bundle", "service-cwd"):
        session.add_argument("--" + name, type=Path)
    session.add_argument(
        "--service-artifact-root",
        type=Path,
        help="Fresh output tree used by the selected public service command",
    )
    session.add_argument(
        "--environment-python", type=Path, default=Path(sys.executable)
    )
    session.add_argument("--url", default="http://127.0.0.1:8080")
    for name, default in (
        ("startup-timeout", 1800),
        ("request-timeout", 3600),
        ("shutdown-timeout", 300),
    ):
        session.add_argument("--" + name, type=float, default=default)
    session.add_argument("--poll-interval", type=float, default=0.1)
    comparison = commands.add_parser(
        "compare", help="Verify local/NFS output bytes and compare matched sessions"
    )
    comparison.add_argument("--sessions", type=Path, nargs="+", required=True)
    for name in ("profile", "reference-fields", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    environment = commands.add_parser("environment")
    for name in ("profile", "output"):
        environment.add_argument("--" + name, type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "environment":
        probe_environment(args)
        return 0
    if args.command != "session":
        return compare(args)

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    previous = {
        signum: signal.signal(signum, interrupted)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        return run_session(args)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
