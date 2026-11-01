#!/usr/bin/env python3
"""Finite Lepton acceptance of a sealed CFD release image.

This versioned host controller supplies its QA wrapper without changing the
frozen source image. It never builds models or computes cross-image speedups.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402

BENCHMARK_SHA256 = "d7c0a1f5ae859e0ad51d40c3e46857160f78618bcfe7af5dd37e92528d9d5dbb"
ORIGIN = "c85aed8f5e354cc210339cb73d72080cbd493992"
GATES = [
    "preflight",
    "prepare",
    "startup",
    "request.first",
    "request.repeat",
    "residency",
    "isolation",
    "shutdown",
]
BEGIN = "PNMS_CFD_IMAGE_QA_BEGIN"
END = "PNMS_CFD_IMAGE_QA_END"
MAX_REPORT = 256 * 1024

# Stdlib only until the verified, shipped product and existing benchmark load.
# The identical helpers execute in CPU contract tests and in the Lepton image.
REMOTE = r"""
import argparse, base64, hashlib, json, math, os, pathlib, runpy, signal, sys
Path = pathlib.Path
PREFIXES = ("application-cli/", "scripts/", "python/", "crates/worker-runtime/config/")
def require(value, message):
    if not value:
        raise ValueError(message)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def identity(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
def file_identity(path):
    path = Path(path)
    with path.open("rb") as stream:
        return dict(sha256=hashlib.file_digest(stream, "sha256").hexdigest(), size_bytes=path.stat().st_size)
def read(path):
    return json.loads(Path(path).read_bytes())
def bind_sources(workspace, installed, metadata, inventory, source_sha, inventory_sha):
    workspace, installed, inventory = Path(workspace).resolve(), Path(installed).resolve(), Path(inventory)
    require(file_identity(inventory)["sha256"] == inventory_sha, "source inventory differs")
    meta, records = read(metadata), read(inventory)
    require(meta.get("source_sha") == source_sha and meta.get("source_inventory_sha256") == inventory_sha, "source metadata differs")
    require(isinstance(records, dict) and records, "source inventory is empty")
    selected = {}
    for relative, expected in records.items():
        path = pathlib.PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        expected = {key: expected[key] for key in ("sha256", "size_bytes")}
        original = workspace / relative
        require(not original.is_symlink() and original.resolve().is_relative_to(workspace) and file_identity(original) == expected, "workspace source differs: " + relative)
        if relative.startswith(PREFIXES):
            target = installed / relative
            require(not target.is_symlink() and target.resolve().is_relative_to(installed) and file_identity(target) == expected, "installed source differs: " + relative)
            selected[relative] = expected
    actual = {path.relative_to(installed).as_posix() for path in installed.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    require(actual == set(selected), "installed source file inventory differs")
    return dict(workspace_files=len(records), installed_files=len(selected), installed_inventory_sha256=identity(canonical(selected))["sha256"])
def validate_startup(report, backend):
    require(report.get("status") == "passed" and report.get("policy") == "expected-fp32-v1" and report.get("session_load_count") == 2, "independent startup golden policy/count differs")
    models = report.get("models", [])
    require(len(models) == 2 and {item.get("point_count") for item in models} == {60000, 8095}, "startup goldens do not cover full and tail")
    for item in models:
        require(item.get("backend") == backend and item.get("shape") == [1, item["point_count"], 4] and item.get("dtype") == "float32" and item.get("finite") is True, "startup golden model differs")
        comparison = item.get("comparison")
        require(isinstance(comparison, dict) and all(type(comparison.get(key)) in (int, float) and math.isfinite(comparison[key]) and comparison[key] >= 0 for key in ("rtol", "atol", "max_absolute_error", "max_relative_error")), "independent startup golden comparison is missing")
def validate_requests(rows):
    require(isinstance(rows, list) and len(rows) == 2, "exactly two full requests required")
    for row in rows:
        require(isinstance(row.get("run_id"), str) and row["run_id"] and row.get("parity") == "passed" and row.get("artifacts_verified") is True, "request did not pass scientific/artifact gates")
        require(type(row.get("seconds")) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] > 0, "request duration differs")
        metadata = row.get("metadata", {})
        require(metadata.get("session_load_count") == 2 and type(metadata.get("initialization_ms")) in (int, float) and math.isfinite(metadata["initialization_ms"]) and metadata["initialization_ms"] > 0, "request resident session count/initialization differs")
        require({"physical_fields", "standardized_fields", "native_metadata", "http-results.json"} <= row.get("files", {}).keys(), "request downloads are incomplete")
        require(set(row.get("artifact_paths", {})) == {"physical_fields", "standardized_fields", "native_metadata"} and len(set(row["artifact_paths"].values())) == 3, "request output inventory differs")
        for path in row["artifact_paths"].values():
            require(pathlib.PurePosixPath(path).is_absolute() and ".." not in pathlib.PurePosixPath(path).parts, "request artifact path is unsafe")
        for record in row["files"].values():
            require(isinstance(record, dict) and len(record.get("sha256", "")) == 64 and type(record.get("size_bytes")) is int and record["size_bytes"] > 0, "request artifact identity differs")
    first, second = rows
    require(first["run_id"] != second["run_id"] and first.get("output") != second.get("output"), "requests reused an identity or output directory")
    require(first["metadata"]["initialization_ms"] == second["metadata"]["initialization_ms"], "resident model initialization changed")
    require(set(first["artifact_paths"].values()).isdisjoint(second["artifact_paths"].values()) and {str(pathlib.PurePosixPath(path).parent) for path in first["artifact_paths"].values()}.isdisjoint({str(pathlib.PurePosixPath(path).parent) for path in second["artifact_paths"].values()}), "requests reused artifact output paths")
    for name in ("physical_fields", "standardized_fields"):
        require(first["files"][name] == second["files"][name], "same-seed field bytes changed")
    require(first.get("canonical_report_sha256") == second.get("canonical_report_sha256") and len(first.get("canonical_report_sha256", "")) == 64, "canonical report changed")
def session_arguments(config, output, command, files):
    command.write_bytes(canonical([sys.executable, "-m", "pnms_app.image_entrypoint"]))
    mount = config["mount"].rstrip("/")
    root = Path(config.get("repository", "/workspace"))
    return argparse.Namespace(runtime=config["backend"], profile=root/"qa/inference/cfd_parity_surface_run1_transolver.json", request=root/"plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json", reference_fields=Path(config.get("reference_fields") or mount+"/native-inference/cfd-full-20261021t164815-r3/reference/fields.json"), input_root=Path(config.get("input_root") or mount+"/native-inference/cfd-full-20261021t164815-r3/inputs/drivaerml"), service_command_json=command, service_files_json=files, output=output, expected_image=config["image"], expected_source_sha=config["source_sha"], run_id=config["run_id"], ordinal=0, checkpoint=Path(mount+"/assets/transolver/15e1c59a166037e2/Transolver.0.501.mdlus"), stats=Path(mount+"/assets/transolver/15e1c59a166037e2/global_stats.json"), native_bundle=Path("/opt/physicsnemo-release"), service_cwd=Path("/opt/pnms/source"), service_artifact_root=output/"service-artifacts", environment_python=Path(sys.executable), url="http://127.0.0.1:8080", startup_timeout=min(1800, config["timeout"]), request_timeout=min(1800, config["timeout"]), shutdown_timeout=300, poll_interval=0.1)
def registry_snapshot(plan, process, context, benchmark, proc_root=Path("/proc")):
    import redis
    client = redis.Redis.from_url(plan["environment"]["REDIS_URL"], decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
    try:
        records = [json.loads(raw) for raw in client.hgetall("gpu:registry").values()]
    finally:
        client.close()
    require(len(records) == 1, "expected exactly one native worker registry slot")
    worker = records[0]
    require(worker.get("status") == "available" and worker.get("deployment_id") == plan["deployment_id"] and worker.get("bundle_digest") == context["bundle_digest"] and worker.get("workflow_id") == context["workflow"], "native registry ownership differs")
    startup = worker.get("model_cache", {}).get("warmup", {}).get("self_test", {})
    validate_startup(startup, context["runtime"])
    stack = benchmark["verify_native_stack"](process, context, proc_root=proc_root)
    require(worker.get("pid") == stack["pid"] and isinstance(worker.get("worker_instance_id"), str) and worker["worker_instance_id"], "native worker process identity differs")
    expected = Path("/opt/physicsnemo-release/workflow/lib/libtransolver_workflow.so").resolve()
    mapped = set()
    for line in (proc_root / str(stack["pid"]) / "maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libtransolver_workflow.so"):
            mapped.add(Path(fields[5]).resolve())
    require(mapped == {expected} and expected.is_file(), "native worker mapped a foreign Transolver library")
    proof = dict(pid=stack["pid"], start=stack["start"], worker_instance_id=worker["worker_instance_id"], bundle_digest=worker["bundle_digest"], library=dict(path=str(expected), **file_identity(expected)))
    return proof, startup


def start_stack(args, context, benchmark):
    process, stream = benchmark["start_service"](args, context)
    try:
        http = benchmark["HTTP"](args.url)
        benchmark["wait_ready"](args, context, http, process)
        return process, stream, http
    except BaseException:
        try:
            benchmark["stop_service"](process, args.shutdown_timeout)
        finally:
            stream.close()
        raise
def request_record(row, destination, actual):
    metadata = row["details"]["metadata"]
    artifacts = actual["execution"]["outputs"]
    paths = {item["name"]: item["storage_path"] for item in artifacts}
    record = {key: row[key] for key in ("run_id", "seconds", "parity", "artifacts_verified", "canonical_report_sha256", "files")}
    record.update(output=str(destination), metadata={key: metadata[key] for key in ("session_load_count", "initialization_ms")}, artifact_paths=paths)
    native = actual.get("payload", {}).get("native") or actual.get("execution", {}).get("native", {})
    record["provenance"] = {key: native.get(key) for key in ("source_sha", "backend", "release_image", "model_build_sources", "attempt")}
    return record

def remote_main(config):
    output = Path(config["output"])
    report = dict(schema_version=1, status="failed", profile="cfd-image", run_id=config["run_id"], source_sha=config["source_sha"], image_digest=config["image"], backend=config["backend"], bundle_digest=config["bundle_digest"], source_inventory_sha256=config["inventory_sha256"], benchmark_sha256=config["benchmark_sha256"], controller_sha256=config["controller_sha256"], expected_cases=config["gates"], cases=[], requests=[], residency=[], recompiled=False)
    process, stream, benchmark = None, None, None
    owned = False
    def gate(name, action):
        result = action()
        report["cases"].append(dict(name=name, status="passed"))
        return result
    def interrupted(signum, frame):
        raise TimeoutError("finite CFD image QA interrupted: " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        output.mkdir(parents=True, exist_ok=False)
        owned = True
        repository = Path(config.get("repository", "/workspace"))
        def preflight():
            bindings = bind_sources(repository, Path("/opt/pnms/source"), "/opt/physicsnemo-qa/source.json", "/opt/physicsnemo-qa/source-inventory.json", config["source_sha"], config["inventory_sha256"])
            path = repository / "qa/scripts/benchmark_pnms_cfd.py"
            require(file_identity(path)["sha256"] == config["benchmark_sha256"], "existing benchmark bytes differ")
            # Load product modules before the benchmark prepends /workspace.
            import pnms_app.bundle, pnms_app.image_entrypoint
            require(Path(pnms_app.image_entrypoint.__file__).resolve() == Path("/opt/pnms/source/application-cli/src/pnms_app/image_entrypoint.py"), "entrypoint did not load shipped installed product")
            bundle = Path("/opt/physicsnemo-release")
            seal = pnms_app.bundle.verify(bundle, expected=config["bundle_digest"])
            descriptor = read(bundle / "bundle.json")
            origins = {"full": config["origin"], "tail": config["origin"]}
            require(descriptor.get("adapter") == "transolver_surface" and descriptor.get("backend") == config["backend"] and descriptor.get("source_sha") == config["source_sha"] and descriptor.get("model_build_sources") == origins and seal["bundle_digest"] == config["bundle_digest"], "shipped bundle identity differs")
            report["model_build_sources"] = descriptor["model_build_sources"]
            bindings["binaries"] = {name: file_identity(Path("/opt/pnms/bin") / name) for name in ("native-worker", "worker-runtime", "inference_server")}
            report["source_bindings"] = bindings
            files = {str(Path("/opt/pnms/source") / relative): {key: item[key] for key in ("sha256", "size_bytes")} for relative, item in read("/opt/physicsnemo-qa/source-inventory.json").items() if relative.startswith(PREFIXES)}
            files.update({str(Path("/opt/pnms/bin") / name): record for name, record in bindings["binaries"].items()})
            files[str(bundle / "inventory.json")] = file_identity(bundle / "inventory.json")
            (output / "service-files.json").write_bytes(canonical(files))
            return runpy.run_path(str(path), run_name="cfd_image_verified_benchmark")
        benchmark = gate("preflight", preflight)
        args = session_arguments(config, output, output/"service-command.json", output/"service-files.json")
        context = gate("prepare", lambda: benchmark["prepare"](args))
        report["hardware"] = context["hardware"]
        report["environment"] = context["environment"]
        report["service_command"] = context["command"]
        def startup():
            nonlocal process, stream
            process, stream, http = start_stack(args, context, benchmark)
            plans = list(context["artifact_root"].glob(".pnms/service-*/launch.json"))
            require(len(plans) == 1, "fresh shipped image service launch inventory differs")
            plan = read(plans[0])
            require(plan["in_image"] is True and plan["skip_build"] is True and plan["bundle_digest"] == config["bundle_digest"] and plan["repository"] == "/opt/pnms/source", "shipped service plan differs")
            proof, golden = registry_snapshot(plan, process, context, benchmark)
            report["startup"] = golden
            report["residency"].append(proof)
            context["native_stack_verified"] = True
            return http, plan
        http, plan = gate("startup", startup)
        for label in ("first", "repeat"):
            def request():
                destination = output / ("request-" + str(len(report["requests"])))
                row = benchmark["request_sample"](http, context, destination)
                actual = read(destination / "http-results.json")
                native = actual.get("payload", {}).get("native") or actual.get("execution", {}).get("native", {})
                require(native.get("model_build_sources") == report["model_build_sources"], "actual request model origins differ")
                from qa.scripts.run_native_cfd_http_job import assert_result
                assert_result(actual, run_id=row["run_id"], source_sha=config["source_sha"], image=config["image"], artifact_root=context["artifact_root"], capture_inputs=False, model_build_sources=report["model_build_sources"], backend=config["backend"])
                record = request_record(row, destination, actual)
                report["requests"].append(record)
                proof, golden = registry_snapshot(plan, process, context, benchmark)
                require(golden == report["startup"], "resident startup goldens changed")
                report["residency"].append(proof)
            gate("request." + label, request)
        gate("residency", lambda: require(all(row == report["residency"][0] for row in report["residency"]), "resident worker/library changed"))
        gate("isolation", lambda: validate_requests(report["requests"]))
        def shutdown():
            from qa.native_inference.pnms_http_job import _descendants
            children = _descendants(process.pid)
            benchmark["stop_service"](process, args.shutdown_timeout)
            remaining = []
            for pid, start in children.items():
                stat = Path("/proc") / str(pid) / "stat"
                if stat.exists():
                    fields = stat.read_text().rsplit(")", 1)[1].split()
                    if fields[0] != "Z" and fields[19] == str(start):
                        remaining.append(pid)
            require(process.returncode == 0 and not remaining, "shipped image stack did not retire cleanly")
            report["shutdown"] = dict(returncode=process.returncode, descendants_remaining=remaining)
        gate("shutdown", shutdown)
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        if benchmark and process and process.poll() is None:
            try:
                benchmark["stop_service"](process, 30)
            except BaseException as error:
                report["status"] = "failed"
                report["cleanup_error"] = str(error)
        if stream:
            stream.close()
        raw = canonical(report)
        require(len(raw) <= 262144, "report exceeds finite transport")
        if owned:
            (output / "summary.json").write_bytes(raw)
        print("PNMS_CFD_IMAGE_QA_BEGIN", flush=True)
        print(json.dumps(dict(report_base64=base64.b64encode(raw).decode(), **identity(raw)), sort_keys=True), flush=True)
        print("PNMS_CFD_IMAGE_QA_END", flush=True)
    return 0 if report["status"] == "passed" else 1
if __name__ == "__main__":
    raise SystemExit(remote_main(json.loads(sys.argv[1])))
"""

_HELPERS = {"__name__": "cfd_image_host_contract"}
exec(REMOTE, _HELPERS)  # noqa: S102 - Execute the reviewed literal wrapper.
require = _HELPERS["require"]
identity = _HELPERS["identity"]
canonical = _HELPERS["canonical"]


def strict_json(raw):
    def pairs(items):
        record = {}
        for key, value in items:
            require(key not in record, "duplicate evidence field: " + key)
            record[key] = value
        return record

    def invalid(value):
        raise ValueError("nonfinite evidence value: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("cfd-image",))
    parser.set_defaults(job_timeout=7200)
    parser.add_argument("--backend", choices=("aoti", "tensorrt"), required=True)
    parser.add_argument("--expected-bundle-digest", required=True)
    parser.add_argument("--expected-source-inventory-sha256", required=True)
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument("--reference-fields")
    parser.add_argument("--input-root")
    return parser


def validate_summary(
    report,
    *,
    run_id,
    source_sha,
    image_digest,
    profile,
    backend,
    bundle_digest,
    source_inventory_sha256,
):
    for key, value in {
        "schema_version": 1,
        "status": "passed",
        "profile": profile,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "backend": backend,
        "bundle_digest": bundle_digest,
        "source_inventory_sha256": source_inventory_sha256,
        "benchmark_sha256": BENCHMARK_SHA256,
        "controller_sha256": identity(Path(__file__).read_bytes())["sha256"],
        "model_build_sources": {"full": ORIGIN, "tail": ORIGIN},
        "recompiled": False,
    }.items():
        require(report.get(key) == value, "CFD release image identity differs: " + key)
    require(
        report.get("expected_cases") == GATES
        and report.get("cases")
        == [{"name": name, "status": "passed"} for name in GATES],
        "CFD release image gate inventory differs",
    )
    require(
        "error" not in report and "cleanup_error" not in report,
        "CFD release image carries a failure",
    )
    _HELPERS["validate_startup"](report.get("startup", {}), backend)
    _HELPERS["validate_requests"](report.get("requests"))
    for request in report["requests"]:
        actual = request.get("provenance", {})
        for key, value in {
            "source_sha": source_sha,
            "backend": backend,
            "release_image": image_digest,
            "model_build_sources": {"full": ORIGIN, "tail": ORIGIN},
        }.items():
            require(actual.get(key) == value, "request provenance differs: " + key)
        require(
            re.fullmatch(r"[a-f0-9-]{36}", actual.get("attempt", "")),
            "request attempt provenance is missing",
        )
    rows = report.get("residency", [])
    require(
        len(rows) == 3 and all(row == rows[0] for row in rows),
        "resident worker identity changed",
    )
    observed = rows[0]
    require(
        type(observed.get("pid")) is int
        and observed["pid"] > 0
        and str(observed.get("start", "")).isdigit()
        and observed.get("worker_instance_id")
        and observed.get("bundle_digest") == bundle_digest
        and observed.get("library", {})
        .get("path", "")
        .startswith("/opt/physicsnemo-release/workflow/lib/libtransolver_workflow.so"),
        "resident process/library proof differs",
    )
    bindings = report.get("source_bindings", {})
    require(
        type(bindings.get("workspace_files")) is int
        and bindings["workspace_files"] > 0
        and type(bindings.get("installed_files")) is int
        and bindings["installed_files"] > 0
        and re.fullmatch(
            r"[a-f0-9]{64}", bindings.get("installed_inventory_sha256", "")
        )
        and set(bindings.get("binaries", {}))
        == {"native-worker", "worker-runtime", "inference_server"},
        "installed image source/binaries are unbound",
    )
    for record in [observed["library"], *bindings["binaries"].values()]:
        require(
            re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", ""))
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] > 0,
            "installed binary/library identity differs",
        )
    hardware = report.get("hardware", {})
    require(
        "H100" in hardware.get("name", "")
        and hardware.get("uuid", "").startswith("GPU-")
        and hardware.get("compute_capability") == [9, 0],
        "actual H100 observation missing",
    )
    command = report.get("service_command", [])
    require(
        len(command) == 3
        and Path(command[0]).name.startswith("python")
        and command[1:] == ["-m", "pnms_app.image_entrypoint"],
        "public shipped image entrypoint differs",
    )
    require(
        report.get("shutdown") == {"returncode": 0, "descendants_remaining": []},
        "shipped image stack shutdown differs",
    )
    return report


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    config = {
        "image": args.image,
        "source_sha": args.expected_source_sha,
        "inventory_sha256": args.expected_source_inventory_sha256,
        "bundle_digest": args.expected_bundle_digest,
        "backend": args.backend,
        "origin": ORIGIN,
        "benchmark_sha256": BENCHMARK_SHA256,
        "controller_sha256": identity(Path(__file__).read_bytes())["sha256"],
        "gates": GATES,
        "run_id": args.run_id,
        "output": f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
        "mount": args.mount_target,
        "repository": args.repository,
        "reference_fields": args.reference_fields,
        "input_root": args.input_root,
        "timeout": int(args.job_timeout) - 120,
    }
    runner = [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{int(args.job_timeout) - 60}s",
        "/opt/nvidia/nvidia_entrypoint.sh",
        "python3",
        "-B",
        "-c",
        REMOTE,
        json.dumps(config, sort_keys=True),
    ]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def extract_evidence(output):
    require(
        output.count(BEGIN) == 1 and output.count(END) == 1,
        "missing or ambiguous CFD image summary",
    )
    payload = output.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    transport = strict_json(payload)
    require(
        isinstance(transport.get("report_base64"), str)
        and len(transport["report_base64"]) <= 4 * ((MAX_REPORT + 2) // 3),
        "CFD image report transport exceeds limit",
    )
    raw = base64.b64decode(transport["report_base64"], validate=True)
    require(
        len(raw) <= MAX_REPORT
        and identity(raw) == {key: transport[key] for key in ("sha256", "size_bytes")},
        "CFD image report transport identity differs",
    )
    report = strict_json(raw)
    require(isinstance(report, dict), "CFD image report must be an object")
    return report, raw


def extract_summary(output):
    return extract_evidence(output)[0]


def run(args):
    require(
        not args.keep_job, "CFD image qualification requires owned finite-job cleanup"
    )
    require(not args.assets, "CFD image QA uses pinned reference/input paths")
    require(
        math.isfinite(args.job_timeout) and 180 < args.job_timeout <= 7200,
        "CFD image timeout must exceed 180 and be at most 7200 seconds",
    )
    for value in (args.expected_bundle_digest, args.expected_source_inventory_sha256):
        require(
            re.fullmatch(r"[a-f0-9]{64}", value),
            "full bundle/inventory SHA256 required",
        )
    for value in (args.repository, args.reference_fields, args.input_root):
        if value:
            path = PurePosixPath(value)
            require(
                path.is_absolute() and ".." not in path.parts,
                "CFD image paths must be absolute without traversal",
            )
    for value in (args.reference_fields, args.input_root):
        if value:
            require(
                PurePosixPath(value).is_relative_to(PurePosixPath(args.mount_target)),
                "CFD reference/input paths must be inside the mounted QA root",
            )
    root = args.artifact_dir.resolve() / args.run_id
    captured = {}

    def extract(output):
        report, raw = extract_evidence(output)
        captured.update(report=report, raw=raw)
        return report

    def validate(report, **selected):
        validate_summary(
            report,
            **selected,
            backend=args.backend,
            bundle_digest=args.expected_bundle_digest,
            source_inventory_sha256=args.expected_source_inventory_sha256,
        )
        require(
            captured.get("report") == report, "raw report belongs to another summary"
        )
        (root / "image-report.json").write_bytes(captured["raw"])

    code = lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=extract,
        summary_validator=validate,
        historical_first=True,
    )
    if root.exists():
        final = json.loads((root / "summary.json").read_bytes())
        resource_cleanup = {"status": "not-confirmed"}
        if final.get("job_id") and not final.get("cleanup_errors"):
            environment = {
                **os.environ,
                "LEPTON_WORKSPACE_ID": args.workspace_id,
                "NO_COLOR": "1",
                "TERM": "dumb",
            }
            if args.workspace_url:
                environment["LEPTON_WORKSPACE_URL"] = args.workspace_url
            audit_code, output = lifecycle.lep(
                ["job", "get", "-i", final["job_id"]],
                env=environment,
                timeout=min(args.command_timeout, 30),
            )
            (root / "job-absence.log").write_text(output)
            absent = audit_code != 0 and re.search(r"\b404\b", output) is not None
            resource_cleanup = {
                "status": "absent-404" if absent else "failed",
                "job_id": final["job_id"],
                "exit_code": audit_code,
            }
            if not absent:
                final.setdefault("cleanup_errors", []).append(
                    "owned job absence was not confirmed by 404"
                )
                final["status"] = "failed"
                code = code or 1
                lifecycle.write_json(root / "summary.json", final)
        lifecycle.write_json(
            root / "controller-identity.json",
            {
                "controller": identity(Path(__file__).read_bytes()),
                "wrapper": identity(REMOTE.encode()),
                "benchmark_sha256": BENCHMARK_SHA256,
            },
        )
        lifecycle.write_json(
            root / "image-proof.json",
            {
                "schema_version": 1,
                "status": "passed"
                if code == 0 and final.get("status") == "passed"
                else "failed",
                "run_id": args.run_id,
                "source_sha": args.expected_source_sha,
                "image_digest": args.image,
                "backend": args.backend,
                "bundle_digest": args.expected_bundle_digest,
                "source_inventory_sha256": args.expected_source_inventory_sha256,
                "cleanup_errors": final.get("cleanup_errors", []),
                "resource_cleanup": resource_cleanup,
                "report": identity(captured["raw"]) if "raw" in captured else None,
                "controller_sha256": identity(Path(__file__).read_bytes())["sha256"],
            },
        )
    return code


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
