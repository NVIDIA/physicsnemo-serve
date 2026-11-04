#!/usr/bin/env python3
"""Qualify a self-contained native release image through its real Rust HTTP API.

Run inside the final immutable image on Lepton H100 with the earlier qualified
reference mounted. No compilation is performed. The normal entrypoint starts
Redis, scheduler, native worker, API and results; two sequential requests prove
reuse and exercise actual HTTP downloads before field/metric comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from qa.native_inference.cfd_contract import (
    CASE_POINT_COUNTS,
    CASE_PINS,
    FieldContractError,
    inference_seed,
    verify_file,
)
from qa.native_inference.cfd_native import package_native_fields
from qa.native_inference.cfd_reference import reference_model_asset_arguments
from qa.scripts.cfd_parity_contract import (
    read_json_object,
    sha256_file,
    write_json_atomic,
)
from qa.scripts.run_native_cfd_job import Job, resource_envelope


WORKFLOW = "physicsnemo-cfd-transolver-native"
SUMMARY_BEGIN = "NATIVE_CFD_HTTP_QA_SUMMARY_BEGIN"
SUMMARY_END = "NATIVE_CFD_HTTP_QA_SUMMARY_END"
CASES = [
    "preflight",
    "start_stack",
    "readiness",
    "first_request",
    "second_request",
    "redis_completion",
    "artifact_downloads",
    "native_fields",
    "shared_evaluator",
    "parity",
    "shutdown",
]
ARTIFACTS = {
    "physical_fields": "physical.bin",
    "standardized_fields": "standardized.bin",
    "native_metadata": "metadata.json",
    "fx.f32": "inputs/fx.f32",
    "embedding.f32": "inputs/embedding.f32",
    "permutation.i64": "inputs/permutation.i64",
    "source_cell_ids.i64": "inputs/source_cell_ids.i64",
}


def pairs(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, list) or len(value) % 2:
        raise FieldContractError("Redis returned malformed field/value pairs")
    return dict(zip(value[::2], value[1::2], strict=True))


def native_process_snapshot(pid, *, proc_root=Path("/proc")):
    process = proc_root / str(pid)
    executable = os.readlink(process / "exe")
    if Path(executable).name != "native-worker":
        raise FieldContractError(
            "registered native worker is not the native-worker executable"
        )
    threads = sorted(
        int(path.parent.name)
        for path in (process / "task").glob("*/comm")
        if path.read_text().strip() == "native-workflow"
    )
    if not threads:
        raise FieldContractError(
            "native worker is missing a persistent C++ host thread candidate"
        )
    children = process / "task" / str(pid) / "children"
    if children.is_file() and children.read_text().strip():
        raise FieldContractError("native worker unexpectedly launched a child process")
    return {"pid": pid, "executable": executable, "native_thread_ids": threads}


def assert_persistent_native_process(initial, current):
    # CUDA/LibTorch helper threads may inherit native-workflow as their comm.
    # Names identify candidates, not a unique host. Registry/model-session
    # checks independently establish reuse; retain a live native TID as well.
    retained = sorted(
        set(initial["native_thread_ids"]) & set(current["native_thread_ids"])
    )
    if (
        initial["pid"] != current["pid"]
        or initial["executable"] != current["executable"]
        or not retained
    ):
        raise FieldContractError(
            "native process or all native thread candidates changed"
        )
    return retained


def assert_same_worker_registry(initial, current):
    if set(initial) != set(current):
        raise FieldContractError("native worker registry slot changed")
    keys = (
        "pid",
        "stream",
        "worker_instance_id",
        "workflow_id",
        "executor_class",
        "device_index",
    )
    for slot, before in initial.items():
        after = current[slot]
        if any(before.get(key) != after.get(key) for key in keys):
            raise FieldContractError(
                "native worker restarted between sequential HTTP requests"
            )
        if before.get("model_cache", {}).get("warmup", {}).get(
            "self_test"
        ) != after.get("model_cache", {}).get("warmup", {}).get("self_test"):
            raise FieldContractError("resident model startup evidence changed")


class RedisProbe:
    def call(self, *args):
        result = subprocess.run(
            ["redis-cli", "--json", "-h", "127.0.0.1", "-p", "6379", *map(str, args)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def registry(self):
        rows = pairs(self.call("HGETALL", "gpu:registry"))
        return {key: json.loads(value) for key, value in rows.items()}


class HTTP:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def json(self, path, *, body=None, expected=200):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if body is not None else "GET",
        )
        # urlopen has no application-level retry. A timed-out POST is uncertain;
        # never resubmit it, since the API currently has no idempotency key.
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != expected:
                raise FieldContractError(
                    f"unexpected HTTP status {response.status} for {path}"
                )
            return json.load(response)

    def download(self, path, destination, descriptor):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            urllib.request.urlopen(self.base_url + path, timeout=120) as response,
            destination.open("xb") as output,
        ):
            digest, count = hashlib.sha256(), 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > descriptor["size_bytes"]:
                    raise FieldContractError("HTTP artifact exceeds advertised size")
                digest.update(chunk)
                output.write(chunk)
        if (
            count != descriptor["size_bytes"]
            or digest.hexdigest() != descriptor["sha256"]
        ):
            raise FieldContractError(
                "HTTP artifact bytes differ from advertised digest/size"
            )
        return {"sha256": digest.hexdigest(), "size_bytes": count}


def assert_result(
    result,
    *,
    run_id,
    source_sha,
    image,
    artifact_root,
    capture_inputs,
    compilation_source_sha=None,
    model_build_sources=None,
    backend="aoti",
):
    execution, payload = result.get("execution", {}), result.get("payload", {})
    if (
        execution.get("run_id") != run_id
        or execution.get("workflow") != WORKFLOW
        or execution.get("status") != "succeeded"
    ):
        raise FieldContractError(
            "HTTP execution did not succeed for the submitted native workflow"
        )
    native = payload.get("native", {})
    if model_build_sources:
        expected_sources = set(model_build_sources.values())
        common_source = (
            next(iter(expected_sources)) if len(expected_sources) == 1 else None
        )
        if (
            native.get("model_build_sources") != model_build_sources
            or native.get("model_build_source_sha") != common_source
        ):
            raise FieldContractError(
                "HTTP alias compilation source provenance differs from selected packages"
            )
    elif (
        compilation_source_sha is not None
        and native.get("model_build_source_sha") != compilation_source_sha
    ):
        raise FieldContractError(
            "HTTP model compilation source differs from selected qualified packages"
        )
    for key, value in {
        "runtime": "cpp",
        "backend": backend,
        "case_id": "run_1",
        "run_seed": 42,
        "permutation_seed": inference_seed(42, "run_1"),
        "source_sha": source_sha,
        "release_image": image,
        "mesh_sha256": CASE_PINS["run_1"]["sha256"],
        "geometry_sha256": CASE_PINS["run_1"]["geometry_sha256"],
    }.items():
        if native.get(key) != value:
            raise FieldContractError(f"HTTP native provenance {key} mismatch")
    attempt = native.get("attempt")
    if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f-]{36}", attempt):
        raise FieldContractError("HTTP native attempt identity is missing")
    root = (
        artifact_root
        / f"native-{hashlib.sha256(run_id.encode()).hexdigest()}"
        / attempt
    )
    outputs = execution.get("outputs", [])
    names = [item.get("name") for item in outputs]
    expected = set(ARTIFACTS if capture_inputs else list(ARTIFACTS)[:3])
    if set(names) != expected or len(names) != len(expected):
        raise FieldContractError(
            "HTTP native artifact inventory is incomplete or duplicated"
        )
    count = CASE_POINT_COUNTS["run_1"]
    sizes = {
        "physical_fields": count * 16,
        "standardized_fields": count * 16,
        "fx.f32": count * 8,
        "embedding.f32": count * 24,
        "permutation.i64": count * 8,
        "source_cell_ids.i64": count * 8,
    }
    for item in outputs:
        name = item["name"]
        path = Path(item["storage_path"])
        if path != root / ARTIFACTS[name]:
            raise FieldContractError(
                "HTTP artifact escaped its immutable native attempt"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise FieldContractError("HTTP artifact digest is invalid")
        if name in sizes and item.get("size_bytes") != sizes[name]:
            raise FieldContractError(
                "HTTP native array size differs from full case shape"
            )
    if payload.get("output_path") not in (
        None,
        str(root / "physical.bin"),
    ) or execution.get("output_path") != str(root / "physical.bin"):
        raise FieldContractError("HTTP primary output differs from physical fields")
    return root, {item["name"]: item for item in outputs}, native["metadata"]


def assert_redis_completion(redis, *, prefix, runs, registry):
    releases = redis.call("XRANGE", prefix + "release", "-", "+")
    results = redis.call("XRANGE", prefix + "results", "-", "+")
    release_by_run, results_by_run = {}, {}
    for destination, rows in ((release_by_run, releases), (results_by_run, results)):
        for event_id, raw in rows:
            fields = pairs(raw)
            run_id = fields["run_id"]
            if run_id in destination:
                raise FieldContractError(
                    "duplicate terminal result/reservation release"
                )
            destination[run_id] = (event_id, json.loads(fields["payload"]))
    if set(release_by_run) != set(runs) or set(results_by_run) != set(runs):
        raise FieldContractError(
            "terminal stream inventory does not cover both HTTP requests"
        )
    for run_id in runs:
        release = release_by_run[run_id][1]
        if (
            release.get("resource_id") != 0
            or release.get("memory_mb") != 60000
            or release.get("status") != "completed"
        ):
            raise FieldContractError(
                "native completion did not release the original reservation"
            )
    if redis.call("HGET", "scheduler:active_reserved_memory_mb", "0") not in (
        None,
        "0",
    ):
        raise FieldContractError("scheduler still holds an active reservation")
    if redis.call("XLEN", prefix + "dlq"):
        raise FieldContractError("native HTTP workflow emitted a dead-letter message")
    if len(registry) != 1:
        raise FieldContractError("native HTTP job requires one registered worker")
    worker = next(iter(registry.values()))
    stream = worker["stream"]
    groups = [pairs(row) for row in redis.call("XINFO", "GROUPS", stream)]
    if len(groups) != 1 or groups[0].get("pending") != 0:
        raise FieldContractError("native execution delivery remains pending")
    group = groups[0]["name"]
    executions = redis.call("XRANGE", stream, "-", "+")
    if len(executions) != len(runs):
        raise FieldContractError("scheduler dispatched an unexpected execution count")
    for message_id, raw in executions:
        run_id = pairs(raw)["run_id"]
        key = f"native-attempt:{len(stream)}:{stream}:{len(group)}:{group}:{len(message_id)}:{message_id}"
        marker = pairs(redis.call("HGETALL", key))
        if marker.get("state") != "committed" or json.loads(
            marker.get("ids", "[]")
        ) != [release_by_run[run_id][0], results_by_run[run_id][0]]:
            raise FieldContractError(
                "native durable completion marker differs from terminal events"
            )
    return {
        "release_events": len(releases),
        "result_events": len(results),
        "native_pending": 0,
        "active_reserved_memory_mb": 0,
        "worker_pid": worker["pid"],
    }


class HTTPJob:
    command = Job.command

    def __init__(self, args):
        self.args, self.root = args, args.output.resolve()
        self.summary = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.source_sha,
            "image_digest": args.image,
            "status": "running",
            "qualification": "native-http-fields-and-metrics-v1",
            "expected_cases": CASES,
            "cases": [],
            "compiled_models_rebuilt": False,
            "qa_driver_sha256": sha256_file(Path(__file__)),
            "resources": resource_envelope(),
        }
        self.stack = None

    def save(self):
        write_json_atomic(self.root / "summary.json", self.summary)

    def case(self, name, function):
        record = {"name": name, "status": "running"}
        self.summary["cases"].append(record)
        start = time.monotonic()
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
                f"native HTTP: {name} {record['status']} ({record['seconds']:.1f}s)",
                flush=True,
            )

    def preflight(self):
        if not re.fullmatch(
            r".+@sha256:[0-9a-f]{64}", self.args.image
        ) or not re.fullmatch(r"[0-9a-f]{40}", self.args.source_sha):
            raise FieldContractError(
                "actual immutable release image and source SHA are required"
            )
        if os.environ.get("SOURCE_SHA") != self.args.source_sha:
            raise FieldContractError(
                "image SOURCE_SHA differs from submitted release revision"
            )
        root = self.args.release_root.resolve(strict=True)
        deployment = read_json_object(root / "deployment.json")
        backend = deployment.get("backend", "aoti")
        if backend not in {"aoti", "tensorrt"}:
            raise FieldContractError(
                "installed CFD deployment backend or executor differs"
            )
        self.backend = backend
        self.executor_class = deployment["executor_class"]
        inventory = read_json_object(root / "release-inventory.json")
        receipt = root / "build-receipt.json"
        qualified = read_json_object(receipt)
        if (
            qualified.get("backend", "aoti") != backend
            or inventory.get("backend", "aoti") != backend
        ):
            raise FieldContractError(
                "installed CFD receipt backend differs from deployment"
            )
        self.summary["backend"] = backend
        compiled = qualified.get("source_sha")
        model_sources = deployment.get("model_build_sources")
        if model_sources:
            if (
                not isinstance(model_sources, dict)
                or inventory.get("model_build_sources") != model_sources
                or not all(
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value)
                    for value in model_sources.values()
                )
            ):
                raise FieldContractError(
                    "installed actual alias compilation sources differ"
                )
            actual = set(model_sources.values())
            self.summary["model_build_sources"] = model_sources
            self.summary["compilation_source_sha"] = (
                next(iter(actual)) if len(actual) == 1 else None
            )
        else:
            if (
                not re.fullmatch(r"[0-9a-f]{40}", str(compiled))
                or inventory.get("compilation_source_sha") != compiled
                or os.environ.get("MODEL_BUILD_SOURCE_SHA") != compiled
            ):
                raise FieldContractError(
                    "image model compilation source differs from qualified receipt"
                )
            self.summary["compilation_source_sha"] = compiled
        if inventory.get("source_sha") != self.args.source_sha or inventory.get(
            "compile_evidence_sha256"
        ) != sha256_file(receipt):
            raise FieldContractError(
                "installed release inventory is not bound to its source/receipt"
            )
        if (
            qualified.get("status") != "passed"
            or qualified.get("qualification") != "full-run1-fields-and-metrics-v1"
            or qualified.get("summary_sha256") != sha256_file(root / "summary.json")
        ):
            raise FieldContractError(
                "image does not contain a fully qualified compile receipt"
            )
        for asset in deployment["model_assets"]:
            path = Path(asset["path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise FieldContractError(
                    "compiled model/statistics are not self-contained in release image"
                )
            verify_file(path, asset)
        config = read_json_object(root / "worker_runtime_config.json")
        if config["roles"]["scheduler"]["config"].get("batching_enabled") is not False:
            raise FieldContractError(
                "native scheduler request batching must be disabled"
            )
        for port in (6379, 8080):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise FieldContractError(
                        "HTTP QA must own its fresh local service and Redis ports"
                    )
        return deployment

    def start(self):
        log = (self.root / "stack.log").open("w")
        environment = dict(os.environ)
        environment.update(
            RELEASE_IMAGE_DIGEST=self.args.image, HEALTH_STUB_ENABLED="false"
        )
        self.stack = subprocess.Popen(
            [str(self.args.entrypoint)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        log.close()

    def stop(self):
        if self.stack is None:
            return
        try:
            try:
                os.killpg(self.stack.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.stack.wait(timeout=self.args.shutdown_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.stack.pid, signal.SIGKILL)
            self.stack.wait()
            raise FieldContractError(
                "native stack did not drain during graceful shutdown"
            )
        finally:
            subprocess.run(
                ["redis-cli", "-h", "127.0.0.1", "-p", "6379", "SHUTDOWN", "NOSAVE"],
                capture_output=True,
                timeout=30,
            )
            self.stack = None

    def ready(self, http, redis):
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            tail = self.stack_tail()
            if self.stack.poll() is not None:
                self.startup_failure("native stack exited before readiness", tail)
            fatal = re.search(
                r"gave up: (native-worker|inference-server|results|scheduler) entered FATAL state",
                tail,
            )
            if fatal:
                self.startup_failure(
                    f"{fatal.group(1)} entered supervisor FATAL state before readiness",
                    tail,
                )
            try:
                health = http.json("/readyz")
                readiness = http.json(f"/v1/infer/{WORKFLOW}/readiness")
                registry = redis.registry()
                if (
                    health.get("status") == "ready"
                    and readiness.get("readiness", {}).get("ready") is True
                    and len(registry) == 1
                ):
                    worker = next(iter(registry.values()))
                    if (
                        worker.get("executor_class")
                        == getattr(self, "executor_class", "native.transolver.aoti")
                        and worker.get("status") == "available"
                    ):
                        write_json_atomic(
                            self.root / "readiness.json",
                            {"http": readiness, "registry": registry},
                        )
                        return registry
            except (urllib.error.URLError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(2)
        self.startup_failure(
            "native HTTP/worker readiness timed out", self.stack_tail()
        )

    def stack_tail(self):
        try:
            with (self.root / "stack.log").open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 16384))
                return log.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def startup_failure(self, message, tail):
        # The entrypoint can stay alive after supervisord exhausts worker
        # restarts. Expose the cause in historical job logs before cleanup.
        print(
            "NATIVE_CFD_STACK_FAILURE "
            + json.dumps({"error": message, "stack_tail": tail}),
            flush=True,
        )
        raise FieldContractError(message + "; inspect stack.log and job stdout")

    def request(self, http, label, capture):
        root = self.root / label
        root.mkdir()
        submitted = http.json(
            f"/v1/infer/{WORKFLOW}/run",
            body={
                "parameters": {
                    "case_id": "run_1",
                    "seed": 42,
                    "capture_inputs": capture,
                }
            },
            expected=202,
        )
        write_json_atomic(root / "submit.json", submitted)
        if (
            submitted.get("pipeline") != ["schedule", "execute", "results"]
            or submitted.get("workflow") != WORKFLOW
        ):
            raise FieldContractError(
                "HTTP submission did not choose explicit native pipeline"
            )
        run_id = submitted["run_id"]
        deadline = time.monotonic() + self.args.request_timeout
        while time.monotonic() < deadline:
            status = http.json(f"/v1/infer/{WORKFLOW}/{run_id}/status")
            write_json_atomic(root / "latest_status.json", status)
            if status.get("status") in ("succeeded", "failed", "cancelled"):
                if status["status"] != "succeeded":
                    raise FieldContractError(f"native HTTP run failed: {status}")
                try:
                    result = http.json(f"/v1/infer/{WORKFLOW}/{run_id}/results")
                except urllib.error.HTTPError as error:
                    if error.code != 404:
                        raise
                    time.sleep(0.1)
                    continue
                write_json_atomic(root / "results.json", result)
                return run_id, result
            time.sleep(2)
        raise FieldContractError("native HTTP request did not reach terminal status")


def run(args, *, job_factory=HTTPJob, qualify_service=None):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(exist_ok=False)
    job, http, redis = job_factory(args), HTTP("http://127.0.0.1:8080"), RedisProbe()
    try:
        model_asset_args = []

        def preflight():
            model_asset_args.extend(reference_model_asset_arguments(args.reference))
            return job.preflight()

        deployment = job.case("preflight", preflight)
        job.case("start_stack", job.start)
        registry = job.case("readiness", lambda: job.ready(http, redis))
        native_process = native_process_snapshot(next(iter(registry.values()))["pid"])
        write_json_atomic(job.root / "native-process.json", native_process)
        first = job.case("first_request", lambda: job.request(http, "first", True))
        second = job.case("second_request", lambda: job.request(http, "second", False))
        runs = [first[0], second[0]]
        assert_same_worker_registry(registry, redis.registry())
        after = native_process_snapshot(native_process["pid"])
        retained = assert_persistent_native_process(native_process, after)
        write_json_atomic(
            job.root / "native-process-after.json",
            {**after, "retained_native_thread_ids": retained},
        )
        deadline = time.monotonic() + 30

        def completed():
            while True:
                try:
                    result = assert_redis_completion(
                        redis,
                        prefix=os.environ.get("REDIS_STREAM_PREFIX", "physicsnemo:"),
                        runs=runs,
                        registry=registry,
                    )
                    write_json_atomic(job.root / "redis-completion.json", result)
                    return result
                except FieldContractError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)

        job.case("redis_completion", completed)
        prepared = []

        def downloads():
            for label, (run_id, result), capture in (
                ("first", first, True),
                ("second", second, False),
            ):
                root, artifacts, metadata = assert_result(
                    result,
                    run_id=run_id,
                    source_sha=args.source_sha,
                    image=args.image,
                    artifact_root=Path(deployment["artifact_root"]),
                    capture_inputs=capture,
                    backend=deployment.get("backend", "aoti"),
                    compilation_source_sha=job.summary["compilation_source_sha"],
                    model_build_sources=job.summary.get("model_build_sources"),
                )
                records = {}
                for name, descriptor in artifacts.items():
                    query = urllib.parse.urlencode({"artifact": name})
                    records[name] = http.download(
                        f"/v1/infer/{WORKFLOW}/{run_id}/results?{query}",
                        job.root / label / "downloads" / ARTIFACTS[name],
                        descriptor,
                    )
                    verify_file(root / ARTIFACTS[name], descriptor)
                downloaded_metadata = read_json_object(
                    job.root / label / "downloads/metadata.json"
                )
                if downloaded_metadata != metadata:
                    raise FieldContractError(
                        "HTTP metadata download differs from persisted result provenance"
                    )
                prepared.append((root, artifacts, metadata))
                write_json_atomic(job.root / label / "downloads.json", records)
            for name in ("physical_fields", "standardized_fields"):
                if prepared[0][1][name]["sha256"] != prepared[1][1][name]["sha256"]:
                    raise FieldContractError(
                        "sequential same-seed native HTTP predictions differ"
                    )
            for key in ("initialization_ms", "session_load_count"):
                if prepared[0][2].get(key) != prepared[1][2].get(key):
                    raise FieldContractError(
                        "native HTTP requests did not reuse resident model initialization"
                    )
            if prepared[0][0] == prepared[1][0] or first[0] == second[0]:
                raise FieldContractError(
                    "native HTTP requests reused an artifact attempt"
                )

        job.case("artifact_downloads", downloads)
        profile = read_json_object(args.profile)
        native_root = prepared[0][0]
        candidate = native_root / "fields.json"
        job.case(
            "native_fields",
            lambda: package_native_fields(
                profile=profile,
                metadata_path=native_root / "metadata.json",
                receipt_path=args.release_root / "build-receipt.json",
                input_dumps=native_root / "inputs",
                output_path=candidate,
                image=args.image,
                source_sha=args.source_sha,
                compilation_source_sha=job.summary["compilation_source_sha"],
            ),
        )
        job.summary["candidate_fields"] = str(candidate)
        evaluation = job.root / "evaluation"
        job.case(
            "shared_evaluator",
            lambda: job.command(
                "shared-evaluator",
                [
                    sys.executable,
                    "-m",
                    "qa.scripts.run_native_cfd_reference",
                    "evaluate",
                    *model_asset_args,
                    "--profile",
                    args.profile,
                    "--request",
                    args.request,
                    "--input-root",
                    args.input_root,
                    "--image",
                    args.image,
                    "--source-sha",
                    args.source_sha,
                    "--fields",
                    candidate,
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
                    args.reference,
                    "--candidate",
                    candidate,
                    "--candidate-evaluation",
                    evaluation / "evaluation.json",
                    "--output",
                    job.root / "comparison.json",
                ],
            ),
        )
        if qualify_service is not None:
            qualify_service(job, http, redis, deployment, registry)
        for asset in deployment["model_assets"]:
            verify_file(asset["path"], asset)
        job.case("shutdown", job.stop)
        if [record["name"] for record in job.summary["cases"]] != job.summary[
            "expected_cases"
        ] or any(record["status"] != "passed" for record in job.summary["cases"]):
            raise FieldContractError(
                "native HTTP qualification case inventory did not pass"
            )
        job.summary["status"] = "passed"
        job.save()
        return 0
    except BaseException as error:
        job.summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        job.save()
        raise
    finally:
        if job.stack is not None:
            try:
                job.stop()
            except BaseException as cleanup_error:
                job.summary.update(
                    status="failed",
                    cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
                job.save()
        print(SUMMARY_BEGIN, flush=True)
        print(json.dumps(job.summary, allow_nan=False, sort_keys=True), flush=True)
        print(SUMMARY_END, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "profile", "request", "reference", "input-root"):
        result.add_argument(f"--{name}", type=Path, required=True)
    for name in ("run-id", "image", "source-sha"):
        result.add_argument(f"--{name}", required=True)
    result.add_argument(
        "--release-root", type=Path, default=Path("/opt/physicsnemo-native")
    )
    result.add_argument(
        "--entrypoint", type=Path, default=Path("/app/scripts/entrypoint.sh")
    )
    result.add_argument("--startup-timeout", type=int, default=1800)
    result.add_argument("--request-timeout", type=int, default=3600)
    result.add_argument("--shutdown-timeout", type=int, default=300)
    result.add_argument("--command-timeout", type=int, default=7200)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
