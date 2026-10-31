"""Exact-image native service qualification; no model compilation is performed."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, "/opt/native-release/source")
from native_release import (
    contained,
    file_identity,
    image_digest,
    read_json,
    require,
    required_http_cases,
    verify_bundle,
    write_json,
)
from native_release_job import environment
from native_release_faults import run_idle_crash, reject_startup
from native_release_artifacts import seal_evidence
from qa.scripts.run_native_cfd_http_job import (
    HTTP,
    HTTPJob,
    RedisProbe,
    native_process_snapshot,
    pairs,
)

BEGIN = "NATIVE_RELEASE_BUILD_SUMMARY_BEGIN"
END = "NATIVE_RELEASE_BUILD_SUMMARY_END"


def submit_fixture(http, workflow, case, *, release_root=None, input_files=None):
    """Use the public upload route, binding transmitted bytes to sealed fixtures."""
    path = f"/v1/infer/{workflow}/run"
    files = case.get("files", {})
    if not files:
        return http.json(path, body={"parameters": case["request"]}, expected=202)
    rules = input_files or {}
    require(
        isinstance(files, dict) and set(files) <= set(rules),
        "fixture files must name server-declared ingress fields",
    )
    require(
        all(not rule["required"] or name in files for name, rule in rules.items()),
        "fixture is missing a required ingress file",
    )
    boundary = "physicsnemo-" + uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="parameters"\r\nContent-Type: application/json\r\n\r\n'.encode()
        + json.dumps(case["request"], allow_nan=False).encode()
        + b"\r\n"
    ]
    for name, descriptor in files.items():
        require(
            isinstance(name, str)
            and name.replace("_", "").replace("-", "").isalnum()
            and name.isascii(),
            "invalid fixture file field",
        )
        source = contained(Path(release_root) / "qualification", descriptor["source"])
        size = descriptor["size_bytes"]
        require(
            type(size) is int and 0 <= size <= rules[name]["max_size_bytes"],
            "fixture file violates server size policy",
        )
        with source.open("rb") as stream:
            data = stream.read(size + 1)
        require(
            len(data) == descriptor["size_bytes"]
            and hashlib.sha256(data).hexdigest() == descriptor["sha256"],
            "fixture input bytes differ from sealed identity",
        )
        media_type = descriptor["media_type"]
        require(
            media_type in rules[name]["media_types"]
            and "\r" not in media_type
            and "\n" not in media_type
            and len(data) <= rules[name]["max_size_bytes"],
            "fixture file violates server ingress policy",
        )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.bin"\r\nContent-Type: {media_type}\r\n\r\n'.encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        http.base_url + path,
        data=b"".join(parts),
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
        method="POST",
    )
    # An uncertain POST is never retried.
    with urllib.request.urlopen(request, timeout=60) as response:
        require(
            response.status == 202, "unexpected multipart fixture submission status"
        )
        return json.load(response)


def compare_expected(actual, expected, *, rtol=1e-5, atol=1e-6):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and set(expected) <= set(actual),
            "response fields differ from acceptance fixture",
        )
        for key, value in expected.items():
            compare_expected(actual[key], value, rtol=rtol, atol=atol)
    elif isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected),
            "response array shape differs from acceptance fixture",
        )
        for left, right in zip(actual, expected, strict=True):
            compare_expected(left, right, rtol=rtol, atol=atol)
    elif type(expected) in (int, float):
        require(
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            "response numerical value differs from acceptance fixture",
        )
    else:
        require(
            type(actual) is type(expected) and actual == expected,
            "response value differs from acceptance fixture",
        )


def verify_expected_artifacts(descriptors, expected):
    """Downloads also verify these identities, tying actual bytes to the fixture."""
    if expected is None:
        return
    require(isinstance(expected, dict), "expected_artifacts must be an identity map")
    actual = {item["name"]: item for item in descriptors}
    require(
        len(actual) == len(descriptors) and set(actual) == set(expected),
        "artifact inventory differs from authored acceptance fixture",
    )
    for name, identity in expected.items():
        require(
            isinstance(identity, dict) and set(identity) == {"sha256", "size_bytes"},
            "expected artifact requires SHA256 and byte size",
        )
        require(
            all(actual[name].get(key) == identity[key] for key in identity),
            "artifact bytes differ from authored acceptance fixture",
        )


class MemoryMonitor:
    """Sample pod RSS/cgroup and NVML device memory; sampled peaks are lower bounds."""

    def __init__(self, interval=0.5):
        self.interval, self.phase, self.samples = interval, "preflight", []
        self.done = threading.Event()
        self.thread = None
        self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
        require(self.nvml.nvmlInit_v2() == 0, "NVML monitoring initialization failed")
        self.handle = ctypes.c_void_p()
        require(
            self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0,
            "NVML monitoring device lookup failed",
        )

    def sample(self):
        class Memory(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong),
            ]

        memory = Memory()
        require(
            self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) == 0,
            "NVML monitoring sample failed",
        )
        rss = 0
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss += int(line.split()[1]) * 1024
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        record = {
            "monotonic_seconds": time.monotonic(),
            "phase": self.phase,
            "pod_process_rss_bytes": rss,
            "gpu_used_bytes": memory.used,
            "gpu_free_bytes": memory.free,
            "gpu_total_bytes": memory.total,
        }
        for name in ("current", "peak", "max"):
            path = Path("/sys/fs/cgroup") / ("memory." + name)
            if path.is_file():
                value = path.read_text().strip()
                record["cgroup_memory_" + name + "_bytes"] = (
                    None if value == "max" else int(value)
                )
        self.samples.append(record)

    def start(self):
        self.sample()

        def monitor():
            try:
                while not self.done.wait(self.interval):
                    self.sample()
            except BaseException as error:
                self.error = str(error)

        self.error = None
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=5)
        self.sample()
        self.nvml.nvmlShutdown()
        require(self.error is None, "resource monitoring failed: " + str(self.error))
        phases = {}
        for sample in self.samples:
            record = phases.setdefault(
                sample["phase"],
                {
                    "samples": 0,
                    "sampled_peak_gpu_used_bytes": 0,
                    "sampled_peak_pod_rss_bytes": 0,
                    "minimum_gpu_free_bytes": sample["gpu_free_bytes"],
                },
            )
            record["samples"] += 1
            record["sampled_peak_gpu_used_bytes"] = max(
                record["sampled_peak_gpu_used_bytes"], sample["gpu_used_bytes"]
            )
            record["sampled_peak_pod_rss_bytes"] = max(
                record["sampled_peak_pod_rss_bytes"], sample["pod_process_rss_bytes"]
            )
            record["minimum_gpu_free_bytes"] = min(
                record["minimum_gpu_free_bytes"], sample["gpu_free_bytes"]
            )
        return {
            "schema_version": 1,
            "requested_interval_seconds": self.interval,
            "scope": "all processes visible in this job PID namespace and NVML device0; sampled peaks are lower bounds, not allocator peaks",
            "phases": phases,
            "samples": self.samples,
        }


class ServiceJob(HTTPJob):
    def __init__(self, args, release, deployment):
        self.args, self.root, self.stack = args, args.output.resolve(), None
        self.release, self.deployment = release, deployment
        self.workflow = release["workflow"]
        self.summary = {
            "schema_version": 1,
            "status": "running",
            "run_id": args.run_id,
            "source_sha": release["source_sha"],
            "image_digest": args.image,
            "workflow": self.workflow,
            "bundle_digest": release["bundle_digest"],
            "target": release["target"],
            "expected_cases": required_http_cases(
                {"qualification": release["qualification"]}
            ),
            "cases": [],
            "compiled_models_rebuilt": False,
        }
        self.runs, self.results, self.monitor = [], {}, None

    def case(self, name, function):
        if self.monitor:
            self.monitor.phase = name
        return super().case(name, function)

    def ready(self, http, redis):
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            tail = self.stack_tail()
            require(self.stack.poll() is None, "service supervisor exited; " + tail)
            require(
                "entered FATAL state" not in tail[-12000:],
                "supervised service failed; " + tail,
            )
            try:
                health = http.json("/readyz")
                readiness = http.json(f"/v1/infer/{self.workflow}/readiness")
                registry = redis.registry()
                if (
                    health.get("status") == "ready"
                    and readiness.get("readiness", {}).get("ready") is True
                    and len(registry) == 1
                ):
                    worker = next(iter(registry.values()))
                    proof = (
                        worker.get("model_cache", {})
                        .get("warmup", {})
                        .get("self_test", {})
                    )
                    if (
                        worker.get("status") == "available"
                        and worker.get("workflow_id") == self.workflow
                        and proof.get("status") == "passed"
                        and proof.get("policy") == "expected-fp32-v1"
                        and proof.get("session_load_count")
                        == len(self.release["models"])
                        and isinstance(proof.get("models"), list)
                        and len(proof["models"]) == len(self.release["models"])
                        and {model.get("alias") for model in proof["models"]}
                        == set(self.release["models"])
                    ):
                        if "live-native-readiness-v1" in self.release.get(
                            "management_capabilities", []
                        ):
                            native = readiness.get("native", {})
                            require(
                                native.get("ready") is True
                                and "live-native-readiness-v1"
                                in native.get("capabilities", []),
                                "managed image has no live native readiness proof",
                            )
                            require(
                                native.get("release_image") == self.args.image
                                and native.get("bundle_digest")
                                == self.release["bundle_digest"]
                                and native.get("deployment_sha256")
                                == file_identity(
                                    Path(self.args.release_root) / "deployment.json"
                                )["sha256"],
                                "native readiness is not bound to the qualified image/bundle/config",
                            )
                            require(
                                native.get("worker_instance_id")
                                == worker.get("worker_instance_id")
                                and native.get("startup") == proof,
                                "native readiness differs from the live worker startup evidence",
                            )
                            startup_names = [
                                case["name"]
                                for case in self.deployment.get("startup_cases", [])
                            ]
                            if startup_names:
                                coverage = native.get("startup", {}).get("workflow", {})
                                require(
                                    coverage.get("status") == "passed"
                                    and [
                                        case.get("name")
                                        for case in coverage.get("cases", [])
                                    ]
                                    == startup_names
                                    and all(
                                        case.get("status") == "passed"
                                        for case in coverage["cases"]
                                    ),
                                    "managed workflow startup replay did not pass",
                                )
                            write_json(self.root / "startup-native.json", native)
                        return registry
            except (urllib.error.URLError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.5)
        raise TimeoutError(
            "native service startup/self-test deadline exceeded; " + self.stack_tail()
        )

    def request(self, http, label, case):
        root = self.root / label
        root.mkdir()
        submission = submit_fixture(
            http,
            self.workflow,
            case,
            release_root=getattr(self.args, "release_root", None),
            input_files=getattr(self, "deployment", {}).get("input_files", {}),
        )
        write_json(root / "submit.json", submission)
        require(
            submission.get("workflow") == self.workflow
            and submission.get("pipeline") == ["schedule", "execute", "results"],
            "request did not use native pipeline",
        )
        run_id = submission["run_id"]
        self.runs.append(run_id)
        deadline = time.monotonic() + self.args.request_timeout
        result = None
        while time.monotonic() < deadline:
            status = http.json(f"/v1/infer/{self.workflow}/{run_id}/status")
            if status.get("status") in {"succeeded", "failed", "cancelled"}:
                require(
                    status["status"] == "succeeded",
                    "native request failed: " + json.dumps(status),
                )
                try:
                    result = http.json(f"/v1/infer/{self.workflow}/{run_id}/results")
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 404:
                        raise
                    # ResultsRole status can precede its result write. Retry GET
                    # under this original deadline; never replay the POST.
            time.sleep(0.1)
        require(result is not None, "request/result publication deadline exceeded")
        # Preserve the actual wire response even when contract validation fails.
        write_json(root / "results.json", result)
        execution, payload = result.get("execution", {}), result.get("payload", {})
        require(
            execution.get("run_id") == run_id
            and execution.get("status") == "succeeded",
            "canonical execution envelope differs",
        )
        native = execution.get("native", {})
        require(
            native.get("source_sha") == self.release["source_sha"]
            and native.get("release_image") == self.args.image
            and native.get("runtime") == "cpp",
            "native result provenance differs",
        )
        require(
            native.get("metadata", {}).get("session_load_count")
            == len(self.release["models"]),
            "resident model alias coverage differs",
        )
        compare_expected(
            payload,
            case["expected"],
            rtol=case.get("rtol", 1e-5),
            atol=case.get("atol", 1e-6),
        )
        verify_expected_artifacts(
            execution.get("outputs", []), case.get("expected_artifacts")
        )
        self.results[run_id] = (label, result)
        return run_id, result

    def completion(self, redis):
        releases, results = {}, {}
        for stream, output in (("release", releases), ("results", results)):
            for event, raw in redis.call("XRANGE", "physicsnemo:" + stream, "-", "+"):
                fields = pairs(raw)
                require(
                    fields["run_id"] not in output,
                    "duplicate terminal completion or release",
                )
                output[fields["run_id"]] = (event, json.loads(fields["payload"]))
        require(
            set(releases) == set(results) == set(self.runs),
            "terminal event inventory differs from submitted runs",
        )
        row = next(iter(redis.registry().values()))
        deliveries = redis.call("XRANGE", row["stream"], "-", "+")
        require(
            len(deliveries) == len(self.runs), "unexpected scheduler dispatch count"
        )
        groups = [
            pairs(value) for value in redis.call("XINFO", "GROUPS", row["stream"])
        ]
        require(
            len(groups) == 1 and groups[0]["pending"] == 0,
            "execute deliveries remain pending",
        )
        for message, raw in deliveries:
            fields = pairs(raw)
            original = json.loads(fields["payload"])
            released = releases[fields["run_id"]][1]
            require(
                released.get("resource_id") == original.get("resource_id")
                and released.get("memory_mb") == original.get("memory_mb"),
                "release differs from original scheduler reservation",
            )
            stream, group = row["stream"], groups[0]["name"]
            key = f"native-attempt:{len(stream)}:{stream}:{len(group)}:{group}:{len(message)}:{message}"
            marker = pairs(redis.call("HGETALL", key))
            require(
                marker.get("state") == "committed"
                and json.loads(marker.get("ids", "[]"))
                == [releases[fields["run_id"]][0], results[fields["run_id"]][0]],
                "guarded completion marker differs from accepted events",
            )
        # ResultsRole can publish the API result before Scheduler consumes the
        # independently queued release. Durable completion proofs above must
        # already match; only the reservation counter is allowed to settle.
        deadline = time.monotonic() + 30
        while redis.call("HGET", "scheduler:active_reserved_memory_mb", "0") not in (
            None,
            "0",
        ):
            remaining = deadline - time.monotonic()
            require(
                remaining > 0,
                "scheduler reservation did not settle before completion deadline",
            )
            time.sleep(min(0.1, remaining))
        require(redis.call("XLEN", "physicsnemo:dlq") == 0, "dead-letter event")
        write_json(
            self.root / "completion.json",
            {
                "runs": self.runs,
                "release_count": len(releases),
                "result_count": len(results),
                "pending": 0,
            },
        )


def run(args):
    root = args.release_root
    release = verify_bundle(root)
    deployment = read_json(root / "deployment.json")
    for name in ("startup_timeout", "request_timeout"):
        if getattr(args, name, None) is None:
            setattr(args, name, deployment.get(name + "_seconds", 300))
    args.output.mkdir(parents=True, exist_ok=False)
    job = ServiceJob(args, release, deployment)
    http, redis = HTTP("http://127.0.0.1:8080"), RedisProbe()
    try:

        def preflight():
            require(
                os.environ.get("SOURCE_SHA") == release["source_sha"],
                "image source differs from release inventory",
            )
            image_digest(args.image)
            for port in (6379, 8080):
                with socket.socket() as probe:
                    require(
                        probe.connect_ex(("127.0.0.1", port)) != 0,
                        "service QA must own fresh Redis/API ports",
                    )
            job.summary["environment"] = environment(
                release["target"], backend=release.get("backend", "aoti")
            )

        job.case("preflight", preflight)
        job.monitor = MemoryMonitor()
        job.monitor.start()

        def startup():
            job.start()
            registry = job.ready(http, redis)
            write_json(job.root / "startup.json", registry)
            return registry

        registry = job.case("startup", startup)
        worker = next(iter(registry.values()))
        before = native_process_snapshot(worker["pid"])
        fixtures = release["qualification"]["cases_document"]["cases"]
        cases = {case["name"]: case for case in fixtures}
        for name in release["qualification"]["required_cases"]:
            job.case(
                "fixture." + name,
                lambda name=name: job.request(http, "fixture-" + name, cases[name]),
            )
        first = cases[release["qualification"]["required_cases"][0]]

        def malformed():
            try:
                # Keep the supported representation and valid sealed input
                # files so this gate tests malformed canonical parameters,
                # rather than an unsupported Content-Type or missing upload.
                submit_fixture(
                    http,
                    job.workflow,
                    {**first, "request": []},
                    release_root=args.release_root,
                    input_files=deployment.get("input_files", {}),
                )
            except urllib.error.HTTPError as error:
                require(
                    error.code in {400, 422}, "malformed request failed unexpectedly"
                )
                return
            raise ValueError("malformed native request was accepted")

        job.case("malformed_request", malformed)

        def overlap():
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(job.request, http, "overlap-" + str(index), first)
                    for index in range(2)
                ]
                values = [future.result() for future in futures]
            attempts = [
                result[1]["execution"]["native"]["attempt"] for result in values
            ]
            require(
                len(set(attempts)) == 2, "overlapping requests shared output attempt"
            )
            current = native_process_snapshot(worker["pid"])
            require(
                before["pid"] == current["pid"]
                and set(before["native_thread_ids"])
                & set(current["native_thread_ids"]),
                "resident host changed between requests",
            )

        job.case("overlap", overlap)

        job.case(
            "worker_restart", lambda: run_idle_crash(job, http, redis, first, worker)
        )
        job.case("redis_completion", lambda: job.completion(redis))

        def downloads():
            inventory = []
            for run_id, (label, result) in job.results.items():
                for descriptor in result["execution"].get("outputs", []):
                    query = urllib.parse.urlencode({"artifact": descriptor["name"]})
                    record = http.download(
                        f"/v1/infer/{job.workflow}/{run_id}/results?{query}",
                        job.root / label / "downloads" / descriptor["name"],
                        descriptor,
                    )
                    inventory.append(
                        {"run_id": run_id, "name": descriptor["name"], **record}
                    )
            write_json(
                job.root / "artifact-downloads.json",
                {
                    "executed": True,
                    "artifacts": inventory,
                    "scope": "JSON-only adapters can validly return zero artifact files",
                },
            )

        job.case("artifacts", downloads)

        job.case(
            "admission_rejection",
            lambda: reject_startup(job, redis, deployment, kind="gpu"),
        )
        job.case(
            "tampered_model",
            lambda: reject_startup(job, redis, deployment, kind="model"),
        )
        job.case("shutdown", job.stop)
        require(
            [case["name"] for case in job.summary["cases"]]
            == job.summary["expected_cases"]
            and all(case["status"] == "passed" for case in job.summary["cases"]),
            "required native qualification inventory differs",
        )
        job.summary["status"] = "passed"
        return job.summary
    except BaseException as error:
        job.summary.update(
            status="failed", error=type(error).__name__ + ": " + str(error)
        )
        print(
            "NATIVE_RELEASE_STACK_FAILURE "
            + json.dumps({"stack_tail": job.stack_tail()}),
            flush=True,
        )
        raise
    finally:
        if job.stack is not None:
            try:
                job.stop()
            except BaseException as error:
                job.summary.update(status="failed", cleanup_error=str(error))
        if job.monitor is not None:
            try:
                measurements = job.monitor.stop()
                write_json(job.root / "memory.json", measurements)
                job.summary["memory_evidence"] = file_identity(job.root / "memory.json")
                job.summary["memory"] = {
                    key: value
                    for key, value in measurements.items()
                    if key != "samples"
                }
            except BaseException as error:
                job.summary.update(status="failed", monitoring_error=str(error))
        job.save()
        evidence_error = None
        try:
            seal_evidence(job.root, job.summary)
        except BaseException as error:
            evidence_error = error
            job.summary.update(status="failed", evidence_error=str(error))
            job.save()
        print(BEGIN, flush=True)
        print(json.dumps(job.summary, sort_keys=True), flush=True)
        print(END, flush=True)
        if evidence_error is not None:
            raise RuntimeError(
                "native qualification evidence sealing failed"
            ) from evidence_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--release-root", type=Path, default=Path("/opt/physicsnemo-release")
    )
    parser.add_argument(
        "--entrypoint", type=Path, default=Path("/app/scripts/entrypoint.sh")
    )
    parser.add_argument("--startup-timeout", type=int)
    parser.add_argument("--request-timeout", type=int)
    parser.add_argument("--shutdown-timeout", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
