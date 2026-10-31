"""Full CFD image scientific qualification plus supervised service fault gates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time
import urllib.error

from native_release import (
    file_identity,
    read_json,
    require,
    required_http_cases,
    verify_bundle,
    write_json,
)
from native_release_artifacts import seal_evidence
from native_release_faults import run_idle_crash, run_active_crash, reject_startup
from native_release_http_job import MemoryMonitor, ServiceJob
from native_release_job import environment
from qa.scripts import run_native_cfd_http_job as scientific

BEGIN = "NATIVE_RELEASE_BUILD_SUMMARY_BEGIN"
END = "NATIVE_RELEASE_BUILD_SUMMARY_END"


def verify_reference(record):
    require(
        isinstance(record, dict)
        and file_identity(Path(record["path"]))
        == {key: record[key] for key in ("sha256", "size_bytes")},
        "qualified full CFD reference bytes changed or are unavailable on this node",
    )
    return Path(record["path"])


def verify_cfd_startup(proof, *, backend="aoti"):
    require(
        proof.get("status") == "passed"
        and proof.get("policy") == "expected-fp32-v1"
        and proof.get("session_load_count") == 2,
        "CFD readiness requires trusted expected-output model self-tests",
    )
    models = proof.get("models", [])
    require(
        len(models) == 2
        and {model.get("point_count") for model in models} == {60000, 8095},
        "CFD golden self-test must cover the full and exact tail sessions",
    )
    for model in models:
        comparison = model.get("comparison") or {}
        error = comparison.get("max_absolute_error")
        require(
            model.get("shape") == [1, model["point_count"], 4]
            and model.get("dtype") == "float32"
            and model.get("finite") is True
            and model.get("backend") == backend
            and comparison.get("rtol") == 1e-5
            and comparison.get("atol") == 1e-6
            and type(error) in (int, float)
            and math.isfinite(error)
            and error >= 0,
            "CFD golden self-test shape, numerical policy or comparison evidence differs",
        )
    return proof


class CFDJob(scientific.HTTPJob):
    def __init__(self, args, release):
        super().__init__(args)
        self.release = release
        self.workflow = release["workflow"]
        self.runs, self.results = [], {}
        self.monitor = None
        self.field_hashes = None
        self.summary.update(
            workflow=self.workflow,
            bundle_digest=release["bundle_digest"],
            target=release["target"],
            expected_cases=required_http_cases(release),
            qualification="native-cfd-full-service-v1",
        )

    def case(self, name, function):
        if self.monitor:
            self.monitor.phase = name
        return super().case(name, function)

    def preflight(self):
        verify_bundle(self.args.release_root)
        orchestration = read_json(self.args.release_root / "orchestration.json")
        verify_reference(orchestration["reference"])
        deployment = super().preflight()
        self.summary["environment"] = environment(
            self.release["target"], backend=self.release["backend"]
        )
        self.monitor = MemoryMonitor()
        self.monitor.start()
        return deployment

    def ready(self, http, redis):
        registry = super().ready(http, redis)
        proof = (
            next(iter(registry.values()))
            .get("model_cache", {})
            .get("warmup", {})
            .get("self_test", {})
        )
        verify_cfd_startup(proof, backend=self.release["backend"])
        return registry

    def request(self, http, label, capture):
        run_id, result = super().request(http, label, capture)
        scientific.assert_result(
            result,
            run_id=run_id,
            source_sha=self.args.source_sha,
            image=self.args.image,
            artifact_root=Path(
                read_json(self.args.release_root / "deployment.json")["artifact_root"]
            ),
            capture_inputs=capture,
            backend=self.release["backend"],
            compilation_source_sha=self.summary["compilation_source_sha"],
            model_build_sources=self.summary.get("model_build_sources"),
        )
        artifacts = {item["name"]: item for item in result["execution"]["outputs"]}
        values = {
            name: {key: artifacts[name][key] for key in ("sha256", "size_bytes")}
            for name in ("physical_fields", "standardized_fields")
        }
        if self.field_hashes is None:
            self.field_hashes = values
        require(
            values == self.field_hashes,
            "reused/recovered CFD request field bytes differ from the scientifically qualified first request",
        )
        self.runs.append(run_id)
        self.results[run_id] = (label, result)
        return run_id, result

    completion = ServiceJob.completion


def service_gates(job, http, redis, deployment, registry):
    def malformed():
        try:
            http.json(
                f"/v1/infer/{job.workflow}/run", body={"parameters": []}, expected=202
            )
        except urllib.error.HTTPError as error:
            require(error.code in {400, 422}, "malformed request failed unexpectedly")
            return
        raise ValueError("malformed CFD request was accepted")

    job.case("malformed_request", malformed)

    def overlap():
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(job.request, http, "overlap-" + str(index), False)
                for index in range(2)
            ]
            values = [future.result() for future in futures]
        require(
            len({value[1]["payload"]["native"]["attempt"] for value in values}) == 2,
            "overlapping CFD requests shared an artifact attempt",
        )

    job.case("overlap", overlap)
    worker = next(iter(redis.registry().values()))
    replacement = job.case(
        "worker_restart", lambda: run_idle_crash(job, http, redis, False, worker)
    )
    job.case(
        "active_recovery",
        lambda: run_active_crash(job, http, redis, False, replacement),
    )

    # ResultsRole and scheduler release processing are asynchronous; one GET/POST
    # is never replayed while waiting for the existing commits to settle.
    def settled_completion():
        deadline = time.monotonic() + 30
        while True:
            try:
                return job.completion(redis)
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)

    job.case("final_completion", settled_completion)
    job.case(
        "admission_rejection",
        lambda: reject_startup(job, redis, deployment, kind="gpu"),
    )
    job.case(
        "tampered_model", lambda: reject_startup(job, redis, deployment, kind="model")
    )


def run(args):
    release_root = Path(os.environ["NATIVE_DEPLOYMENT_CONFIG"]).parent
    release = verify_bundle(release_root)
    orchestration = read_json(release_root / "orchestration.json")
    source = Path("/opt/native-release/source")
    cfg = scientific.parser().parse_args(
        [
            "--output",
            str(args.output),
            "--run-id",
            args.run_id,
            "--image",
            args.image,
            "--source-sha",
            release["source_sha"],
            "--release-root",
            str(release_root),
            "--profile",
            str(source / "qa/inference/cfd_parity_surface_run1_transolver.json"),
            "--request",
            str(
                source
                / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
            ),
            "--reference",
            str(verify_reference(orchestration["reference"])),
            "--input-root",
            orchestration["input_root"],
            "--startup-timeout",
            "300",
            "--request-timeout",
            "1200",
            "--shutdown-timeout",
            "300",
        ]
    )
    require(
        release["workflow"] == scientific.WORKFLOW,
        "CFD preset workflow ID differs from the pinned scientific contract",
    )
    holder = []

    def factory(settings):
        job = CFDJob(settings, release)
        holder.append(job)
        return job

    try:
        scientific.run(cfg, job_factory=factory, qualify_service=service_gates)
    finally:
        if holder:
            job = holder[0]
            try:
                if job.monitor:
                    measurements = job.monitor.stop()
                    write_json(job.root / "memory.json", measurements)
                    job.summary["memory_evidence"] = file_identity(
                        job.root / "memory.json"
                    )
                    job.summary["memory"] = {
                        key: value
                        for key, value in measurements.items()
                        if key != "samples"
                    }
                job.save()
                seal_evidence(job.root, job.summary)
            except BaseException as error:
                job.summary.update(status="failed", evidence_error=str(error))
                job.save()
                raise
            finally:
                print(BEGIN, flush=True)
                print(json.dumps(job.summary, sort_keys=True), flush=True)
                print(END, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
