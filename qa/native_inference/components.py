"""Bounded native CPU component QA inside an immutable Lepton job image."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = REPO_ROOT / "native-inference/deploy"
sys.path.insert(0, str(DEPLOY_ROOT))
from native_release import file_identity, require, write_json  # noqa: E402
from native_release_artifacts import seal  # noqa: E402


SUMMARY_BEGIN = "NATIVE_COMPONENT_QA_SUMMARY_BEGIN"
SUMMARY_END = "NATIVE_COMPONENT_QA_SUMMARY_END"
QUALIFICATION = "native-cpu-components-v1"
CASES = (
    "preflight",
    "sdk-core",
    "workflow-mock",
    "python-native",
    "rust-native",
    "lint",
)
TEST_CASES = frozenset({"sdk-core", "workflow-mock", "python-native", "rust-native"})

# Keep the sealed component inventory limited to the carried native release suites.
PYTHON_SUITES = (
    "qa/scripts/test_create_native_raw_variant.py",
    "tests/test_export_native_cfd_release.py",
    "tests/test_native_authoring.py",
    "tests/test_native_cfd_contract.py",
    "tests/test_native_cfd_http_job.py",
    "tests/test_native_cfd_job.py",
    "tests/test_native_cfd_local_assets.py",
    "tests/test_native_cfd_overlay_guard.py",
    "tests/test_native_cfd_startup.py",
    "tests/test_native_health_stub.py",
    "tests/test_native_http_artifact_goldens.py",
    "tests/test_native_http_completion.py",
    "tests/test_native_http_multipart.py",
    "tests/test_native_http_negative_transport.py",
    "tests/test_native_http_startup_acceptance.py",
    "tests/test_native_image_backends.py",
    "tests/test_native_image_build.py",
    "tests/test_native_inference_provider_qa.py",
    "tests/test_native_inference_qa_assets.py",
    "tests/test_native_inference_qa_consumer.py",
    "tests/test_native_inference_qa_contract.py",
    "tests/test_native_inference_qa_job.py",
    "tests/test_native_inference_transolver_qa.py",
    "tests/test_native_lepton_components.py",
    "tests/test_native_qa_image.py",
    "tests/test_native_qa_workflow.py",
    "tests/test_native_release.py",
    "tests/test_native_release_artifact_retry.py",
    "tests/test_native_release_artifacts.py",
    "tests/test_native_release_backends.py",
    "tests/test_native_release_cfd_orchestration.py",
    "tests/test_native_release_controllers.py",
    "tests/test_native_release_evidence.py",
    "tests/test_native_release_faults.py",
    "tests/test_native_release_orchestration.py",
    "tests/test_native_release_qa_imports.py",
    "tests/test_native_user_cli.py",
    "tests/test_native_user_release.py",
    "tests/test_native_workflow_release_ci.py",
    "tests/test_run_lepton_native_cfd.py",
    "tests/test_run_lepton_native_inference_qa.py",
)


def expected_cases(profile="full"):
    require(profile == "full", "unsupported native component profile")
    return list(CASES)


def validate_counts(counts):
    require(isinstance(counts, dict), "component test counts are missing")
    for name in ("tests", "failed", "skipped"):
        require(
            type(counts.get(name)) is int and counts[name] >= 0,
            "invalid component test counts",
        )
    require(
        counts["tests"] > 0 and counts["failed"] == counts["skipped"] == 0,
        "required component tests must execute without failures or skips",
    )


def validate_summary(report, *, run_id, source_sha, image_digest, profile):
    require(isinstance(report, dict), "component summary must be an object")
    execution = report.get("execution")
    require(
        isinstance(execution, dict) and execution.get("provider") == "lepton",
        "component execution provider must be Lepton",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", source_sha), "full source SHA required")
    require(
        re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image_digest),
        "immutable component image required",
    )
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "profile": profile,
        "status": "passed",
        "qualification": QUALIFICATION,
        "expected_cases": expected_cases(profile),
    }
    for key, value in identity.items():
        require(
            type(report.get(key)) is type(value) and report[key] == value,
            f"component summary {key} differs",
        )
    cases = report.get("cases")
    require(
        isinstance(cases, list) and all(isinstance(case, dict) for case in cases),
        "component case results are missing",
    )
    require(
        [case.get("name") for case in cases] == expected_cases(profile),
        "component executed case inventory differs",
    )
    for case in cases:
        seconds = case.get("seconds")
        require(
            case.get("status") == "passed"
            and type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds >= 0,
            "every component case must execute and pass",
        )
        if case["name"] in TEST_CASES:
            validate_counts(case.get("counts"))


def junit_counts(path, *, names=None):
    tree = ET.parse(path)
    cases = list(tree.getroot().iter("testcase"))
    if names is not None:
        require(
            {case.get("name") for case in cases} == set(names),
            "CTest case inventory differs",
        )
    counts = {"tests": len(cases), "failed": 0, "skipped": 0}
    for case in cases:
        counts["failed"] += int(
            case.find("failure") is not None or case.find("error") is not None
        )
        counts["skipped"] += int(
            case.find("skipped") is not None
            or case.get("status") in {"notrun", "disabled"}
        )
    validate_counts(counts)
    return counts


def rust_counts(path):
    rows = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;",
        Path(path).read_text(),
    )
    counts = {
        "tests": sum(int(row[0]) for row in rows),
        "failed": sum(int(row[1]) for row in rows),
        "skipped": sum(int(row[2]) for row in rows),
    }
    validate_counts(counts)
    return counts


class Context:
    def __init__(self, args, build_root):
        self.args = args
        self.build = Path(build_root)
        self.reports = args.output / "reports"
        self.reports.mkdir()
        self.logs = self.reports / "logs"
        self.logs.mkdir()
        self.deadline = time.monotonic() + args.timeout_seconds
        self.sdk = self.build / "sdk"
        self.consumer = self.build / "consumer/physicsnemo-qa-consumer"
        self.env = {
            **os.environ,
            "CARGO_NET_OFFLINE": "true",
            "CARGO_BUILD_JOBS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NATIVE_QA_CONSUMER": str(self.consumer),
            "PYTHONPATH": os.pathsep.join(
                [
                    str(REPO_ROOT),
                    str(REPO_ROOT / "scripts"),
                    str(REPO_ROOT / "python"),
                    str(REPO_ROOT / "python/e2s_tools"),
                    "/usr/lib/python3/dist-packages",
                ]
            ),
            "LD_LIBRARY_PATH": str(self.build / "workflow")
            + os.pathsep
            + str(self.sdk / "lib")
            + os.pathsep
            + os.environ.get("LD_LIBRARY_PATH", ""),
        }
        self.env.setdefault("CARGO_TARGET_DIR", str(self.build / "cargo"))
        self.summary = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image_digest,
            "profile": args.profile,
            "status": "running",
            "qualification": QUALIFICATION,
            "expected_cases": expected_cases(args.profile),
            "cases": [],
            "execution": {
                "provider": "lepton",
                "resource_shape": args.resource_shape,
                "cpu_only": True,
            },
        }

    def save(self):
        write_json(self.args.output / "summary.json", self.summary)
        write_json(self.reports / "summary.json", self.summary)

    def command(self, name, command):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "component job deadline exceeded")
        timeout = min(self.args.stage_timeout, remaining)
        command = [str(value) for value in command]
        write_json(
            self.logs / (name + ".command.json"),
            {"argv": command, "cwd": str(REPO_ROOT), "timeout_seconds": timeout},
        )
        output = self.logs / (name + ".log")
        with output.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                raise TimeoutError(
                    "component command deadline exceeded: " + name
                ) from None
        require(code == 0, "component command failed: " + name)
        return output

    def case(self, name, function):
        require(
            name == CASES[len(self.summary["cases"])], "component case order differs"
        )
        record = {"name": name, "status": "running"}
        self.summary["cases"].append(record)
        start = time.monotonic()
        try:
            counts = function()
            if name in TEST_CASES:
                validate_counts(counts)
                record["counts"] = counts
            record["status"] = "passed"
        except BaseException as error:
            record.update(
                status="failed", error=type(error).__name__ + ": " + str(error)
            )
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            self.save()


def preflight(ctx):
    require(
        os.environ.get("SOURCE_SHA") == ctx.args.expected_source_sha,
        "component image source differs",
    )
    source = ctx.command("git-source", ["git", "rev-parse", "HEAD"]).read_text().strip()
    require(source == ctx.args.expected_source_sha, "component checkout source differs")
    ctx.command("git-clean", ["git", "diff", "--exit-code", "HEAD", "--"])
    require(
        not list(Path("/dev").glob("nvidia*")), "component job must use a CPU resource"
    )
    require(
        not Path("/var/run/docker.sock").exists(),
        "component job must not use Docker-in-Docker",
    )
    ctx.command(
        "python-dependencies",
        [
            sys.executable,
            "-c",
            "import torch,vtk,onnx,redis,pytest,yaml,jsonschema,numpy,xarray,zarr,netCDF4; print('required CPU dependencies imported')",
        ],
    )
    for name, command in (
        ("rust-version", ["rustc", "--version"]),
        ("cmake-version", ["cmake", "--version"]),
        ("redis-version", ["redis-server", "--version"]),
        ("ruff-version", ["ruff", "--version"]),
    ):
        ctx.command(name, command)
    tracked = (
        ctx.command("git-files", ["git", "ls-files", "-z"]).read_bytes().split(b"\0")
    )
    files = []
    for value in tracked:
        if value:
            name = os.fsdecode(value)
            path = REPO_ROOT / name
            require(
                path.is_file() and not path.is_symlink(),
                "regular tracked component source required",
            )
            files.append({"path": name, **file_identity(path)})
    write_json(
        ctx.reports / "source.json",
        {
            "source_sha": source,
            "files": files,
            "python": sys.version,
            "execution": ctx.summary["execution"],
        },
    )


def sdk_core(ctx):
    build = ctx.build / "core"
    ctx.command(
        "sdk-configure",
        [
            "cmake",
            "-S",
            REPO_ROOT / "native-inference",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPNMIR_BUILD_TESTS=ON",
            "-DPNMIR_ENABLE_AOTI=OFF",
            "-DPNMIR_ENABLE_ONNXRUNTIME=OFF",
            "-DPNMIR_ENABLE_TENSORRT=OFF",
        ],
    )
    ctx.command("sdk-build", ["cmake", "--build", build, "--parallel", "2"])
    report = ctx.reports / "sdk-core.xml"
    ctx.command(
        "sdk-ctest",
        [
            "ctest",
            "--test-dir",
            build,
            "--output-on-failure",
            "--no-tests=error",
            "--output-junit",
            report,
        ],
    )
    ctx.command("sdk-install", ["cmake", "--install", build, "--prefix", ctx.sdk])
    return junit_counts(report)


def workflow_mock(ctx):
    build = ctx.build / "workflow"
    ctx.command(
        "workflow-configure",
        [
            "cmake",
            "-S",
            REPO_ROOT / "native-inference/workflows/runtime",
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON",
            "-DPNMS_ENABLE_MOCK_BACKEND=ON",
            "-DCMAKE_PREFIX_PATH=" + str(ctx.sdk),
            "-DPNMS_CUSTOM_WORKFLOW_SOURCE_DIR="
            + str(REPO_ROOT / "native-inference/workflows/runtime/tests/custom"),
            "-DPNMS_CUSTOM_WORKFLOW_TARGET=pnms_test_custom",
        ],
    )
    ctx.command("workflow-build", ["cmake", "--build", build, "--parallel", "2"])
    report = ctx.reports / "workflow-mock.xml"
    ctx.command(
        "workflow-ctest",
        [
            "ctest",
            "--test-dir",
            build,
            "--output-on-failure",
            "--no-tests=error",
            "--output-junit",
            report,
        ],
    )
    ctx.command(
        "consumer-configure",
        [
            "cmake",
            "-S",
            REPO_ROOT / "qa/native_inference/cpp",
            "-B",
            ctx.build / "consumer",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPNMIR_QA_WITH_CUDA=OFF",
            "-DCMAKE_PREFIX_PATH=" + str(ctx.sdk),
        ],
    )
    ctx.command(
        "consumer-build",
        ["cmake", "--build", ctx.build / "consumer", "--parallel", "2"],
    )
    require(ctx.consumer.is_file(), "installed SDK CPU consumer was not built")
    return junit_counts(report, names={"workflow_host_tests", "workflow_custom_tests"})


def python_native(ctx):
    files = [REPO_ROOT / name for name in PYTHON_SUITES]
    require(
        files and all(path.is_file() for path in files),
        "native Python test inventory is empty or incomplete",
    )
    write_json(
        ctx.reports / "python-inventory.json",
        [
            {"path": path.relative_to(REPO_ROOT).as_posix(), **file_identity(path)}
            for path in files
        ],
    )
    report = ctx.reports / "python-native.xml"
    ctx.command(
        "python-native",
        [
            sys.executable,
            "-m",
            "pytest",
            *files,
            "-q",
            "-o",
            "xfail_strict=true",
            "--junitxml=" + str(report),
        ],
    )
    return junit_counts(report)


def rust_native(ctx):
    log = ctx.command(
        "rust-native",
        [
            "cargo",
            "test",
            "--offline",
            "--locked",
            "-p",
            "inference_server",
            "-p",
            "native-worker",
            "-p",
            "worker-runtime",
            "--",
            "--include-ignored",
            "--test-threads=2",
        ],
    )
    return rust_counts(log)


def lint(ctx):
    ctx.command("ruff-check", ["ruff", "check", "."])
    ctx.command("ruff-format", ["ruff", "format", "--check", "."])
    ctx.command("rust-fmt", ["cargo", "fmt", "--all", "--", "--check"])
    ctx.command(
        "rust-clippy",
        [
            "cargo",
            "clippy",
            "--offline",
            "--locked",
            "-p",
            "inference_server",
            "-p",
            "native-worker",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )


def run(args):
    require(
        0 < args.stage_timeout <= args.timeout_seconds <= 7200,
        "finite component deadlines required",
    )
    expected_cases(args.profile)
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="native-component-build-") as build:
        ctx = Context(args, build)
        try:
            for name, function in zip(
                CASES,
                (preflight, sdk_core, workflow_mock, python_native, rust_native, lint),
                strict=True,
            ):
                ctx.case(name, lambda function=function: function(ctx))
            ctx.summary["status"] = "passed"
            validate_summary(
                ctx.summary,
                run_id=args.run_id,
                source_sha=args.expected_source_sha,
                image_digest=args.image_digest,
                profile=args.profile,
            )
            return ctx.summary
        except BaseException as error:
            ctx.summary.update(
                status="failed", error=type(error).__name__ + ": " + str(error)
            )
            raise
        finally:
            ctx.save()
            try:
                manifest = seal(
                    ctx.reports,
                    args.output / "bundle.tar",
                    run_id=args.run_id,
                    source_sha=args.expected_source_sha,
                    image_digest=args.image_digest,
                    qualification=QUALIFICATION,
                )
                write_json(args.output / "manifest.json", manifest)
            except BaseException as error:
                ctx.summary.update(
                    status="failed",
                    evidence_error=type(error).__name__ + ": " + str(error),
                )
                ctx.save()
                raise
            finally:
                print(SUMMARY_BEGIN, flush=True)
                print(json.dumps(ctx.summary, sort_keys=True), flush=True)
                print(SUMMARY_END, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--profile", choices=("full",), default="full")
    parser.add_argument("--resource-shape", default="cpu.medium")
    parser.add_argument("--timeout-seconds", type=float, default=7140)
    parser.add_argument("--stage-timeout", type=float, default=3600)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
