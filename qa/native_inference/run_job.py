#!/usr/bin/env python3
"""Build and consume native model packages in one Lepton GPU batch job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

if __package__:
    from .contract import (
        SUMMARY_BEGIN,
        SUMMARY_END,
        compare_f32,
        expected_cases,
        sha256,
        validate_summary,
        verify_build,
        verify_model,
        write_json,
    )
else:
    from contract import (
        SUMMARY_BEGIN,
        SUMMARY_END,
        compare_f32,
        expected_cases,
        sha256,
        validate_summary,
        verify_build,
        verify_model,
        write_json,
    )


HELD_OUT = (
    (10.0, -3.0, 0.5, 7.25),
    (-8.0, 32.0, -0.125, 0.0),
    (1.5, -16.0, 128.0, 0.25),
)
BACKENDS = ("aoti", "tensorrt")


class Context:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.native_root = args.native_root.resolve()
        self.builder = args.builder
        self.runtime = args.runtime
        self.consumer = args.consumer
        self.workflow = args.workflow
        self.device = "cuda:0"
        self.report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image_digest,
            "profile": args.profile,
            "status": "running",
            "expected_cases": expected_cases(args.profile),
            "cases": [],
            "stages": {
                "build": {"status": "running"},
                "consumer": {"status": "blocked"},
            },
        }

    def save(self):
        write_json(self.root / "summary.json", self.report)

    def case(self, name, function):
        if name not in self.report["expected_cases"] or any(
            c["name"] == name for c in self.report["cases"]
        ):
            raise ValueError(f"unknown or duplicate QA case: {name}")
        record = {"name": name, "status": "running"}
        self.report["cases"].append(record)
        self.save()
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

    def run(self, name, args, *, cwd=None, check=True):
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        command = [str(a) for a in args]
        write_json(
            logs / f"{name}.command.json",
            {"argv": command, "cwd": str(cwd or self.root)},
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        stdout_path, stderr_path = (
            logs / f"{name}.stdout.log",
            logs / f"{name}.stderr.log",
        )
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd or self.root,
                env=env,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=self.args.command_timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        if check and code:
            detail = (
                f"{name} failed with exit {code}; see {stdout_path} and {stderr_path}"
            )
            for stream, path in (("stdout", stdout_path), ("stderr", stderr_path)):
                with path.open("rb") as log:
                    log.seek(max(0, path.stat().st_size - 4096))
                    tail = log.read(4096).decode("utf-8", errors="replace")
                detail += f"\n{stream} tail (last 4096 bytes):\n{tail}"
            raise RuntimeError(detail)
        return subprocess.CompletedProcess(
            command,
            code,
            stdout_path.read_text(errors="replace"),
            stderr_path.read_text(errors="replace"),
        )

    def builder_args(self, operation, project, backends, output=None):
        command = [
            self.builder,
            operation,
            str(project),
            "--executor",
            "local",
            "--device",
            self.device,
            "--runtime",
            self.runtime,
            "--json",
        ]
        for backend in backends:
            command += ["--backend", backend]
        if output is not None:
            command += ["--output", str(output)]
        return command

    def build_project(self, name, project, backends, output, *, update_lock=False):
        command = self.builder_args("build", project, backends, output)
        if update_lock:
            command.append("--update-lock")
        result = self.run(name, command)
        response = json.loads(result.stdout)
        if response.get("status") != "complete":
            raise ValueError("builder did not report a complete candidate")
        packages = verify_build(output, backends)
        return {
            "name": name,
            "model_dir": str(Path(output) / "model"),
            "packages": packages,
        }

    def relocate(self, record):
        target = self.root / "consumer" / record["name"] / "model"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = Path(record["model_dir"])
        verify_model(original, list(record["packages"]))
        shutil.copytree(original, target)
        packages = verify_model(target, list(record["packages"]))
        if sha256(target / "model-release.json") != sha256(
            original / "model-release.json"
        ):
            raise ValueError("release inventory changed during handoff")
        return {
            **record,
            "model_dir": str(target),
            "packages": packages,
            "release_sha256": sha256(target / "model-release.json"),
        }


def preflight(ctx):
    identity = json.loads(ctx.args.source_metadata.read_text())
    if identity.get("source_sha") != ctx.args.expected_source_sha:
        raise ValueError("QA image source SHA differs from submitted source SHA")
    for program in (
        ctx.builder,
        ctx.runtime,
        ctx.consumer,
        *([ctx.workflow] if ctx.args.profile == "full" else []),
    ):
        if not shutil.which(program):
            raise ValueError(f"required installed executable is unavailable: {program}")
    import torch
    import tensorrt

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ValueError("native QA requires an available CUDA GPU")
    info = ctx.run(
        "gpu",
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
    )
    metadata = {
        "source": identity,
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "nvidia_smi": info.stdout.strip(),
        "runtime_sha256": sha256(shutil.which(ctx.runtime)),
        "consumer_sha256": sha256(shutil.which(ctx.consumer)),
    }
    write_json(ctx.root / "environment.json", metadata)


def prepare_affine(ctx):
    project = ctx.root / "projects" / "affine"
    ctx.run(
        "affine.prepare",
        [
            sys.executable,
            ctx.native_root / "tools/examples/prepare_configured_affine.py",
            "--output",
            project,
        ],
    )
    return project


def build_affine(ctx):
    project = prepare_affine(ctx)

    def config_check():
        result = ctx.run(
            "affine.config",
            ctx.builder_args("check", project, BACKENDS) + ["--config-only"],
        )
        if json.loads(result.stdout).get("status") != "configuration-ok":
            raise ValueError("configuration check did not pass")
        if (project / "model-build.lock.json").exists():
            raise ValueError("configuration check unexpectedly published a lock")

    def eager_check():
        result = ctx.run(
            "affine.check",
            ctx.builder_args("check", project, BACKENDS, ctx.root / "check-affine"),
        )
        if json.loads(result.stdout).get("status") != "checked":
            raise ValueError("eager check did not pass")

    ctx.case("affine.config", config_check)
    ctx.case("affine.check", eager_check)
    first = ctx.case(
        "affine.build_a",
        lambda: ctx.build_project(
            "affine-a", project, BACKENDS, ctx.root / "build/affine-a"
        ),
    )
    # Modify only the task-owned project; the completed A package stays intact.
    shutil.copyfile(project / "checkpoint-b.pt", project / "checkpoint-a.pt")

    def reject_changed_lock():
        output = ctx.root / "build/rejected-checkpoint"
        lock = project / "model-build.lock.json"
        previous = lock.read_bytes()
        result = ctx.run(
            "affine.lock_rejection",
            ctx.builder_args("build", project, BACKENDS, output),
            check=False,
        )
        diagnostics = json.loads(result.stdout).get("diagnostics", [])
        if result.returncode == 0 or not any(
            "lock" in json.dumps(d).lower() for d in diagnostics
        ):
            raise ValueError("changed checkpoint was not rejected by the project lock")
        if (
            lock.read_bytes() != previous
            or (output / "model/model-release.json").exists()
        ):
            raise ValueError("rejected build changed its lock or published a release")

    ctx.case("affine.lock_rejection", reject_changed_lock)
    second = ctx.case(
        "affine.build_b",
        lambda: ctx.build_project(
            "affine-b", project, BACKENDS, ctx.root / "build/affine-b", update_lock=True
        ),
    )
    return [{**first, "checkpoint": "a"}, {**second, "checkpoint": "b"}]


def expected_affine(checkpoint, values):
    return [x + 3.0 if checkpoint == "a" else 4.0 - 0.5 * x for x in values]


def native_args(ctx, package, backend, input_path, output_path, metadata_path):
    return [
        ctx.runtime,
        "run",
        str(package),
        "--backend",
        backend,
        "--device",
        ctx.device,
        "--input-file",
        f"input={input_path}",
        "--output-file",
        f"output={output_path}",
        "--output-metadata",
        str(metadata_path),
    ]


def consume_affine_cli(ctx, record, backend):
    package = Path(record["packages"][backend])
    destination = Path(record["model_dir"]).parent / backend / "cli"
    destination.mkdir(parents=True)
    comparisons = []
    for index, values in enumerate(HELD_OUT):
        input_path = destination / f"input-{index}.f32"
        output_path = destination / f"output-{index}.f32"
        metadata_path = destination / f"metadata-{index}.json"
        input_path.write_bytes(struct.pack("<4f", *values))
        ctx.run(
            f"{record['name']}.{backend}.cli{index}",
            native_args(ctx, package, backend, input_path, output_path, metadata_path),
            cwd=destination,
        )
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema_version") != 1
            or metadata.get("completed") is not True
            or metadata.get("backend") != backend
            or metadata.get("execution_device") != {"type": "cuda", "index": 0}
            or metadata.get("outputs")
            != [
                {
                    "name": "output",
                    "dtype": "float32",
                    "shape": [4],
                    "byte_size": 16,
                    "device": {"type": "cpu", "index": 0},
                }
            ]
        ):
            raise ValueError(
                "native CLI metadata does not match the requested affine contract"
            )
        comparisons.append(
            compare_f32(
                output_path.read_bytes(), expected_affine(record["checkpoint"], values)
            )
        )
    write_json(destination / "comparisons.json", comparisons)


def consume_affine_sdk(ctx, record, backend):
    destination = Path(record["model_dir"]).parent / backend / "sdk"
    destination.mkdir(parents=True)
    inputs = []
    for index, values in enumerate(HELD_OUT):
        path = destination / f"input-{index}.f32"
        path.write_bytes(struct.pack("<4f", *values))
        inputs.append(str(path))
    ctx.run(
        f"{record['name']}.{backend}.sdk",
        [ctx.consumer, record["packages"][backend], backend, str(destination), *inputs],
        cwd=destination,
    )
    metadata = json.loads((destination / "metadata.json").read_text())
    if (
        metadata.get("completed") is not True
        or metadata.get("backend") != backend
        or metadata.get("execution_device") != {"type": "cuda", "index": 0}
        or metadata.get("requests") != len(inputs)
        or metadata.get("executor_count") != 1
    ):
        raise ValueError("SDK consumer did not reuse the requested CUDA executor")
    comparisons = [
        compare_f32(
            (destination / f"output-{index}.f32").read_bytes(),
            expected_affine(record["checkpoint"], values),
        )
        for index, values in enumerate(HELD_OUT)
    ]
    write_json(destination / "comparisons.json", comparisons)


def reject_native_input(ctx, record, backend, *, missing_payload):
    kind = "missing_payload" if missing_payload else "truncated_input"
    destination = Path(record["model_dir"]).parent / backend / kind
    destination.mkdir(parents=True)
    package = Path(record["packages"][backend])
    if missing_payload:
        original = package
        package = destination / "package"
        shutil.copytree(original, package)
        manifest = json.loads((package / "model.json").read_text())
        (package / manifest["artifacts"][0]["path"]).unlink()
    input_path, output_path, metadata_path = (
        destination / "input.f32",
        destination / "output.f32",
        destination / "metadata.json",
    )
    input_path.write_bytes(
        struct.pack("<4f", *HELD_OUT[0]) if missing_payload else b"\0"
    )
    result = ctx.run(
        f"{record['name']}.{backend}.{kind}",
        native_args(ctx, package, backend, input_path, output_path, metadata_path),
        cwd=destination,
        check=False,
    )
    diagnostic = (result.stdout + result.stderr).lower()
    expected_error = (
        ("artifact", "file", "exist", "open")
        if missing_payload
        else ("input file size",)
    )
    if result.returncode == 0 or not any(
        token in diagnostic for token in expected_error
    ):
        raise ValueError(
            f"native CLI did not reject {kind} with an actionable diagnostic"
        )
    if output_path.exists() or metadata_path.exists():
        raise ValueError(
            "rejected native execution published output or completion metadata"
        )


def consume_affine(ctx, records):
    for record in records:
        for backend in BACKENDS:
            prefix = f"affine.{record['checkpoint']}.{backend}"
            ctx.case(prefix + ".cli", lambda: consume_affine_cli(ctx, record, backend))
            ctx.case(prefix + ".sdk", lambda: consume_affine_sdk(ctx, record, backend))
            ctx.case(
                prefix + ".missing_payload",
                lambda: reject_native_input(ctx, record, backend, missing_payload=True),
            )
            ctx.case(
                prefix + ".truncated_input",
                lambda: reject_native_input(
                    ctx, record, backend, missing_payload=False
                ),
            )


def write_junit(ctx):
    cases = ctx.report["cases"]
    suite = ET.Element(
        "testsuite",
        name="native-inference",
        tests=str(len(cases)),
        failures=str(sum(c["status"] == "failed" for c in cases)),
        skipped=str(sum(c["status"] == "blocked" for c in cases)),
    )
    for case in cases:
        node = ET.SubElement(
            suite, "testcase", name=case["name"], time=str(case.get("seconds", 0))
        )
        if case["status"] == "failed":
            ET.SubElement(node, "failure", message=case.get("error", "failed"))
        elif case["status"] == "blocked":
            ET.SubElement(
                node,
                "skipped",
                message="blocked by an earlier failure; overall QA failed",
            )
    ET.ElementTree(suite).write(
        ctx.root / "junit.xml", encoding="unicode", xml_declaration=True
    )


def handoff_models(ctx, records):
    relocated = [ctx.relocate(record) for record in records]
    # Preserve producer evidence under new paths. Accidental references to the
    # original build tree must fail during standalone native consumption.
    producer = ctx.root / "producer"
    producer.mkdir()
    for name in ("projects", "build", "builds", "check-affine"):
        original = ctx.root / name
        if original.exists():
            original.rename(producer / name)
    write_json(
        ctx.root / "handoff.json",
        {
            "schema_version": 1,
            "run_id": ctx.args.run_id,
            "source_sha": ctx.args.expected_source_sha,
            "models": relocated,
        },
    )
    return relocated


def execute(args):
    # Never overwrite a prior run, including a failed attempt.
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    ctx = Context(args)
    stage = "build"

    def terminate(signum, frame):
        raise SystemExit(f"terminated by signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        ctx.case("environment", lambda: preflight(ctx))
        records = build_affine(ctx)
        if args.profile == "full":
            if __package__:
                from . import transolver
            else:
                import transolver
            records += transolver.prepare_and_build(ctx, args.assets)

        deployed = ctx.case("handoff", lambda: handoff_models(ctx, records))
        ctx.report["stages"]["build"]["status"] = "passed"
        stage = "consumer"
        ctx.report["stages"][stage]["status"] = "running"
        ctx.save()
        consume_affine(ctx, [r for r in deployed if "checkpoint" in r])
        if args.profile == "full":
            transolver.run_consumers(ctx, [r for r in deployed if "point_count" in r])
        ctx.report["stages"][stage]["status"] = "passed"
        ctx.report["status"] = "passed"
        validate_summary(
            ctx.report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            profile=args.profile,
        )
    except BaseException as error:
        ctx.report.update(status="failed", error=f"{type(error).__name__}: {error}")
        ctx.report["stages"][stage]["status"] = "failed"
    finally:
        completed = {c["name"] for c in ctx.report["cases"]}
        ctx.report["cases"].extend(
            {"name": name, "status": "blocked"}
            for name in ctx.report["expected_cases"]
            if name not in completed
        )
        ctx.save()
        write_junit(ctx)
        print(SUMMARY_BEGIN, flush=True)
        print(
            json.dumps(ctx.report, separators=(",", ":"), allow_nan=False), flush=True
        )
        print(SUMMARY_END, flush=True)
        signal.signal(signal.SIGTERM, previous_handler)
    return 0 if ctx.report["status"] == "passed" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    result.add_argument("--run-id", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--expected-source-sha", required=True)
    result.add_argument("--image-digest", required=True)
    result.add_argument(
        "--assets",
        type=Path,
        help="Use this local manifest offline; full profile otherwise downloads pinned assets",
    )
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument(
        "--native-root", type=Path, default=Path("/opt/physicsnemo/native-inference")
    )
    result.add_argument("--builder", default="pnms-model-builder")
    result.add_argument(
        "--runtime", default="/opt/physicsnemo-inference/bin/physicsnemo-infer"
    )
    result.add_argument(
        "--consumer", default="/opt/physicsnemo-inference/bin/physicsnemo-qa-consumer"
    )
    result.add_argument(
        "--workflow", default="/opt/physicsnemo-inference/bin/physicsnemo-transolver"
    )
    result.add_argument("--command-timeout", type=int, default=1800)
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_id):
        cli.error("--run-id must be 1-64 lowercase letters, digits or hyphens")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_source_sha):
        cli.error("--expected-source-sha must be a full Git SHA")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.image_digest):
        cli.error("--image-digest must be an immutable image reference")
    if args.command_timeout <= 0:
        cli.error("--command-timeout must be positive")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
