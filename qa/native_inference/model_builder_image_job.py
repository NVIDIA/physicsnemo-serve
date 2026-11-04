#!/usr/bin/env python3
"""Run the shipped Model Builder image payload directly in a Lepton GPU job.

This standalone driver uses only installed Model Builder code. The sibling
contract.py and mounted source snapshot supply independent evidence checks and
fixtures; no Docker daemon or repository Python import is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

if __package__:
    from . import contract
else:
    import contract

sha256 = contract.sha256
write_json = contract.write_json
SUMMARY_BEGIN = "MODEL_BUILDER_IMAGE_QA_SUMMARY_BEGIN"
SUMMARY_END = "MODEL_BUILDER_IMAGE_QA_SUMMARY_END"
BACKENDS = ("aoti", "tensorrt")
HELD_OUT = (
    (10.0, -3.0, 0.5, 7.25),
    (-8.0, 32.0, -0.125, 0.0),
    (1.5, -16.0, 128.0, 0.25),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_cases(profile="builder-image"):
    require(profile == "builder-image", "Unknown Model Builder image QA profile")
    return (
        ["environment", "sdk.consumer.build"]
        + [
            f"affine.{backend}.{operation}"
            for backend in BACKENDS
            for operation in ("check", "build", "cli", "sdk", "repeat", "relocation")
        ]
        + ["handoff"]
    )


def validate_summary(summary, *, run_id, source_sha, image_digest, profile):
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "profile": profile,
        "status": "passed",
        "qa_scope": "model-builder-image-payload",
        "docker_launcher_qa": "component-only",
    }
    require(
        isinstance(summary, dict)
        and type(summary.get("schema_version")) is int
        and all(summary.get(key) == value for key, value in identity.items()),
        "Builder image QA identity/status/scope mismatch",
    )
    names = expected_cases(profile)
    cases = summary.get("cases")
    require(
        summary.get("expected_cases") == names
        and isinstance(cases, list)
        and all(isinstance(case, dict) for case in cases)
        and sorted(case.get("name", "") for case in cases) == sorted(names)
        and all(case.get("status") == "passed" for case in cases),
        "Every required builder image case must execute and pass exactly once",
    )
    for case in cases:
        details = case.get("details")
        require(isinstance(details, dict), "Missing builder image case evidence")
        name = case["name"]
        if name == "environment":
            fields = (
                "runner_sha256",
                "contract_sha256",
                "source_manifest_sha256",
                "installed_builder_sha256",
                "entrypoint_sha256",
                "runtime_sha256",
            )
            require(
                all(
                    re.fullmatch(r"[0-9a-f]{64}", str(details.get(field, "")))
                    for field in fields
                )
                and details.get("installed_builder_matches_source") is True
                and str(details.get("gpu_uuid", "")).startswith("GPU-")
                and details.get("device") == {"type": "cuda", "index": 0},
                "Installed builder/source/GPU provenance is incomplete",
            )
        elif name == "sdk.consumer.build":
            require(
                re.fullmatch(r"[0-9a-f]{64}", str(details.get("consumer_sha256", ""))),
                "SDK consumer identity is missing",
            )
        elif name == "handoff":
            require(
                re.fullmatch(r"[0-9a-f]{64}", str(details.get("inventory_sha256", "")))
                and type(details.get("file_count")) is int
                and details["file_count"] > 0
                and details.get("verified") is True,
                "Persisted artifact inventory is incomplete",
            )
        elif name.endswith((".build", ".repeat")):
            require(
                all(
                    re.fullmatch(r"[0-9a-f]{64}", str(details.get(field, "")))
                    for field in ("build_sha256", "release_sha256")
                )
                and details.get("case_count") == 3
                and details.get("verified") is True,
                "Serialized build receipt or native verification is incomplete",
            )
        elif name.endswith(".check"):
            require(
                details.get("status") == "configuration-ok"
                and details.get("configuration_only") is True,
                "Model Builder recipe configuration check did not pass",
            )
        else:
            require(
                details.get("case_count") == len(HELD_OUT)
                and details.get("parity") == "passed"
                and details.get("backend") == name.split(".")[1]
                and details.get("device") == {"type": "cuda", "index": 0}
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(details.get("package_sha256", ""))
                ),
                "Independent CUDA SDK/CLI parity or package identity is missing",
            )


def extract_summary(output):
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    require(
        begin >= 0 and end >= 0, "Job logs contain no complete builder image QA summary"
    )
    report = json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())
    require(isinstance(report, dict), "Builder image summary must be an object")
    return report


def result_document(output):
    """Keep NVIDIA entrypoint banners while accepting exactly one CLI result."""
    decoder, cursor, values = json.JSONDecoder(), 0, []
    while match := re.search(r"(?m)^\s*\{", output[cursor:]):
        start = cursor + match.start()
        start += len(output[start:]) - len(output[start:].lstrip())
        try:
            value, length = decoder.raw_decode(output[start:])
        except ValueError:
            cursor = start + 1
            continue
        values.append(value)
        cursor = start + length
    require(
        len(values) == 1 and isinstance(values[0], dict),
        "Expected exactly one structured builder result",
    )
    return values[0]


def inventory(root):
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        require(
            not path.is_symlink(), "Source/artifact inventory cannot contain symlinks"
        )
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    require(records, "Source/artifact inventory cannot be empty")
    return records


def verify_source(root, manifest, digest, revision):
    require(
        sha256(manifest) == digest,
        "Source manifest digest differs from submitted manifest",
    )
    document = json.loads(Path(manifest).read_text())
    require(
        type(document.get("schema_version")) is int
        and document["schema_version"] == 1
        and re.fullmatch(
            r"[0-9a-f]{64}", str(document.get("source_archive_sha256", ""))
        )
        and document.get("source_sha") == revision,
        "Source manifest revision/schema differs",
    )
    actual = {
        record["path"]: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "mode": stat.S_IMODE((Path(root) / record["path"]).stat().st_mode),
            "symlink": None,
        }
        for record in inventory(root)
    }
    require(
        document.get("files") == actual,
        "Source snapshot inventory differs from pinned manifest",
    )
    return document


def compare_builder_sources(installed, source):
    def python_files(root):
        return [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            for path in sorted(root.rglob("*.py"))
            if path.is_file() and not path.is_symlink()
        ]

    actual, expected = python_files(Path(installed)), python_files(Path(source))
    require(
        actual and actual == expected,
        "The installed Model Builder Python bytes differ from staged source",
    )
    return hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()


def builder_command(operation, recipe, output, backend, runtime):
    command = [
        "/opt/nvidia/nvidia_entrypoint.sh",
        "pnms-model-builder",
        operation,
        "--recipe",
        str(recipe),
        "--executor",
        "local",
        "--device",
        "cuda:0",
        "--runtime",
        str(runtime),
        "--backend",
        backend,
        "--output",
        str(output),
        "--json",
    ]
    if operation == "check":
        command.append("--config-only")
    return command


class Context:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.source = args.source_root.resolve()
        self.runtime = args.sdk_prefix / "bin/physicsnemo-infer"
        self.recipe = (
            self.source / "native-inference/model-builder/models/affine/recipe.json"
        )
        self.consumer = self.root / "sdk-consumer/physicsnemo-qa-consumer"
        self.report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image_digest,
            "profile": "builder-image",
            "status": "running",
            "qa_scope": "model-builder-image-payload",
            "docker_launcher_qa": "component-only",
            "expected_cases": expected_cases(),
            "cases": [],
        }
        self.environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME"):
            self.environment.pop(key, None)
        self.environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def save(self):
        write_json(self.root / "summary.json", self.report)

    def case(self, name, function):
        require(
            name in expected_cases()
            and not any(case["name"] == name for case in self.report["cases"]),
            "Unknown or duplicate builder image case",
        )
        record = {"name": name, "status": "running"}
        self.report["cases"].append(record)
        self.save()
        started = time.monotonic()
        try:
            value = function()
            record.update(status="passed", details=value)
            return value
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            self.save()

    def run(self, name, arguments):
        destination = self.root / "logs" / name
        destination.mkdir(parents=True)
        command = [str(value) for value in arguments]
        write_json(
            destination / "command.json", {"argv": command, "cwd": str(self.root)}
        )
        with (
            (destination / "stdout.log").open("w") as stdout,
            (destination / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.root,
                env=self.environment,
                stdout=stdout,
                stderr=stderr,
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
        write_json(destination / "exit.json", {"exit_code": code})
        if code:
            tails = [
                (destination / f"{stream}.log").read_text(errors="replace")[-4096:]
                for stream in ("stdout", "stderr")
            ]
            raise RuntimeError(
                f"{name} failed with exit {code}; evidence: {destination}\n"
                + "\n".join(tails)
            )
        return (destination / "stdout.log").read_text(errors="replace")


def preflight(ctx):
    args = ctx.args
    require(
        sha256(__file__) == args.expected_runner_sha256,
        "Mounted QA runner bytes differ",
    )
    require(
        sha256(contract.__file__) == args.expected_contract_sha256,
        "Mounted QA contract bytes differ",
    )
    verify_source(
        ctx.source,
        args.source_manifest,
        args.source_manifest_sha256,
        args.expected_source_sha,
    )
    import model_builder
    import tensorrt
    import torch

    installed = Path(model_builder.__file__).resolve().parent
    require(
        not installed.is_relative_to(ctx.source),
        "QA must use the installed Model Builder, not the mounted checkout",
    )
    fingerprint = compare_builder_sources(
        installed, ctx.source / "native-inference/model-builder/src/model_builder"
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "Builder image QA requires exactly one allocated CUDA GPU",
    )
    for program in ("pnms-model-builder", str(ctx.runtime), "cmake", "c++"):
        require(
            shutil.which(program),
            f"Required builder image executable unavailable: {program}",
        )
    listing = result_document(
        ctx.run(
            "entrypoint-list",
            [
                "/opt/nvidia/nvidia_entrypoint.sh",
                "pnms-model-builder",
                "list",
                "--json",
            ],
        )
    )
    require(
        any(item.get("name") == "affine" for item in listing.get("models", [])),
        "Installed entrypoint does not list the affine recipe",
    )
    info = torch.cuda.get_device_properties(0)
    uuid = str(info.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    details = {
        "runner_sha256": sha256(__file__),
        "contract_sha256": sha256(contract.__file__),
        "source_manifest_sha256": args.source_manifest_sha256,
        "installed_builder_sha256": fingerprint,
        "installed_builder_matches_source": True,
        "entrypoint_sha256": sha256("/opt/nvidia/nvidia_entrypoint.sh"),
        "runtime_sha256": sha256(ctx.runtime),
        "installed_builder_path": str(installed),
        "gpu_uuid": uuid,
        "gpu_name": info.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "device": {"type": "cuda", "index": 0},
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
    }
    write_json(ctx.root / "environment.json", details)
    return details


def build_consumer(ctx):
    import torch

    command = [
        "cmake",
        "-S",
        ctx.source / "qa/native_inference/cpp",
        "-B",
        ctx.consumer.parent,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_PREFIX_PATH={ctx.args.sdk_prefix};{Path(torch.__file__).parent}",
    ]
    if Path("/opt/hpcx/ucx/lib").is_dir():
        command.append("-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/hpcx/ucx/lib")
    ctx.run("consumer.configure", command)
    ctx.run(
        "consumer.compile", ["cmake", "--build", ctx.consumer.parent, "--parallel", "4"]
    )
    return {"consumer_sha256": sha256(ctx.consumer)}


def checked_build(ctx, backend, operation, output):
    response = result_document(
        ctx.run(
            f"affine.{backend}.{operation}",
            builder_command("build", ctx.recipe, output, backend, ctx.runtime),
        )
    )
    require(
        response.get("status") == "complete",
        "Installed Model Builder did not complete the build",
    )
    contract.verify_build(output, [backend])
    return {
        "verified": True,
        "case_count": 3,
        "build_sha256": sha256(output / "build.json"),
        "release_sha256": sha256(output / "model/model-release.json"),
    }


def consume(ctx, backend, model, label, *, sdk):
    package = Path(contract.verify_model(model, [backend])[backend])
    destination = ctx.root / "consumers" / backend / label
    destination.mkdir(parents=True)
    inputs, comparisons = [], []
    for index, values in enumerate(HELD_OUT):
        path = destination / f"input-{index}.f32"
        path.write_bytes(struct.pack("<4f", *values))
        inputs.append(path)
    if sdk:
        ctx.run(
            f"{backend}.{label}", [ctx.consumer, package, backend, destination, *inputs]
        )
        metadata = json.loads((destination / "metadata.json").read_text())
        require(
            metadata.get("completed") is True
            and metadata.get("backend") == backend
            and metadata.get("execution_device") == {"type": "cuda", "index": 0}
            and metadata.get("executor_count") == 1
            and metadata.get("requests") == len(HELD_OUT),
            "Direct C++ SDK did not execute the requested CUDA backend three times",
        )
    else:
        for index, path in enumerate(inputs):
            metadata_path = destination / f"metadata-{index}.json"
            ctx.run(
                f"{backend}.{label}-{index}",
                [
                    ctx.runtime,
                    "run",
                    package,
                    "--backend",
                    backend,
                    "--device",
                    "cuda:0",
                    "--input-file",
                    f"input={path}",
                    "--output-file",
                    f"output={destination / f'output-{index}.f32'}",
                    "--output-metadata",
                    metadata_path,
                ],
            )
            metadata = json.loads(metadata_path.read_text())
            require(
                metadata.get("completed") is True
                and metadata.get("backend") == backend
                and metadata.get("execution_device") == {"type": "cuda", "index": 0},
                "C++ CLI did not execute the requested CUDA backend",
            )
    for index, values in enumerate(HELD_OUT):
        comparisons.append(
            contract.compare_f32(
                (destination / f"output-{index}.f32").read_bytes(),
                [2.0 * value + 1.0 for value in values],
            )
        )
    write_json(destination / "comparisons.json", comparisons)
    return {
        "case_count": len(HELD_OUT),
        "parity": "passed",
        "backend": backend,
        "device": {"type": "cuda", "index": 0},
        "package_sha256": sha256(package / "model.json"),
    }


def relocated_consume(ctx, backend, *, producer_root=None):
    """Use ctx.root for new evidence while isolating the original producer root."""
    producer_root = Path(producer_root or ctx.root).resolve()
    original = producer_root / "build" / backend / "model"
    destination = ctx.root / "relocated" / backend / "model"
    shutil.copytree(original, destination)
    require(
        inventory(original) == inventory(destination),
        "Compiled package changed during relocation",
    )
    producers = [producer_root / name for name in ("build", "repeat")]
    hidden = Path(tempfile.mkdtemp(prefix=".relocation-producers-", dir=producer_root))
    moved = []
    try:
        for path in producers:
            backup = hidden / path.name
            path.rename(backup)
            moved.append((path, backup))
        require(
            all(not path.exists() for path in producers),
            "Original producer paths remain available during relocation",
        )
        return {
            **consume(ctx, backend, destination, "relocated-sdk", sdk=True),
            "producer_paths": [str(path) for path in producers],
            "producer_paths_unavailable": True,
        }
    finally:
        # Rename preserves immutable bytes/modes; retain backups if restore fails.
        for path, backup in reversed(moved):
            backup.rename(path)
        hidden.rmdir()


def backend_cases(ctx, backend):
    prefix = f"affine.{backend}"
    output = ctx.root / "build" / backend

    def check():
        response = result_document(
            ctx.run(
                prefix + ".check",
                builder_command(
                    "check",
                    ctx.recipe,
                    ctx.root / "check" / backend,
                    backend,
                    ctx.runtime,
                ),
            )
        )
        require(
            response.get("status") == "configuration-ok",
            "Installed Model Builder recipe configuration check did not pass",
        )
        return {"status": "configuration-ok", "configuration_only": True}

    ctx.case(prefix + ".check", check)
    ctx.case(prefix + ".build", lambda: checked_build(ctx, backend, "build", output))
    ctx.case(
        prefix + ".cli",
        lambda: consume(ctx, backend, output / "model", "cli", sdk=False),
    )
    ctx.case(
        prefix + ".sdk",
        lambda: consume(ctx, backend, output / "model", "sdk", sdk=True),
    )
    ctx.case(
        prefix + ".repeat",
        lambda: checked_build(ctx, backend, "repeat", ctx.root / "repeat" / backend),
    )

    ctx.case(prefix + ".relocation", lambda: relocated_consume(ctx, backend))


def handoff(ctx):
    verify_source(
        ctx.source,
        ctx.args.source_manifest,
        ctx.args.source_manifest_sha256,
        ctx.args.expected_source_sha,
    )
    for backend in BACKENDS:
        contract.verify_build(ctx.root / "build" / backend, [backend])
        contract.verify_build(ctx.root / "repeat" / backend, [backend])
        contract.verify_model(ctx.root / "relocated" / backend / "model", [backend])
    records = [
        record
        for record in inventory(ctx.root)
        if record["path"] not in ("summary.json", "handoff.json")
    ]
    write_json(ctx.root / "handoff.json", {"format_version": 1, "files": records})
    require(
        records
        == [
            record
            for record in inventory(ctx.root)
            if record["path"] not in ("summary.json", "handoff.json")
        ],
        "Persisted artifact inventory changed",
    )
    return {
        "inventory_sha256": sha256(ctx.root / "handoff.json"),
        "file_count": len(records),
        "verified": True,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--expected-runner-sha256", required=True)
    result.add_argument("--expected-contract-sha256", required=True)
    result.add_argument(
        "--sdk-prefix", type=Path, default=Path("/opt/physicsnemo-inference")
    )
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-source-sha", required=True)
    result.add_argument("--image-digest", required=True)
    result.add_argument(
        "--profile", choices=("builder-image",), default="builder-image"
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--command-timeout", type=float, default=1200)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    ctx = Context(args)
    code = 1
    try:
        ctx.case("environment", lambda: preflight(ctx))
        ctx.case("sdk.consumer.build", lambda: build_consumer(ctx))
        for backend in BACKENDS:
            backend_cases(ctx, backend)
        ctx.case("handoff", lambda: handoff(ctx))
        ctx.report["status"] = "passed"
        validate_summary(
            ctx.report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            profile=args.profile,
        )
        code = 0
    except BaseException as error:
        ctx.report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        ctx.save()
        print(SUMMARY_BEGIN, flush=True)
        print(json.dumps(ctx.report, sort_keys=True), flush=True)
        print(SUMMARY_END, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
