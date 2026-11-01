#!/usr/bin/env python3
"""Acceptance of the public local pnms workflow inside a finite Lepton job."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

if __package__:
    from .contract import sha256, verify_build, write_json
else:
    from contract import sha256, verify_build, write_json

SUMMARY_BEGIN = "PNMS_APPLICATION_QA_SUMMARY_BEGIN"
SUMMARY_END = "PNMS_APPLICATION_QA_SUMMARY_END"
NATIVE_TARGETS = tuple(
    (backend, template, "application/json")
    for backend in ("aoti", "tensorrt")
    for template in ("tensor", "custom")
) + (("aoti", "tensor", "multipart/form-data"),)


def candidate_key(item):
    return (
        item.get("backend"),
        item.get("template"),
        item.get("ingress", "application/json"),
    )


def candidate_name(item):
    backend, template, ingress = candidate_key(item)
    return f"native.{backend}." + (
        "files" if ingress == "multipart/form-data" else template
    )


def raw_parameters():
    return {
        "values": {
            "artifact": "features",
            "dtype": "float32",
            "shape": [4],
            "byte_order": "little",
            "layout": "contiguous",
            "size_bytes": 16,
        }
    }


def configure_file_project(app):
    import yaml

    descriptor = app / "pnms.yaml"
    project = yaml.safe_load(descriptor.read_text())
    project["workflow"]["tensor_adapter"] = {
        "model_alias": "primary",
        "inputs": {"input": {"request_field": "values", "codec": "raw-fp32-le"}},
        "outputs": {"output": {"response_field": "values", "codec": "json-array-fp32"}},
    }
    project["ingress"] = {
        "content_type": "multipart/form-data",
        "files": [
            {
                "name": "features",
                "required": True,
                "media_types": ["application/octet-stream"],
                "max_size_mb": 1,
            }
        ],
        "form_schema_inline": {
            "type": "object",
            "required": ["values"],
            "properties": {"values": {"const": raw_parameters()["values"]}},
            "additionalProperties": False,
        },
    }
    descriptor.write_text(yaml.safe_dump(project, sort_keys=False))
    cases_path = app / "fixtures/cases.json"
    cases = json.loads(cases_path.read_text())
    for index, case in enumerate(cases["cases"]):
        relative = f"fixtures/raw-{index}.f32"
        (app / relative).write_bytes(struct.pack("<4f", *case["request"]["values"]))
        case["request"] = raw_parameters()
        case["inputs"] = {"features": relative}
    write_json(cases_path, cases)


def expected_cases(profile="local"):
    if profile != "local":
        raise ValueError("pnms application QA supports the local profile")
    return (
        ["environment"]
        + [
            f"python.{kind}.{operation}"
            for kind in ("new", "legacy")
            for operation in ("init", "check", "build", "run", "dev_edit")
        ]
        + [
            candidate_name(
                {"backend": backend, "template": template, "ingress": ingress}
            )
            + "."
            + operation
            for backend, template, ingress in NATIVE_TARGETS
            for operation in (
                "init",
                "check",
                "build",
                "held_out",
                "cache_reuse",
                "relocation",
                "reject_invalid",
            )
        ]
        + ["environment_lock.refresh"]
    )


def validate_summary(summary, *, run_id, source_sha, image_digest, profile):
    identity = {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "profile": profile,
        "status": "passed",
        "qa_scope": "local-workflow",
        "http_qa": "not_run",
    }
    if not isinstance(summary, dict) or any(
        summary.get(key) != value for key, value in identity.items()
    ):
        raise ValueError("pnms QA summary identity/status/scope mismatch")
    names = expected_cases(profile)
    cases = summary.get("cases")
    if summary.get("expected_cases") != names or not isinstance(cases, list):
        raise ValueError("pnms QA case inventory differs")
    if sorted(case.get("name", "") for case in cases) != sorted(names) or any(
        case.get("status") != "passed" for case in cases
    ):
        raise ValueError(
            "Every required pnms QA case must execute and pass exactly once"
        )
    for case in cases:
        details = case.get("details", {})
        if case["name"].startswith("native.") and case["name"].endswith(".build"):
            for field, length in (
                ("build_id", 32),
                ("bundle_digest", 64),
                ("model_identity", 64),
            ):
                if not re.fullmatch(
                    f"[0-9a-f]{{{length}}}", str(details.get(field, ""))
                ):
                    raise ValueError("Native QA build identities are missing")
            native = details.get("native_checks", {})
            if native.get("verified") is not True or native.get("case_count") != 3:
                raise ValueError("Native build parity evidence is missing")
        if case["name"].endswith(".held_out") and (
            details.get("case_count") != 3 or details.get("parity") != "passed"
        ):
            raise ValueError("Independent held-out native parity is missing")
        if (
            case["name"] == "native.aoti.files.held_out"
            and details.get("input_kind") != "raw-fp32-le"
        ):
            raise ValueError("Raw-file held-out inference evidence is missing")
        if case["name"].startswith("native.") and case["name"].endswith(".cache_reuse"):
            if (
                details.get("model_reused") is not True
                or details.get("workflow_reused") is not True
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(details.get("environment_lock_digest", ""))
                )
                or not re.fullmatch(r"[0-9a-f]{32}", str(details.get("build_id", "")))
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(details.get("bundle_digest", ""))
                )
            ):
                raise ValueError("Latest reused native build identity is missing")
    candidates = summary.get("release_candidates", [])
    required = set(NATIVE_TARGETS)
    if (
        len(candidates) != len(required)
        or {candidate_key(item) for item in candidates} != required
    ):
        raise ValueError(
            "Release candidate handoff must contain all five tested bundles, including raw-file ingress"
        )
    for item in candidates:
        operation = (
            "environment_lock.refresh"
            if candidate_key(item) == ("aoti", "tensor", "application/json")
            else candidate_name(item) + ".cache_reuse"
        )
        build = next(case["details"] for case in cases if case["name"] == operation)
        if any(item.get(key) != build[key] for key in ("build_id", "bundle_digest")):
            raise ValueError(
                "Release candidate bundle identity differs from latest executed build"
            )
    refresh = next(
        case["details"] for case in cases if case["name"] == "environment_lock.refresh"
    )
    if any(
        refresh.get(key) is not True
        for key in (
            "drift_rejected",
            "prior_success_preserved",
            "refresh_passed",
            "model_reused",
            "workflow_reused",
        )
    ) or not re.fullmatch(
        r"[0-9a-f]{64}", str(refresh.get("environment_lock_digest", ""))
    ):
        raise ValueError("Environment lock drift/refresh evidence is missing")


def extract_summary(output):
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    if begin < 0 or end < 0:
        raise ValueError("No complete pnms application QA summary in job logs")
    report = json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())
    if not isinstance(report, dict):
        raise ValueError("pnms QA summary must be an object")
    return report


def docker_free_path(original, destination):
    """Keep installed host tools available while Docker/Lepton are absent."""
    destination.mkdir(parents=True)
    blocked = {"docker", "docker-compose", "lep"}
    for entry in original.split(os.pathsep):
        if not entry or not Path(entry).is_dir():
            continue
        for source in Path(entry).iterdir():
            target = destination / source.name
            if (
                source.name not in blocked
                and not target.exists()
                and source.is_file()
                and os.access(source, os.X_OK)
            ):
                target.symlink_to(source.absolute())
    if any(shutil.which(name, path=str(destination)) for name in blocked):
        raise ValueError("Docker/Lepton unexpectedly visible in isolated QA PATH")
    return str(destination)


class Context:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.repository = args.repository.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.environment = dict(os.environ)
        self.environment.pop("PYTHONPATH", None)
        self.environment.pop("PYTHONHOME", None)
        # Torch/CMake infer the toolkit from nvcc's parent directories. Retain
        # its true prefix when the tool-only PATH exposes nvcc through a link.
        nvcc = shutil.which("nvcc")
        if nvcc:
            toolkit = str(Path(nvcc).resolve().parent.parent)
            self.environment.setdefault("CUDA_HOME", toolkit)
            self.environment.setdefault("CUDAToolkit_ROOT", toolkit)
            self.environment.setdefault("CUDACXX", str(Path(nvcc).resolve()))
        self.environment["PATH"] = docker_free_path(
            os.environ.get("PATH", ""), self.root / "host-tools"
        )
        self.environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        self.report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image_digest,
            "profile": "local",
            "status": "running",
            "qa_scope": "local-workflow",
            "expected_cases": expected_cases(),
            "cases": [],
            "http_qa": "not_run",
            "release_candidates": [],
        }

    def save(self):
        write_json(self.root / "summary.json", self.report)

    def case(self, name, function):
        if name not in self.report["expected_cases"] or any(
            item["name"] == name for item in self.report["cases"]
        ):
            raise ValueError("Unknown or repeated QA case: " + name)
        record = {"name": name, "status": "running"}
        self.report["cases"].append(record)
        self.save()
        started = time.monotonic()
        try:
            details = function()
            record.update(status="passed", details=details or {})
            return details
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            self.save()

    def command(self, label, arguments, *, success=True, structured=True):
        command = [str(value) for value in arguments]
        logs = self.root / "logs" / label
        logs.mkdir(parents=True)
        write_json(
            logs / "command.json", {"argv": command, "cwd": str(self.repository)}
        )
        with (
            (logs / "stdout.log").open("w") as out,
            (logs / "stderr.log").open("w") as err,
        ):
            process = subprocess.Popen(
                command,
                stdout=out,
                stderr=err,
                env=self.environment,
                cwd=self.repository,
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
        write_json(logs / "exit.json", {"exit_code": code})
        if (code == 0) != success:
            raise RuntimeError(f"{label}: unexpected exit {code}; evidence: {logs}")
        if not success or not structured:
            return {"exit_code": code}
        result = json.loads((logs / "stdout.log").read_text())
        write_json(logs / "result.json", result)
        return result

    def pnms(self, label, *arguments, success=True):
        return self.command(
            label,
            [
                self.args.python,
                self.repository / "pnms",
                *arguments,
                "--repository",
                self.repository,
            ],
            success=success,
        )


def preflight(ctx):
    metadata = json.loads(ctx.args.source_metadata.read_text())
    if metadata.get("source_sha") != ctx.args.expected_source_sha:
        raise ValueError("Prepared pnms QA image source SHA differs")
    # Source archives on the shared mount do not need a .git directory. Bind
    # their receipts to the independently selected immutable QA image metadata.
    ctx.environment["PNMS_SOURCE_SHA"] = metadata["source_sha"]
    for path in (
        ctx.repository / "pnms",
        ctx.repository / "application-cli/src/pnms_app/cli.py",
        ctx.args.sdk_prefix / "bin/physicsnemo-infer",
    ):
        if not path.is_file():
            raise ValueError(f"Prepared QA file is missing: {path}")
    import torch
    import tensorrt

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("pnms local QA requires exactly one allocated CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 9 or properties.minor != 0 or "H100" not in properties.name:
        raise ValueError("This QA suite requires the configured H100 target")
    return {
        "gpu": properties.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
        "docker": "absent",
        "lepton_cli": "absent",
        "pnms_sha256": sha256(ctx.repository / "pnms"),
        "runner_sha256": sha256(Path(__file__)),
    }


def python_cases(ctx, kind):
    app = ctx.root / "projects" / ("python-" + kind)
    prefix = "python." + kind
    if kind == "new":
        ctx.case(
            prefix + ".init",
            lambda: ctx.pnms(prefix + ".init", "init", app, "--runtime", "python"),
        )
    else:
        ctx.case(
            prefix + ".init",
            lambda: ctx.command(
                prefix + ".init",
                [
                    ctx.args.python,
                    ctx.repository / "scripts/plugin_dev.py",
                    "init",
                    app,
                ],
                structured=False,
            ),
        )
    ctx.case(prefix + ".check", lambda: ctx.pnms(prefix + ".check", "check", app))
    ctx.case(prefix + ".build", lambda: ctx.pnms(prefix + ".build", "build", app))
    request = ctx.root / "requests" / (kind + ".json")
    write_json(request, {"value": 17})

    def run():
        result = ctx.pnms(prefix + ".run", "run", app, "--input", request)
        if result["result"].get("doubled") != 34:
            raise ValueError(
                "Python public pnms run differs from independent expected output"
            )
        return {"expected": 34, "actual": result["result"]["doubled"]}

    ctx.case(prefix + ".run", run)

    def edit():
        workflow = app / "workflow.py"
        original = workflow.read_text()
        if "value * 2" not in original:
            raise ValueError(
                "Generated Python fixture no longer matches QA edit scenario"
            )
        workflow.write_text(original.replace("value * 2", "value * 3"))
        normal = ctx.pnms(prefix + ".snapshot", "run", app, "--input", request)
        dev = ctx.pnms(prefix + ".dev", "run", app, "--input", request, "--dev")
        if normal["result"].get("doubled") != 34 or dev["result"].get("doubled") != 51:
            raise ValueError(
                "Python snapshot/dev behavior does not preserve source ownership"
            )
        return {"snapshot": 34, "development": 51}

    ctx.case(prefix + ".dev_edit", edit)


def native_cases(ctx, backend, template, ingress="application/json"):
    selection = {"backend": backend, "template": template, "ingress": ingress}
    prefix = candidate_name(selection)
    files = ingress == "multipart/form-data"
    app = ctx.root / "projects" / prefix.removeprefix("native.").replace(".", "-")
    options = [
        "--environment",
        "local",
        "--python",
        ctx.args.python,
        "--runtime",
        ctx.args.sdk_prefix / "bin/physicsnemo-infer",
        "--sdk-prefix",
        ctx.args.sdk_prefix,
        "--gpu",
        "0",
    ]

    def initialize():
        result = ctx.pnms(
            prefix + ".init",
            "init",
            app,
            "--runtime",
            "cpp",
            "--backend",
            backend,
            "--template",
            template,
        )
        if files:
            configure_file_project(app)
        return result

    ctx.case(prefix + ".init", initialize)
    ctx.case(
        prefix + ".check",
        lambda: ctx.pnms(prefix + ".check", "check", app, "--config-only", *options),
    )
    built = None

    def build():
        nonlocal built
        built = ctx.pnms(prefix + ".build", "build", app, *options)
        if (
            built.get("status") != "complete"
            or built.get("checks", {}).get("model") != "passed"
            or built["checks"].get("application") != "passed"
        ):
            raise ValueError("Public pnms build did not pass model/application checks")
        primary = built["models"]["primary"]
        verify_build(
            Path(built["root"]) / primary["output_root"], [backend], case_count=3
        )
        candidate = {
            **selection,
            "build_id": built["build_id"],
            "bundle_digest": built["bundle_digest"],
            "bundle": str(Path(built["root"]) / built["bundle"]),
            "build_receipt": str(Path(built["root"]) / "build.json"),
            "http_qa": "not_run",
        }
        ctx.report["release_candidates"].append(candidate)
        return {
            **candidate,
            "model_identity": primary["identity"],
            "native_checks": {"verified": True, "case_count": 3},
        }

    ctx.case(prefix + ".build", build)
    request = ctx.root / "requests" / (app.name + ".json")
    raw_input = request.with_suffix(".f32")
    input_options = ["--input-file", "features=" + str(raw_input)] if files else []

    def held_out():
        for index, values in enumerate(
            ([10, -3, 0.5, 7.25], [-8, 32, -0.125, 0], [1.5, -16, 128, 0.25])
        ):
            write_json(request, raw_parameters() if files else {"values": values})
            if files:
                raw_input.write_bytes(struct.pack("<4f", *values))
            result = ctx.pnms(
                prefix + f".held-{index}",
                "run",
                app,
                "--input",
                request,
                *input_options,
            )
            actual = result["result"]["payload"]["values"]
            expected = [2 * value + 1 for value in values]
            if len(actual) != 4 or any(
                not math.isfinite(a)
                or not math.isclose(a, e, rel_tol=1e-5, abs_tol=1e-6)
                for a, e in zip(actual, expected)
            ):
                raise ValueError("Native held-out parity failed")
        return {
            "case_count": 3,
            "parity": "passed",
            "input_kind": "raw-fp32-le" if files else "json-array-fp32",
        }

    ctx.case(prefix + ".held_out", held_out)

    def repeat():
        repeated = ctx.pnms(prefix + ".repeat", "build", app, *options)
        if (
            not repeated["models"]["primary"].get("reused")
            or not repeated.get("workflow", {}).get("reused")
            or repeated.get("environment_lock", {}).get("digest")
            != built.get("environment_lock", {}).get("digest")
            or repeated["models"]["primary"]["identity"]
            != built["models"]["primary"]["identity"]
            or repeated.get("status") != "complete"
            or repeated.get("checks", {}).get("application") != "passed"
            or repeated.get("checks", {}).get("model") != "passed"
        ):
            raise ValueError("Repeat build did not reuse the sealed model output")
        latest = {
            "build_id": repeated["build_id"],
            "bundle_digest": repeated["bundle_digest"],
            "bundle": str(Path(repeated["root"]) / repeated["bundle"]),
            "build_receipt": str(Path(repeated["root"]) / "build.json"),
        }
        for candidate in ctx.report["release_candidates"]:
            if candidate_key(candidate) == candidate_key(selection):
                candidate.update(latest)
        return {
            "model_reused": True,
            "workflow_reused": True,
            "environment_lock_digest": repeated.get("environment_lock", {}).get(
                "digest"
            ),
            **latest,
        }

    ctx.case(prefix + ".cache_reuse", repeat)
    if selection == {
        "backend": "aoti",
        "template": "tensor",
        "ingress": "application/json",
    }:
        ctx.case("environment_lock.refresh", lambda: lock_refresh(ctx))

    def relocate():
        relocated = ctx.root / "relocated" / app.name
        relocated.parent.mkdir(exist_ok=True)
        app.rename(relocated)
        for candidate in ctx.report["release_candidates"]:
            if candidate_key(candidate) == candidate_key(selection):
                candidate["bundle"] = str(
                    relocated / Path(candidate["bundle"]).relative_to(app)
                )
                candidate["build_receipt"] = str(
                    relocated / Path(candidate["build_receipt"]).relative_to(app)
                )
        result = ctx.pnms(
            prefix + ".relocated", "run", relocated, "--input", request, *input_options
        )
        if result["result"]["payload"].get("values") != [4, -31, 257, 1.5]:
            raise ValueError("Relocated application changed output")
        return {"producer_path_removed": not app.exists(), "parity": "passed"}

    ctx.case(prefix + ".relocation", relocate)

    def invalid():
        write_json(request, {"values": [1, 2, 3]})
        result = ctx.pnms(
            prefix + ".invalid",
            "run",
            ctx.root / "relocated" / app.name,
            "--input",
            request,
            *input_options,
            success=False,
        )
        return {"invalid_shape_rejected": result["exit_code"] != 0}

    ctx.case(prefix + ".reject_invalid", invalid)


def lock_refresh(ctx):
    candidate = next(
        item
        for item in ctx.report["release_candidates"]
        if candidate_key(item) == ("aoti", "tensor", "application/json")
    )
    app = Path(candidate["build_receipt"]).parents[3]
    lock = app / "pnms.lock"
    original = json.loads(lock.read_text())
    previous = (app / ".pnms/last-build.json").read_bytes()
    original_digest = next(
        case["details"]["environment_lock_digest"]
        for case in ctx.report["cases"]
        if case["name"] == "native.aoti.tensor.cache_reuse"
    )
    original["native"]["gpu"]["name"] = "intentionally-unmatched-qa-target"
    write_json(lock, original)
    changed = lock.read_bytes()
    options = [
        "--environment",
        "local",
        "--python",
        ctx.args.python,
        "--runtime",
        ctx.args.sdk_prefix / "bin/physicsnemo-infer",
        "--sdk-prefix",
        ctx.args.sdk_prefix,
        "--gpu",
        "0",
    ]
    ctx.pnms("lock.drift", "build", app, *options, success=False)
    diagnostic = (ctx.root / "logs/lock.drift/stderr.log").read_text()
    if "pnms.lock" not in diagnostic or "--refresh-lock" not in diagnostic:
        raise ValueError("Changed lock did not produce its public refresh diagnostic")
    if (
        lock.read_bytes() != changed
        or (app / ".pnms/last-build.json").read_bytes() != previous
    ):
        raise ValueError("Rejected drift changed the lock or successful build pointer")
    result = ctx.pnms("lock.refresh", "build", app, "--refresh-lock", *options)
    if (
        result.get("status") != "complete"
        or result.get("environment_lock", {}).get("digest") != original_digest
        or not result["models"]["primary"].get("reused")
        or not result["workflow"].get("reused")
    ):
        raise ValueError(
            "Explicit lock refresh did not restore the observed stack with verified cache reuse"
        )
    latest = {
        "build_id": result["build_id"],
        "bundle_digest": result["bundle_digest"],
        "bundle": str(Path(result["root"]) / result["bundle"]),
        "build_receipt": str(Path(result["root"]) / "build.json"),
    }
    candidate.update(latest)
    return {
        "drift_rejected": True,
        "prior_success_preserved": True,
        "refresh_passed": True,
        "model_reused": True,
        "workflow_reused": True,
        "environment_lock_digest": original_digest,
        **latest,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=Path("/workspace"))
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--sdk-prefix", type=Path, default=Path("/opt/physicsnemo-inference")
    )
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-source-sha", required=True)
    result.add_argument("--image-digest", required=True)
    result.add_argument("--profile", choices=("local",), default="local")
    result.add_argument("--command-timeout", type=float, default=1200)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    ctx = Context(args)
    code = 1
    try:
        ctx.case("environment", lambda: preflight(ctx))
        for kind in ("new", "legacy"):
            python_cases(ctx, kind)
        for backend, template, ingress in NATIVE_TARGETS:
            native_cases(ctx, backend, template, ingress)
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
