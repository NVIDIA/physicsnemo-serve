#!/usr/bin/env python3
"""Package a verified public pnms build and qualify its exact image on Lepton.

Native image acceptance uses sealed application fixtures for JSON tensor/custom
workflows. Packaging never substitutes for GPU acceptance.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "application-cli/src"))

from pnms_app import bundle, images  # noqa: E402
from pnms_app.common import (  # noqa: E402
    contained,
    digest,
    file_identity,
    read_json,
    require,
    verify_inventory,
    write_json,
)
from pnms_app.config import UniqueLoader  # noqa: E402
import yaml  # noqa: E402
from qa.native_inference import contract  # noqa: E402
from scripts import run_lepton_pnms_http_qa as http  # noqa: E402


QA_OPTIONS = (
    "workspace_id",
    "workspace_url",
    "node_group",
    "pull_secret",
    "resource_shape",
    "nfs_mount_base",
    "lustre_dir",
    "lustre_storage",
    "mount_target",
    "job_timeout",
    "poll_interval",
    "command_timeout",
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-inputs", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument(
        "--image", required=True, help="Destination registry/repository:tag"
    )
    parser.add_argument("--promote", help="Optional tag for the exact accepted digest")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=ROOT)
    defaults = http.build_parser()
    for name in QA_OPTIONS:
        value = defaults.get_default(name)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            default=value,
            type=float if name.endswith("timeout") or name == "poll_interval" else str,
        )
    return parser


def _immutable(value):
    return isinstance(value, str) and re.fullmatch(
        r"[^\s@]+@sha256:[a-f0-9]{64}", value
    )


def _verify_source(args, inputs, handoff_root, *, destination=None):
    """Bind Docker's source context to the producer's original frozen bytes."""
    value = inputs.get("source_inventory")
    require(isinstance(value, str) and value, "Verified source inventory is required")
    submitted = handoff_root / value
    inventory_path = submitted.resolve()
    require(
        not submitted.is_symlink()
        and inventory_path.is_relative_to(handoff_root)
        and inventory_path.is_file(),
        "Source inventory must remain in the verified handoff",
    )
    identity = file_identity(inventory_path)
    require(
        identity == inputs.get("source_inventory_identity")
        and identity["sha256"] == inputs.get("source_inventory_sha256"),
        "Source inventory bytes differ from the producer handoff",
    )
    records = read_json(inventory_path)
    require(isinstance(records, dict) and records, "Frozen source inventory is empty")
    repository = args.repository.resolve()
    require(repository.is_dir(), "Packaging repository directory is missing")
    if destination is not None:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=False)
    for relative, record in records.items():
        path = PurePosixPath(relative)
        require(
            relative
            and not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == relative
            and relative != "."
            and "\\" not in relative
            and "\0" not in relative,
            "Unsafe frozen source path: " + relative,
        )
        require(
            isinstance(record, dict)
            and set(record) == {"sha256", "size_bytes", "mode", "symlink"}
            and re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256", "")))
            and type(record.get("size_bytes")) is int
            and record["size_bytes"] >= 0
            and type(record.get("mode")) is int
            and 0 <= record["mode"] <= 0o777
            and record.get("symlink") is None,
            "Invalid frozen source record: " + relative,
        )
        selected = repository / relative
        require(
            selected.resolve().is_relative_to(repository),
            "Frozen source path escapes packaging repository",
        )
        cursor = selected
        while cursor != repository:
            require(
                not cursor.is_symlink(),
                "Packaging source path is a symlink: " + relative,
            )
            cursor = cursor.parent
        require(
            selected.is_file()
            and stat.S_ISREG(selected.stat().st_mode)
            and stat.S_IMODE(selected.stat().st_mode) == record["mode"]
            and file_identity(selected)
            == {"sha256": record["sha256"], "size_bytes": record["size_bytes"]},
            "Packaging source differs from frozen producer source: " + relative,
        )
        if destination is not None:
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(selected, copied)
            copied.chmod(record["mode"])
            require(
                file_identity(copied)
                == {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
                and stat.S_IMODE(copied.stat().st_mode) == record["mode"],
                "Copied packaging source differs from frozen producer source: "
                + relative,
            )
    proof = {
        "status": "passed",
        "inventory": identity,
        "file_count": len(records),
        "repository": str(repository),
    }
    if destination is not None:
        proof["snapshot"] = str(destination)
    return proof


def _http_args(args, inputs, image, runtime, *, python_receipt=None):
    values = [
        "--image",
        image,
        "--expected-source-sha",
        args.expected_source_sha,
        "--run-id",
        args.run_id + "-http",
        "--artifact-dir",
        str(args.artifact_dir),
        "--mode",
        "image",
        "--image-runtime",
        runtime,
        "--runner",
        inputs["http_runner"],
        "--repository",
        "/opt/pnms/source" if runtime == "cpp" else "/app",
    ]
    for name in QA_OPTIONS:
        value = getattr(args, name)
        if value is not None:
            values += ["--" + name.replace("_", "-"), str(value)]
    if python_receipt:
        values += ["--python-build-receipt", str(python_receipt)]
    if runtime == "cpp":
        values += ["--application-fixtures", "--release-acceptance"]
    return values


def _inputs(args):
    require(
        re.fullmatch(r"[a-f0-9]{40}", args.expected_source_sha),
        "A full source SHA is required",
    )
    require(
        _immutable(args.runtime_image),
        "Runtime image must be an immutable registry digest",
    )
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,34}", args.run_id),
        "Release run ID must leave room for the HTTP suffix",
    )
    require(
        re.fullmatch(
            r"[a-z0-9][a-z0-9._:/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", args.image
        ),
        "Destination must be registry/repository:tag",
    )
    require(
        not args.promote
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", args.promote),
        "Promotion must name a registry tag",
    )
    inputs_path = args.release_inputs.resolve()
    inputs = read_json(inputs_path)
    require(
        inputs.get("schema_version") == 1 and inputs.get("status") == "passed",
        "Release handoff must have passed verification",
    )
    require(
        inputs.get("source_sha") == args.expected_source_sha
        and inputs.get("runtime_image") == args.runtime_image,
        "Release source or runtime image differs from the verified handoff",
    )
    inputs["_source_verification"] = _verify_source(args, inputs, inputs_path.parent)
    project = Path(inputs["project"])
    receipt = Path(inputs["build_receipt"])
    project = (inputs_path.parent / project).resolve()
    receipt = (inputs_path.parent / receipt).resolve()
    require(
        project.is_dir()
        and project.is_relative_to(inputs_path.parent)
        and receipt.is_relative_to(project),
        "Release inputs must remain in the verified local project handoff",
    )
    require(
        file_identity(receipt) == inputs.get("build_receipt_identity"),
        "Completed build receipt differs from verified handoff",
    )
    built = read_json(receipt)
    runtime = built.get("runtime")
    require(
        built.get("schema_version") == 1
        and built.get("status") == "complete"
        and runtime in ("cpp", "python"),
        "A completed public application build is required",
    )
    require(
        built.get("source_sha") == args.expected_source_sha
        and built.get("build_id") == inputs.get("build_id")
        and inputs.get("runtime") == runtime
        and re.fullmatch(r"[a-f0-9]{32}", str(built.get("build_id", ""))),
        "Completed build source, runtime or build identity differs",
    )
    project_doc = yaml.load((project / "pnms.yaml").read_text(), Loader=UniqueLoader)
    require(
        project_doc.get("name") == built.get("name")
        and project_doc.get("runtime") == runtime
        and project_doc.get("serve", {}).get("image") == args.runtime_image,
        "Project, completed build and selected runtime image differ",
    )
    require(
        inputs.get("http_runner_sha256") == contract.sha256(http.job.__file__)
        and inputs.get("contract_sha256") == contract.sha256(contract.__file__),
        "Staged HTTP runner or contract differs from this controller",
    )
    runner = PurePosixPath(inputs["http_runner"])
    require(
        runner.is_absolute()
        and ".." not in runner.parts
        and runner.is_relative_to(PurePosixPath(args.mount_target))
        and runner.name == "pnms_http_job.py",
        "HTTP runner must be staged below the shared mount",
    )
    descriptor = None
    if runtime == "cpp":
        require(
            built.get("checks", {}).get("model") == "passed"
            and built.get("checks", {}).get("application") == "passed",
            "Native model and application checks must pass before release",
        )
        require(
            built.get("bundle_digest") == inputs.get("bundle_digest"),
            "Build bundle differs from verified handoff",
        )
        location = contained(receipt.parent, built["bundle"])
        bundle.verify(location, built["bundle_digest"])
        descriptor = read_json(location / "bundle.json")
        require(
            descriptor.get("runtime") == runtime
            and descriptor.get("name") == built["name"]
            and descriptor.get("source_sha") == args.expected_source_sha
            and descriptor.get("backend") in ("aoti", "tensorrt"),
            "Sealed application source, backend or workflow differs",
        )
        fixtures = contained(location, "fixtures/cases.json")
        cases = read_json(fixtures).get("cases")
        require(
            isinstance(cases, list) and cases,
            "Application release requires sealed numerical fixtures",
        )
        descriptor["application_fixtures"] = {
            "path": "fixtures/cases.json",
            "sha256": file_identity(fixtures)["sha256"],
            "case_count": len(cases),
        }
        cases = http.job.application_fixture_cases(
            {
                "bundle": str(location),
                "application_fixtures": descriptor["application_fixtures"],
                "release_acceptance": {
                    "schema_version": 1,
                    "fixture_indices": list(range(len(cases))),
                },
                "input_files": descriptor.get("input_files", {}),
            }
        )
        descriptor["fixture_names"] = [case["name"] for case in cases]
        descriptor["fixture_cases"] = cases
        configuration = read_json(contained(location, descriptor["workflow"]["config"]))
        template = {"tensor.v1": "tensor", "custom.v1": "custom"}.get(
            configuration.get("adapter")
        )
        require(template, "Release HTTP acceptance supports tensor/custom workflows")
        descriptor["target_name"] = "native." + descriptor["backend"] + "." + template
    else:
        verify_inventory(
            contained(receipt.parent, built["python"]["plugin_root"]),
            built["python"]["files"],
        )
    return inputs, project, receipt, built, descriptor


def _command(argv, *, root, name, runner, timeout, env=None):
    argv = [str(value) for value in argv]
    write_json(root / (name + "-command.json"), argv)
    try:
        result = runner(
            argv, capture_output=True, text=True, check=False, timeout=timeout, env=env
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"{name} could not finish: {error}") from error
    (root / (name + ".log")).write_text((result.stdout or "") + (result.stderr or ""))
    require(result.returncode == 0, f"{name} failed; see {root / (name + '.log')}")
    return result.stdout


def _packaged(args, result, receipt, built):
    require(
        result.get("schema_version") == 1
        and result.get("status") == "complete"
        and result.get("build_id") == built["build_id"]
        and Path(result.get("root", "")).resolve() == receipt.parent,
        "Packaging did not preserve the selected completed build",
    )
    image = result.get("image", {})
    reference = image.get("registry_digest")
    require(
        _immutable(reference)
        and images._repository(reference) == images._repository(args.image),
        "Published registry digest belongs to another destination",
    )
    require(
        image.get("status") == "complete"
        and image.get("runtime_image") == args.runtime_image
        and image.get("local_tag") == args.image
        and re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("image_config_digest", "")),
        "Published image status, base or configuration identity differs",
    )
    checks, contents = image.get("checks", {}), image.get("contents", {})
    require(
        checks.get("contents") == "passed" and contents.get("status") == "passed",
        "Published image contents were not verified",
    )
    if built["runtime"] == "cpp":
        require(
            image.get("bundle_digest") == built["bundle_digest"]
            and contents.get("bundle_digest") == built["bundle_digest"]
            and contents.get("source_sha") == args.expected_source_sha
            and checks.get("stack")
            == checks.get("native_load")
            == contents.get("native_load")
            == "passed",
            "Packaged native bundle, source or provider checks differ",
        )
    else:
        require(
            checks.get("python_base") == "passed"
            and isinstance(contents.get("workflow_id"), str)
            and contents["workflow_id"]
            and contents.get("plugin_digest") == digest(built["python"]["files"]),
            "Packaged Python workflow or source snapshot differs",
        )
    return image


def _qualified(args, inputs, image, built, descriptor, python_receipt):
    selected = http.build_parser().parse_args(
        _http_args(
            args,
            inputs,
            image["registry_digest"],
            built["runtime"],
            python_receipt=python_receipt,
        )
    )
    http.validate_args(selected)
    qa_root = args.artifact_dir / selected.run_id
    lifecycle = read_json(qa_root / "summary.json")
    require(
        lifecycle.get("status") == "passed"
        and lifecycle.get("job_state") == "Completed"
        and lifecycle.get("cleanup_errors") == []
        and lifecycle.get("run_id") == selected.run_id
        and lifecycle.get("source_sha") == args.expected_source_sha
        and lifecycle.get("image_digest") == image["registry_digest"]
        and lifecycle.get("job_id"),
        "Exact-image acceptance or owned-job cleanup did not complete",
    )
    report = read_json(qa_root / "job-summary.json")
    http.validate_report(report, args=selected)
    require(
        report.get("qa_scope") == "image-http" and len(report["targets"]) == 1,
        "Release requires exactly one image HTTP target",
    )
    target = report["targets"][0]
    workflow = (
        built["name"] if built["runtime"] == "cpp" else image["contents"]["workflow_id"]
    )
    require(
        target.get("runtime") == built["runtime"]
        and target.get("workflow") == workflow,
        "HTTP workflow differs from selected producer build",
    )
    if built["runtime"] == "cpp":
        require(
            target.get("bundle_digest") == built["bundle_digest"]
            and target.get("name") == descriptor["target_name"],
            "HTTP image bundle, backend or template differs from selected producer build",
        )
        require(
            target.get("application_fixtures") == descriptor["application_fixtures"],
            "HTTP fixtures differ from the selected producer's sealed bundle",
        )
        require(
            target.get("release_acceptance")
            == {
                "schema_version": 1,
                "fixture_indices": list(range(len(descriptor["fixture_names"]))),
            },
            "Promotable image requires expanded release acceptance",
        )
        for operation, index in [
            (f"fixture.{index}", index)
            for index in range(len(descriptor["fixture_names"]))
        ]:
            evidence = next(
                case["details"]["application_fixture"]
                for case in report["cases"]
                if case["name"] == target["name"] + "." + operation
            )
            require(
                evidence.get("name") == descriptor["fixture_names"][index],
                "HTTP request used another named application fixture",
            )
            details = next(
                case["details"]
                for case in report["cases"]
                if case["name"] == target["name"] + "." + operation
            )
            fixture = descriptor["fixture_cases"][index]
            require(
                details.get("artifact_downloads")
                == fixture.get("expected_artifacts", {}),
                "HTTP downloaded artifacts differ from sealed authored goldens",
            )
            if fixture.get("files"):
                require(
                    details.get("transport") == "multipart/form-data"
                    and details.get("input_files") == fixture["files"],
                    "HTTP uploaded files differ from the sealed authored fixture",
                )
    return {
        "lifecycle": file_identity(qa_root / "summary.json"),
        "report": file_identity(qa_root / "job-summary.json"),
        "job_id": lifecycle["job_id"],
        "image": image["registry_digest"],
        "case_count": len(report["cases"]),
        "status": "passed",
        "cleanup_errors": [],
    }


def run(args, *, runner=None):
    runner = runner or subprocess.run
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,34}", args.run_id),
        "Release run ID must leave room for the HTTP suffix",
    )
    root = args.artifact_dir / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "stage": "inputs",
        "source_sha": args.expected_source_sha,
    }
    try:
        inputs, project, receipt, built, descriptor = _inputs(args)
        original = file_identity(receipt)
        record.update(
            build_id=built["build_id"],
            runtime=built["runtime"],
            build_receipt=original,
            release_inputs=file_identity(args.release_inputs),
            bundle_digest=built.get("bundle_digest"),
            packaging_source=inputs["_source_verification"],
        )
        # Validate Lepton configuration before publishing an image. Python's
        # additional receipt binding becomes available after packaging.
        preflight = http.build_parser().parse_args(
            _http_args(
                args,
                inputs,
                args.image.rsplit(":", 1)[0] + "@sha256:" + "0" * 64,
                built["runtime"],
            )
        )
        http.lifecycle.validate_args(preflight)
        require(
            not (args.artifact_dir / preflight.run_id).exists(),
            "HTTP evidence directory already exists; choose a fresh release run ID",
        )
        if built["runtime"] == "cpp":
            http.validate_args(preflight)
        record["stage"] = "source-snapshot"
        context = root / "packaging-source"
        record["packaging_source"] = _verify_source(
            args, inputs, args.release_inputs.resolve().parent, destination=context
        )
        inventory = root / "source-inventory.json"
        shutil.copyfile(
            args.release_inputs.resolve().parent / inputs["source_inventory"], inventory
        )
        require(
            file_identity(inventory) == record["packaging_source"]["inventory"],
            "Packaging source inventory changed after verification",
        )
        metadata = root / "source.json"
        write_json(
            metadata,
            {
                "source_sha": args.expected_source_sha,
                "source_dirty": False,
                "source_inventory_sha256": inputs["source_inventory_sha256"],
            },
        )
        record["stage"] = "packaging"
        output = _command(
            [
                sys.executable,
                "-B",
                context / "pnms",
                "build",
                project,
                "--repository",
                context,
                "--from-build",
                receipt,
                "--image",
                args.image,
                "--push",
            ],
            root=root,
            name="package",
            runner=runner,
            timeout=7200,
            env=os.environ | {"PNMS_SOURCE_METADATA": str(metadata.resolve())},
        )
        import json

        packaged = json.loads(output)
        write_json(root / "package-result.json", packaged)
        require(
            file_identity(receipt) == original,
            "Packaging mutated the original completed build receipt",
        )
        image = _packaged(args, packaged, receipt, built)
        record.update(
            image=image["registry_digest"],
            image_config_digest=image["image_config_digest"],
        )
        python_receipt = None
        if built["runtime"] == "python":
            python_receipt = root / "python-image-build.json"
            write_json(python_receipt, {**built, "image": image})
        record["stage"] = "acceptance"
        _command(
            [
                sys.executable,
                "-B",
                context / "scripts/run_lepton_pnms_http_qa.py",
                *_http_args(
                    args,
                    inputs,
                    image["registry_digest"],
                    built["runtime"],
                    python_receipt=python_receipt,
                ),
            ],
            root=root,
            name="acceptance",
            runner=runner,
            timeout=args.job_timeout + 1800,
        )
        record["acceptance"] = _qualified(
            args, inputs, image, built, descriptor, python_receipt
        )
        require(
            file_identity(receipt) == original,
            "Completed build receipt changed during acceptance",
        )
        if args.promote:
            record["stage"] = "promotion"
            destination = args.image.rsplit(":", 1)[0] + ":" + args.promote
            _command(
                [
                    "docker",
                    "buildx",
                    "imagetools",
                    "create",
                    "--prefer-index=false",
                    "--tag",
                    destination,
                    image["registry_digest"],
                ],
                root=root,
                name="promote",
                runner=runner,
                timeout=600,
            )
            observed = json.loads(
                _command(
                    [
                        "docker",
                        "buildx",
                        "imagetools",
                        "inspect",
                        "--format",
                        "{{json .Manifest.Digest}}",
                        destination,
                    ],
                    root=root,
                    name="promotion-inspect",
                    runner=runner,
                    timeout=300,
                )
            )
            require(
                observed == image["registry_digest"].split("@", 1)[1],
                "Promotion changed the accepted image digest",
            )
            record["promotion"] = {
                "tag": destination,
                "image": image["registry_digest"],
                "status": "passed",
            }
        record.update(status="passed", stage="complete")
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_json(root / "release.json", record)
    return record


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
