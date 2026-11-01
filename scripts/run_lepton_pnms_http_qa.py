#!/usr/bin/env python3
"""Submit pnms host or exact-image HTTP acceptance as one finite Lepton job."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import math
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_http_job as job  # noqa: E402
from qa.native_inference import pnms_local_job as local_job  # noqa: E402


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("http",))
    parser.add_argument("--mode", choices=("host", "image"), default="host")
    parser.add_argument("--image-runtime", choices=("cpp", "python"), default="cpp")
    parser.add_argument(
        "--application-fixtures",
        action="store_true",
        help="Native image QA: replay the first two sealed JSON application fixtures",
    )
    parser.add_argument(
        "--release-acceptance",
        action="store_true",
        help="Require all fixtures and native worker fault gates for promotion",
    )
    parser.add_argument("--python-build-receipt", type=Path)
    parser.add_argument("--build-receipt", action="append", default=[])
    parser.add_argument("--local-summary")
    parser.add_argument("--include-python", action="store_true")
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument(
        "--runner",
        help="Mounted copy of the standalone HTTP driver when testing an ordinary release image",
    )
    parser.add_argument("--python", default="python3")
    parser.add_argument("--bundle-root", default="/opt/physicsnemo-release")
    parser.add_argument("--service-bin-dir")
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=240)
    return parser


def _python_release_receipt(args):
    job.require(
        args.python_build_receipt and args.python_build_receipt.is_file(),
        "Python image QA requires a public pnms build receipt",
    )
    document = json.loads(args.python_build_receipt.read_text())
    image = document.get("image", {})
    contents = image.get("contents", {})
    job.require(
        document.get("status") == "complete"
        and document.get("runtime") == "python"
        and document.get("source_sha") == args.expected_source_sha
        and image.get("status") == "complete"
        and image.get("registry_digest") == args.image
        and image.get("checks", {}).get("contents") == "passed"
        and image.get("checks", {}).get("python_base") == "passed"
        and contents.get("status") == "passed"
        and isinstance(contents.get("workflow_id"), str)
        and contents["workflow_id"]
        and isinstance(document.get("name"), str)
        and document["name"]
        and re.fullmatch(r"[a-f0-9]{64}", contents.get("plugin_digest", "")),
        "Python release image/content/source differs from the completed public build receipt",
    )
    return {
        "workflow": contents["workflow_id"],
        "plugin_digest": contents["plugin_digest"],
        "receipt_sha256": job.sha256(args.python_build_receipt),
    }


def validate_args(args):
    lifecycle.validate_args(args)
    job.require(
        not args.application_fixtures
        or args.mode == "image"
        and args.image_runtime == "cpp",
        "Application fixtures apply only to native image QA",
    )
    job.require(
        not args.release_acceptance
        or args.application_fixtures
        and args.mode == "image"
        and args.image_runtime == "cpp",
        "Release acceptance requires native image application fixtures",
    )
    for value in (args.startup_timeout, args.request_timeout, args.shutdown_timeout):
        job.require(
            math.isfinite(value) and value > 0,
            "HTTP QA deadlines must be finite and positive",
        )
    if args.mode == "host":
        job.require(
            args.image_runtime == "cpp" and not args.python_build_receipt,
            "Python release receipt applies to Python image QA",
        )
        job.require(
            bool(args.build_receipt) != bool(args.local_summary),
            "Host HTTP QA requires --local-summary or --build-receipt",
        )
    else:
        job.require(
            not args.build_receipt
            and not args.local_summary
            and not args.include_python,
            "Image QA uses its baked native bundle",
        )
        if args.image_runtime == "python":
            _python_release_receipt(args)
        else:
            job.require(
                not args.python_build_receipt,
                "Python release receipt applies to Python image QA",
            )
        job.require(
            args.bundle_root == "/opt/physicsnemo-release",
            "Owned image QA must use its baked bundle at /opt/physicsnemo-release",
        )
    mount = PurePosixPath(args.mount_target)
    for value in [args.local_summary] if args.local_summary else args.build_receipt:
        path = PurePosixPath(value)
        job.require(
            path.is_absolute()
            and ".." not in path.parts
            and path.is_relative_to(mount),
            "HTTP QA input must be inside --mount-target without traversal",
        )
    job.require(
        len(set(args.build_receipt)) == len(args.build_receipt),
        "Duplicate build receipts",
    )


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    runner = [
        "exec",
        "/opt/nvidia/nvidia_entrypoint.sh",
        args.python,
        "-B",
        args.runner
        or str(PurePosixPath(args.repository) / "qa/native_inference/pnms_http_job.py"),
        "--mode",
        args.mode,
        "--image-runtime",
        args.image_runtime,
        "--repository",
        args.repository,
        "--python",
        args.python,
        "--mount-root",
        args.mount_target,
        "--bundle-root",
        args.bundle_root,
        "--run-id",
        args.run_id,
        "--output",
        f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}",
        "--expected-source-sha",
        args.expected_source_sha,
        "--image-digest",
        args.image,
        "--expected-runner-sha256",
        job.sha256(job.__file__),
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--shutdown-timeout",
        str(args.shutdown_timeout),
    ]
    if args.image_runtime == "python":
        release = _python_release_receipt(args)
        runner += [
            "--python-workflow",
            release["workflow"],
            "--python-plugin-digest",
            release["plugin_digest"],
            "--python-receipt-sha256",
            release["receipt_sha256"],
        ]
    if args.local_summary:
        runner += ["--local-summary", args.local_summary]
    for receipt in args.build_receipt:
        runner += ["--build-receipt", receipt]
    if args.include_python:
        runner += ["--include-python"]
    if args.application_fixtures:
        runner += ["--application-fixtures"]
    if args.release_acceptance:
        runner += ["--release-acceptance"]
    if args.service_bin_dir:
        runner += ["--service-bin-dir", args.service_bin_dir]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def validate_report(report, *, args):
    targets = report.get("targets")
    job.require(
        isinstance(targets, list) and targets, "HTTP QA has no target inventory"
    )
    native = [target for target in targets if target.get("runtime") == "cpp"]
    python = [target for target in targets if target.get("runtime") == "python"]
    job.require(
        len(native) + len(python) == len(targets), "Unexpected HTTP runtime target"
    )
    job.require(
        all(target.get("source_sha") == args.expected_source_sha for target in targets),
        "HTTP target source differs",
    )
    fixture_targets = [target for target in targets if "application_fixtures" in target]
    requested_fixtures = getattr(args, "application_fixtures", False)
    job.require(
        len(fixture_targets) == int(requested_fixtures)
        and (
            not requested_fixtures
            or args.mode == "image"
            and args.image_runtime == "cpp"
        ),
        "Application fixture coverage differs from requested native image scope",
    )
    job.require(
        all(
            bool(target.get("release_acceptance"))
            == getattr(args, "release_acceptance", False)
            for target in targets
        ),
        "Expanded release acceptance differs from requested scope",
    )
    environment = next(
        (
            case.get("details", {})
            for case in report.get("cases", [])
            if case.get("name") == "environment"
        ),
        {},
    )
    job.require(
        environment.get("runner_sha256") == job.sha256(job.__file__),
        "HTTP QA runner source differs from its controller",
    )
    job.require(
        len(python)
        == (
            1
            if args.mode == "image" and args.image_runtime == "python"
            else int(args.include_python)
        ),
        "Python HTTP coverage differs from requested scope",
    )
    if python and args.mode == "host":
        job.require(
            python[0].get("name") == "python.generated"
            and python[0].get("workflow") == "http-python",
            "Unexpected Python workflow target",
        )
    if args.mode == "image" and args.image_runtime == "python":
        release = _python_release_receipt(args)
        job.require(
            not native
            and len(python) == 1
            and python[0].get("name") == "python.release"
            and python[0].get("plugin") == "/opt/pnms/plugin"
            and python[0].get("workflow") == release["workflow"]
            and python[0].get("plugin_digest") == release["plugin_digest"]
            and python[0].get("build_receipt_sha256") == release["receipt_sha256"]
            and python[0].get("runtime_identity") == job._expected_identity(args.image),
            "Python HTTP QA differs from its exact release image or public build receipt",
        )
        job.require(
            environment.get("python_release") == python[0],
            "Python release environment evidence differs",
        )
    elif args.mode == "image":
        job.require(
            len(native) == 1 and native[0].get("bundle") == args.bundle_root,
            "Image HTTP QA must test its selected baked bundle",
        )
        job.require(
            native[0].get("runtime_identity") == job._expected_identity(args.image),
            "HTTP result is not bound to the submitted release image",
        )
    else:
        job.require(native, "Host HTTP QA has no native targets")
        if args.local_summary:
            evidence = report.get("input_local_summary", {})
            job.require(
                evidence.get("path") == args.local_summary
                and evidence.get("source_sha") == args.expected_source_sha
                and evidence.get("image_digest") == args.image
                and re.fullmatch(r"[a-f0-9]{64}", evidence.get("sha256", "")),
                "HTTP QA local handoff identity differs",
            )
            candidates = evidence.get("candidates", [])
            inventory = {local_job.candidate_key(candidate) for candidate in candidates}
            profile = evidence.get("profile", "local")
            required = job.local_target_inventory(profile)
            job.require(
                len(candidates) == len(required) and inventory == required,
                "HTTP handoff must include all five qualified native bundles, including file ingress"
                if profile == "local"
                else "HTTP handoff must include both qualified fresh venv native bundles",
            )
            if profile == "fresh-venv":
                selected_python = str(
                    PurePosixPath(args.local_summary).parent / "venv/bin/python"
                )
                job.require(
                    evidence.get("python") == selected_python
                    and environment.get("python") == selected_python
                    and environment.get("input_profile") == profile,
                    "HTTP QA did not use the fresh venv selected Python",
                )
            job.require(
                len(native) == len(candidates),
                "HTTP target inventory differs from local handoff",
            )
            for target in native:
                selected = [
                    candidate
                    for candidate in candidates
                    if local_job.candidate_name(candidate) == target.get("name")
                ]
                job.require(
                    len(selected) == 1
                    and all(
                        target.get(key) == selected[0].get(key)
                        for key in ("build_receipt", "bundle_digest", "build_id")
                    )
                    and target.get("ingress", "application/json")
                    == selected[0].get("ingress", "application/json"),
                    "HTTP target differs from the local QA candidate",
                )
        else:
            job.require(
                len(native) == len(args.build_receipt)
                and {target.get("build_receipt") for target in native}
                == set(args.build_receipt),
                "HTTP QA tested different build receipts",
            )
        for target in native:
            job.require(
                target.get("runtime_identity")
                == {"kind": "bundle", "value": target.get("bundle_digest")},
                "Host HTTP QA is not bound to its compiled bundle",
            )
    job.validate_summary(
        report,
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image,
        targets=targets,
    )


def run(args):
    validate_args(args)

    def validate(report, **identity):
        validate_report(report, args=args)

    return lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=job.extract_summary,
        summary_validator=validate,
        historical_first=True,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
