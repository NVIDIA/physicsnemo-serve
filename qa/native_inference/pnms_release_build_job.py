#!/usr/bin/env python3
"""Build one native pnms application and seal its portable release inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

if __package__:
    from . import pnms_local_job as local, pnms_stage_job as source_stage
    from .contract import sha256, write_json
else:
    import pnms_local_job as local
    import pnms_stage_job as source_stage
    from contract import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "native-inference/deploy"))
from native_release import (  # noqa: E402
    contained,
    file_identity,
    image_digest,
    read_json,
    require,
    verify_inventory,
)
from native_release_artifacts import seal  # noqa: E402

SUMMARY_BEGIN = "PNMS_RELEASE_BUILD_SUMMARY_BEGIN"
SUMMARY_END = "PNMS_RELEASE_BUILD_SUMMARY_END"


def expected_cases():
    return ["source.verify", "application.build", "handoff.seal"]


def bundle_verify(path, expected):
    sys.path.insert(0, str(ROOT / "application-cli/src"))
    from pnms_app.bundle import verify

    return verify(path, expected)


def project_document(project, build=None):
    import copy
    import yaml

    descriptor = Path(project) / "pnms.yaml"
    if descriptor.is_file():
        return yaml.safe_load(descriptor.read_text())
    if isinstance(build, dict) and isinstance(build.get("application"), dict):
        return copy.deepcopy(build["application"])
    sys.path.insert(0, str(ROOT / "application-cli/src"))
    from pnms_app.config import load_project

    return load_project(project, ROOT)["document"]


def check_build(
    project, receipt, *, source_sha, runtime_image, require_executable=True
):
    build = read_json(receipt)
    descriptor = project_document(project, build)
    require(
        descriptor.get("runtime") == "cpp"
        and build.get("runtime") == "cpp"
        and build.get("name") == descriptor.get("name"),
        "completed native application/project identity required",
    )
    require(
        descriptor.get("serve", {}).get("image") == runtime_image,
        "project serving image differs from selected runtime image",
    )
    require(
        build.get("schema_version") == 1
        and build.get("status") == "complete"
        and build.get("source_sha") == source_sha
        and re.fullmatch(r"[a-f0-9]{32}", build.get("build_id", ""))
        and build.get("checks", {}).get("model") == "passed"
        and build.get("checks", {}).get("application") == "passed",
        "native build is incomplete or belongs to another source",
    )
    bundle = contained(Path(receipt).parent, build["bundle"], file=False)
    require(bundle.is_dir(), "completed bundle directory required")
    bundle_verify(bundle, build["bundle_digest"])
    runner = contained(bundle, read_json(bundle / "bundle.json")["workflow"]["runner"])
    require(
        not require_executable or runner.stat().st_mode & 0o111,
        "compiled runner executable mode is missing",
    )
    return build, bundle


def seal_build(project, receipt, output, *, source_inventory, **provenance):
    """Keep the immutable receipt and bundle, excluding producer build trees."""
    project, receipt, output = map(Path, (project, receipt, output))
    source_inventory = Path(source_inventory)
    source_inventory_identity = file_identity(source_inventory)
    require(
        source_inventory_identity["sha256"] == provenance["source_inventory_sha256"],
        "source inventory differs from the verified frozen stage",
    )
    image_digest(provenance["image_digest"])
    image_digest(provenance["runtime_image"])
    build, bundle = check_build(
        project,
        receipt,
        source_sha=provenance["source_sha"],
        runtime_image=provenance["runtime_image"],
    )
    transport = output / "transport"
    portable = transport / "project"
    destination = portable / ".pnms/builds" / build["build_id"]
    destination.mkdir(parents=True, exist_ok=False)
    if (project / "pnms.yaml").is_file():
        shutil.copy2(project / "pnms.yaml", portable / "pnms.yaml")
    else:
        import yaml

        document = project_document(project, build)
        (portable / "pnms.yaml").write_text(
            yaml.safe_dump(
                {
                    name: value
                    for name, value in document.items()
                    if not name.startswith("_")
                },
                sort_keys=False,
            )
        )
    shutil.copy2(receipt, destination / "build.json")
    shutil.copyfile(source_inventory, transport / "source-inventory.json")
    require(
        file_identity(transport / "source-inventory.json") == source_inventory_identity,
        "source inventory changed during transport capture",
    )
    # The existing transport rejects links. Preserve bundle semantics and fail
    # rather than dereferencing a link while retaining its original seal.
    shutil.copytree(bundle, destination / build["bundle"], symlinks=True)
    executables = [
        path.relative_to(transport).as_posix()
        for path in sorted((destination / build["bundle"]).rglob("*"))
        if path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o111
    ]
    handoff = {
        "schema_version": 1,
        **provenance,
        "runtime": "cpp",
        "name": build["name"],
        "build_id": build["build_id"],
        "bundle_digest": build["bundle_digest"],
        "project": "project",
        "build_receipt": (destination / "build.json").relative_to(transport).as_posix(),
        "build_receipt_identity": file_identity(receipt),
        "source_inventory": "source-inventory.json",
        "source_inventory_identity": source_inventory_identity,
        "executables": executables,
    }
    write_json(transport / "handoff.json", handoff)
    manifest = seal(
        transport,
        output / "bundle.tar",
        **{
            key: value
            for key, value in handoff.items()
            if key not in ("project", "build_receipt", "executables")
        },
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def restore_handoff(extracted, manifest):
    """Restore safe executable modes lost by the existing byte-only extractor."""
    extracted = Path(extracted).resolve()
    verify_inventory(extracted, manifest["files"], exact=True)
    handoff = read_json(extracted / "handoff.json")
    for key in (
        "source_sha",
        "source_inventory_sha256",
        "image_digest",
        "runtime_image",
        "build_id",
        "bundle_digest",
        "build_receipt_identity",
        "source_inventory_identity",
    ):
        require(
            handoff.get(key) == manifest.get(key), "handoff provenance differs: " + key
        )
    require(handoff.get("project") == "project", "portable project path differs")
    require(
        handoff.get("source_inventory") == "source-inventory.json",
        "portable source inventory path differs",
    )
    source_inventory = contained(extracted, handoff["source_inventory"])
    require(
        file_identity(source_inventory) == handoff["source_inventory_identity"]
        and handoff["source_inventory_identity"]["sha256"]
        == handoff["source_inventory_sha256"],
        "portable source inventory identity differs",
    )
    expected_receipt = f"project/.pnms/builds/{handoff['build_id']}/build.json"
    require(
        handoff.get("build_receipt") == expected_receipt,
        "portable receipt path differs",
    )
    receipt = contained(extracted, expected_receipt)
    require(
        file_identity(receipt) == handoff["build_receipt_identity"],
        "build receipt bytes differ",
    )
    build, bundle = check_build(
        extracted / "project",
        receipt,
        source_sha=handoff["source_sha"],
        runtime_image=handoff["runtime_image"],
        require_executable=False,
    )
    executables = handoff.get("executables")
    require(
        isinstance(executables, list) and len(set(executables)) == len(executables),
        "invalid executable inventory",
    )
    for name in executables:
        path = contained(extracted, name)
        require(path.is_relative_to(bundle), "executable path escapes compiled bundle")
        path.chmod(0o755)
    runner = contained(bundle, read_json(bundle / "bundle.json")["workflow"]["runner"])
    require(
        runner.stat().st_mode & 0o111, "compiled runner executable identity is missing"
    )
    bundle_verify(bundle, build["bundle_digest"])
    return {
        **{
            key: value
            for key, value in handoff.items()
            if key
            not in ("executables", "project", "build_receipt", "source_inventory")
        },
        "status": "passed",
        "project": str(extracted / "project"),
        "build_receipt": str(receipt),
        "source_inventory": str(source_inventory),
        "archive": manifest["archive"],
    }


def build_application(ctx):
    import yaml

    if ctx.args.project:
        relative = Path(ctx.args.project)
        require(
            not relative.is_absolute()
            and relative.parts
            and ".." not in relative.parts,
            "project must be relative to the frozen source snapshot",
        )
        project = (ctx.repository / relative).resolve()
        require(
            project.is_relative_to(ctx.repository.resolve()),
            "project escapes frozen source",
        )
        require(
            (project / "pnms.yaml").is_file() or (project / "plugin.yaml").is_file(),
            "project requires pnms.yaml or an existing native plugin",
        )
    else:
        project = ctx.root / "application"
        template = ctx.args.template or (
            "custom" if ctx.args.backend == "tensorrt" else "tensor"
        )
        ctx.pnms(
            "init",
            "init",
            project,
            "--runtime",
            "cpp",
            "--backend",
            ctx.args.backend,
            "--template",
            template,
            "--repository",
            ctx.repository,
        )
    descriptor = project / "pnms.yaml"
    document = project_document(project)
    require(
        isinstance(document, dict) and document.get("runtime") == "cpp",
        "release build project must use the native cpp runtime",
    )
    document.setdefault("serve", {})["image"] = ctx.args.runtime_image
    # This owned copy uses the prepared image stack; an authored producer venv
    # would otherwise override even an explicit --python selection in Builder.
    document.setdefault("build", {}).pop("venv", None)
    if descriptor.is_file():
        descriptor.write_text(yaml.safe_dump(document, sort_keys=False))
    else:
        # This is the source-stage-owned copy, never the user's original tree.
        profile = project / document["_legacy_native"]["profile"]
        release = yaml.safe_load(profile.read_text())
        release.setdefault("images", {})["server"] = ctx.args.runtime_image
        profile.write_text(yaml.safe_dump(release, sort_keys=False))
    result = ctx.pnms(
        "build",
        "build",
        project,
        "--environment",
        "local",
        "--python",
        ctx.args.python,
        "--sdk-prefix",
        ctx.args.sdk_prefix,
        "--runtime",
        Path(ctx.args.sdk_prefix) / "bin/physicsnemo-infer",
        "--repository",
        ctx.repository,
    )
    require(
        result.get("status") == "complete", "public application build did not complete"
    )
    return project, Path(result["root"]) / "build.json"


def extract_summary(output):
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    require(begin >= 0 and end >= 0, "No complete pnms release-build summary")
    report = json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())
    require(isinstance(report, dict), "release-build summary must be an object")
    return report


def validate_summary(report, *, run_id, source_sha, image_digest, profile):
    for key, value in dict(
        schema_version=1,
        status="passed",
        run_id=run_id,
        source_sha=source_sha,
        image_digest=image_digest,
        profile=profile,
    ).items():
        require(
            type(report.get(key)) is type(value) and report[key] == value,
            "release-build summary differs: " + key,
        )
    require(
        report.get("expected_cases") == expected_cases(),
        "release-build inventory differs",
    )
    require(
        report.get("cases")
        == [{"name": name, "status": "passed"} for name in expected_cases()],
        "every release-build gate must pass exactly once",
    )
    for key, length in (
        ("source_inventory_sha256", 64),
        ("build_id", 32),
        ("bundle_digest", 64),
        ("http_runner_sha256", 64),
        ("contract_sha256", 64),
        ("runner_sha256", 64),
    ):
        require(
            re.fullmatch(f"[a-f0-9]{{{length}}}", str(report.get(key, ""))),
            "release-build identity missing: " + key,
        )
    for key in ("archive", "build_receipt_identity", "source_inventory_identity"):
        identity = report.get(key, {})
        require(
            re.fullmatch(r"[a-f0-9]{64}", str(identity.get("sha256", "")))
            and type(identity.get("size_bytes")) is int
            and identity["size_bytes"] > 0,
            "release-build file identity missing: " + key,
        )
    require(
        report["source_inventory_identity"]["sha256"]
        == report["source_inventory_sha256"],
        "release-build source inventory identity differs from verified stage",
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", type=Path, default=Path("/workspace"))
    result.add_argument("--python", default=sys.executable)
    result.add_argument(
        "--sdk-prefix", type=Path, default=Path("/opt/physicsnemo-inference")
    )
    result.add_argument(
        "--source-metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument(
        "--source-inventory",
        type=Path,
        default=Path("/opt/physicsnemo-qa/source-inventory.json"),
    )
    result.add_argument("--project")
    result.add_argument("--backend", choices=("aoti", "tensorrt"), default="aoti")
    result.add_argument("--template", choices=("tensor", "custom"))
    result.add_argument("--runtime-image", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--command-timeout", type=float, default=2400)
    for name in (
        "run-id",
        "expected-source-sha",
        "image-digest",
        "expected-runner-sha256",
    ):
        result.add_argument("--" + name, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    # The controller launches this driver with the selected interpreter. Pass
    # its actual venv path onward; Builder treats a bare "python3" as a file.
    args.python = sys.executable
    report = dict(
        schema_version=1,
        status="failed",
        profile="release-build",
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image_digest,
        runtime_image=args.runtime_image,
        expected_cases=expected_cases(),
        cases=[],
        runner_sha256=sha256(__file__),
    )
    owned = False
    code = 1
    try:
        image_digest(args.image_digest)
        image_digest(args.runtime_image)
        require(
            report["runner_sha256"] == args.expected_runner_sha256,
            "release-build runner differs",
        )
        ctx = local.Context(args)
        owned = True
        source_inventory = ctx.root / "verified-source-inventory.json"
        shutil.copyfile(args.source_inventory, source_inventory)
        staged = source_stage.stage(
            args.repository,
            args.source_metadata,
            source_inventory,
            ctx.root / "frozen-source",
            args.expected_source_sha,
        )
        ctx.repository = Path(staged["source_root"])
        ctx.environment["PNMS_SOURCE_SHA"] = args.expected_source_sha
        report["source_inventory_sha256"] = sha256(source_inventory)
        report["cases"].append({"name": "source.verify", "status": "passed"})
        project, receipt = build_application(ctx)
        report["cases"].append({"name": "application.build", "status": "passed"})
        runner_root = ctx.root / "http-runner"
        runner_root.mkdir()
        for name in ("pnms_http_job.py", "contract.py"):
            shutil.copy2(
                ctx.repository / "qa/native_inference" / name, runner_root / name
            )
        report.update(
            http_runner=str(runner_root / "pnms_http_job.py"),
            http_runner_sha256=sha256(runner_root / "pnms_http_job.py"),
            contract_sha256=sha256(runner_root / "contract.py"),
        )
        manifest = seal_build(
            project,
            receipt,
            ctx.root,
            source_inventory=source_inventory,
            **{
                key: report[key]
                for key in (
                    "run_id",
                    "source_sha",
                    "source_inventory_sha256",
                    "image_digest",
                    "runtime_image",
                )
            },
        )
        report.update(
            {
                key: manifest[key]
                for key in (
                    "build_id",
                    "bundle_digest",
                    "archive",
                    "build_receipt_identity",
                    "source_inventory_identity",
                )
            }
        )
        report["cases"].append({"name": "handoff.seal", "status": "passed"})
        report["status"] = "passed"
        validate_summary(
            report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            profile="release-build",
        )
        code = 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    if owned:
        write_json(args.output / "release-build-summary.json", report)
    print(SUMMARY_BEGIN, flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print(SUMMARY_END, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
