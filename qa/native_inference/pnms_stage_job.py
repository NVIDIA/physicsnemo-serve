"""Stage the prepared image's frozen source for ordinary Lepton QA jobs."""

import argparse
import json
from pathlib import Path
import re
import shutil

if __package__:
    from .contract import sha256, write_json
else:
    from contract import sha256, write_json

SUMMARY_BEGIN = "PNMS_STAGE_QA_SUMMARY_BEGIN"
SUMMARY_END = "PNMS_STAGE_QA_SUMMARY_END"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage(source, metadata, inventory, output, expected_source_sha):
    source, metadata, inventory, output = map(
        Path, (source, metadata, inventory, output)
    )
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(metadata.read_text())
    require(meta.get("source_sha") == expected_source_sha, "source identity differs")
    require(
        meta.get("source_inventory_sha256") == sha256(inventory),
        "inventory identity differs",
    )
    records = json.loads(inventory.read_text())
    require(isinstance(records, dict) and records, "source inventory is empty")
    for relative, record in records.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        original = source / path
        require(
            original.resolve().is_relative_to(source.resolve()), "source path escapes"
        )
        require(
            record.get("symlink") is None and not original.is_symlink(),
            "source path is a symlink",
        )
        require(
            original.is_file()
            and original.stat().st_size == record["size_bytes"]
            and sha256(original) == record["sha256"],
            "source file identity differs: " + relative,
        )
    require(
        "qa/native_inference/venv-requirements.txt" in records,
        "venv requirements are absent",
    )
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "source"
    for relative, record in records.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        target.chmod(record["mode"])
        require(
            sha256(target) == record["sha256"],
            "copied source identity differs: " + relative,
        )
    manifest = output / "source-manifest.json"
    write_json(
        manifest,
        dict(
            schema_version=1,
            source_sha=expected_source_sha,
            source_archive_sha256=meta["source_archive_sha256"],
            files=records,
        ),
    )
    requirements = destination / "qa/native_inference/venv-requirements.txt"
    return dict(
        source_root=str(destination),
        source_manifest=str(manifest),
        source_manifest_sha256=sha256(manifest),
        requirements_file=str(requirements),
        requirements_sha256=sha256(requirements),
        file_count=len(records),
    )


def extract_summary(output):
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    require(begin >= 0 and end >= 0, "No complete source-stage summary")
    return json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())


def validate_summary(report, *, run_id, source_sha, image_digest, profile):
    for key, value in dict(
        schema_version=1,
        run_id=run_id,
        source_sha=source_sha,
        image_digest=image_digest,
        profile=profile,
        status="passed",
    ).items():
        require(report.get(key) == value, "Stage summary identity differs: " + key)
    require(
        report.get("expected_cases") == ["source.stage"]
        and report.get("cases") == [{"name": "source.stage", "status": "passed"}],
        "Stage case inventory differs",
    )
    root = Path(report["source_manifest"]).parent
    require(
        Path(report["source_root"]) == root / "source"
        and Path(report["requirements_file"])
        == root / "source/qa/native_inference/venv-requirements.txt",
        "Stage output paths differ",
    )
    for field in ("source_manifest_sha256", "requirements_sha256", "runner_sha256"):
        require(
            re.fullmatch(r"[a-f0-9]{64}", report.get(field, "")),
            "Stage identity is missing: " + field,
        )
    require(
        isinstance(report.get("file_count"), int) and report["file_count"] > 0,
        "Stage file inventory is empty",
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", type=Path, default=Path("/workspace"))
    result.add_argument(
        "--metadata", type=Path, default=Path("/opt/physicsnemo-qa/source.json")
    )
    result.add_argument(
        "--inventory",
        type=Path,
        default=Path("/opt/physicsnemo-qa/source-inventory.json"),
    )
    for name in (
        "run-id",
        "expected-source-sha",
        "image-digest",
        "expected-runner-sha256",
    ):
        result.add_argument("--" + name, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = dict(
        schema_version=1,
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image_digest,
        profile="stage",
        status="failed",
        expected_cases=["source.stage"],
        cases=[],
        runner_sha256=sha256(__file__),
    )
    code = 1
    owned_output = False
    try:
        require(
            report["runner_sha256"] == args.expected_runner_sha256,
            "Stage runner identity differs",
        )
        report.update(
            stage(
                args.source,
                args.metadata,
                args.inventory,
                args.output,
                args.expected_source_sha,
            )
        )
        owned_output = True
        report.update(
            status="passed", cases=[{"name": "source.stage", "status": "passed"}]
        )
        validate_summary(
            report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            profile="stage",
        )
        code = 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    if owned_output:
        write_json(args.output / "summary.json", report)
    print(SUMMARY_BEGIN, flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    print(SUMMARY_END, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
