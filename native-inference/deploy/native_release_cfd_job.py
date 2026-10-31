"""Compile and qualify the fixed CFD preset, then seal complete Builder evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from native_release import (
    compilation_key,
    cfd_reference_contract,
    file_identity,
    read_json,
    require,
    verify_completed_build,
    write_json,
)
from native_release_artifacts import seal
from native_release_job import environment, qualify_stack
from qa.scripts import run_native_cfd_job as full

BEGIN = "NATIVE_RELEASE_BUILD_SUMMARY_BEGIN"
END = "NATIVE_RELEASE_BUILD_SUMMARY_END"
CASES = ["preflight", *["full." + name for name in full.CASES], "seal"]


def collect_builds(full_root, handoff, resolved, *, image, keys, stack_fingerprint):
    """Retain Builder proofs and a joint full-field receipt for exact package bytes."""
    full_root, handoff = Path(full_root), Path(handoff)
    receipt = read_json(full_root / "build-receipt.json")
    require(receipt.get("status") == "passed", "full CFD qualification did not pass")
    reference = full_root / "reference/fields.json"
    receipt.update(
        builder_stack_image=resolved["images"]["builder"],
        reference_contract=cfd_reference_contract(resolved),
        qualification_source_sha=resolved["source_sha"],
        compilation_keys=keys,
        native_stack_fingerprint=stack_fingerprint,
        reference_storage=resolved.get("_reference_storage", {}),
        reference={"path": str(reference), **file_identity(reference)},
        input_root=str(full_root / "inputs/drivaerml"),
    )
    proof = handoff / "joint-proof"
    proof.mkdir(parents=True)
    write_json(proof / "build-receipt.json", receipt)
    for src, dest in (
        ("summary.json", "summary.json"),
        ("native/global_stats.json", "global_stats.json"),
        ("environment.json", "environment.json"),
    ):
        shutil.copyfile(full_root / src, proof / dest)
    shutil.copytree(full_root / "startup", proof / "startup")
    for alias, model in resolved["model_aliases"].items():
        points = model["project_document"]["points"]
        candidate = handoff / "aliases" / alias
        shutil.copytree(full_root / "build" / f"transolver{points}", candidate)
        previous = (
            read_json(candidate / "release-provenance.json")
            if (candidate / "release-provenance.json").is_file()
            else {}
        )
        write_json(
            candidate / "release-provenance.json",
            {
                "schema_version": 1,
                "source_sha": previous.get("source_sha", resolved["source_sha"]),
                "qualification_source_sha": resolved["source_sha"],
                "job_image": image,
                "builder_stack_image": resolved["images"]["builder"],
                "compilation_key": keys[alias],
                "native_stack_fingerprint": stack_fingerprint,
            },
        )
        if (candidate / "cfd-release").exists():
            shutil.rmtree(candidate / "cfd-release")
        shutil.copytree(proof, candidate / "cfd-release")
        verify_completed_build(
            candidate,
            backend=resolved["backend"],
            target=resolved["target"]["compilation"],
        )
    sources = {
        alias: read_json(handoff / "aliases" / alias / "release-provenance.json")[
            "source_sha"
        ]
        for alias in resolved["model_aliases"]
    }
    receipt["model_build_sources"] = sources
    receipt["source_sha"] = (
        next(iter(set(sources.values())))
        if len(set(sources.values())) == 1
        else resolved["source_sha"]
    )
    for alias in resolved["model_aliases"]:
        write_json(
            handoff / "aliases" / alias / "cfd-release/build-receipt.json", receipt
        )
    shutil.rmtree(proof)


def write_builder_launcher(path):
    path = Path(path)
    path.write_text(
        "#!/usr/bin/env python3\nfrom model_builder.build.cli import main\nraise SystemExit(main())\n"
    )
    path.chmod(0o755)
    return path


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    request = read_json(args.input_root / "request.json")
    resolved = request["resolved"]
    resolved["paths"]["repository_root"] = "/opt/native-release/source"
    summary = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "source_sha": resolved["source_sha"],
        "image_digest": args.image,
        "expected_cases": CASES,
        "cases": [],
    }
    started = time.monotonic()
    try:
        require(
            os.environ.get("SOURCE_SHA") == resolved["source_sha"],
            "compiler source differs",
        )
        actual = environment(resolved["target"], backend=resolved["backend"])
        qualify_stack(request, backend=resolved["backend"])
        keys = {
            alias: compilation_key(resolved, alias)
            for alias in resolved["model_aliases"]
        }
        require(
            keys
            == {
                alias: item["compilation_key"]
                for alias, item in request["build_aliases"].items()
            },
            "CFD compile identity changed after image assembly",
        )
        summary["cases"].append(
            {
                "name": "preflight",
                "status": "passed",
                "seconds": time.monotonic() - started,
            }
        )
        write_json(args.output / "source.json", {"source_sha": resolved["source_sha"]})
        source = Path("/opt/native-release/source")
        builder = write_builder_launcher(args.output / "model-builder")
        reuse = []
        for alias, record in request["build_aliases"].items():
            if record["action"] == "reuse":
                points = resolved["model_aliases"][alias]["project_document"]["points"]
                reuse += [
                    "--reuse-build",
                    str(points) + "=" + str(args.input_root / "reuse" / alias),
                ]
        cfg = full.parser().parse_args(
            [
                "--output",
                str(args.output / "full"),
                "--run-id",
                args.run_id,
                "--image",
                args.image,
                "--source-sha",
                resolved["source_sha"],
                "--profile",
                str(source / "qa/inference/cfd_parity_surface_run1_transolver.json"),
                "--request",
                str(
                    source
                    / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
                ),
                "--backend",
                resolved["backend"],
                "--native-root",
                str(source / "native-inference"),
                "--builder",
                str(builder),
                "--runtime",
                "/opt/physicsnemo-inference/bin/physicsnemo-infer",
                "--source-metadata",
                str(args.output / "source.json"),
                "--asset-cache",
                str(Path(resolved["lepton"]["mount_target"]) / "assets/transolver"),
                "--command-timeout",
                "7000",
                *reuse,
            ]
        )
        require(full.run(cfg) == 0, "full CFD qualification failed")
        scientific = read_json(args.output / "full/summary.json")
        summary["cases"].extend(
            {**case, "name": "full." + case["name"]} for case in scientific["cases"]
        )
        started = time.monotonic()
        handoff = args.output / "handoff"
        collect_builds(
            args.output / "full",
            handoff,
            resolved,
            image=args.image,
            keys=keys,
            stack_fingerprint=request["stack_fingerprint"],
        )
        write_json(
            handoff / "build-provenance.json",
            {
                "environment": actual,
                "source_sha": resolved["source_sha"],
                "compilation_keys": keys,
                "builder_stack_image": resolved["images"]["builder"],
                "native_stack_fingerprint": request["stack_fingerprint"],
            },
        )
        manifest = seal(
            handoff,
            args.output / "bundle.tar",
            run_id=args.run_id,
            source_sha=resolved["source_sha"],
            image_digest=args.image,
            builder_image=resolved["images"]["builder"],
            compilation_keys=keys,
        )
        write_json(args.output / "manifest.json", manifest)
        summary["cases"].append(
            {"name": "seal", "status": "passed", "seconds": time.monotonic() - started}
        )
        summary["status"] = "passed"
    except BaseException as error:
        summary.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(args.output / "summary.json", summary)
        print(BEGIN, flush=True)
        print(json.dumps(summary, sort_keys=True), flush=True)
        print(END, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--input-root", type=Path, default=Path("/opt/native-release/input")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
