#!/usr/bin/env python3
"""Qualify frozen common-CLI CFD on one finite Lepton H100 allocation."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa.scripts import run_lepton_native_inference_qa as lifecycle  # noqa: E402
from qa.native_inference import pnms_cfd_job as job  # noqa: E402

FROZEN_RUNNER = "c8fc5865c8908b1d5a0e11c4aea6c4881411feacb9de1eb863c305d86ef40b18"
COMPARISON_BEGIN = "PNMS_CFD_COMPARISON_BEGIN"
COMPARISON_END = "PNMS_CFD_COMPARISON_END"
BOUNDARY = "client-http-to-full-canonical-report-v1"
MAX_COMPARISON_BYTES = 256 * 1024

# This host-supplied stdlib wrapper does not modify or replace the frozen job.
# Keep exact file bytes in transport; compact session proofs bind comparison
# receipt identities to the registry image/source selected by this controller.
WRAPPER = r"""
import base64, hashlib, json, pathlib, runpy, sys
def identity(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
path, expected = pathlib.Path(sys.argv[1]), sys.argv[2]
assert identity(path.read_bytes())["sha256"] == expected, "frozen CFD job differs"
module = runpy.run_path(str(path), run_name="pnms_cfd_frozen_adapter")
args = module["parser"]().parse_args(sys.argv[3:])
args.python = pathlib.Path(sys.executable)
summary = module["run"](args)
assert summary["status"] == "passed", "CFD qualification did not pass"
raw = pathlib.Path(summary["comparison"]["path"]).read_bytes()
assert len(raw) <= 262144, "comparison exceeds compact transport limit"
assert identity(raw) == {key: summary["comparison"][key] for key in ("sha256", "size_bytes")}, "comparison bytes changed"
proofs = []
for selected in summary["sessions"]:
    data = pathlib.Path(selected).read_bytes()
    session = json.loads(data)
    proof = {key: session.get(key) for key in ("run_id", "image_digest", "source_sha", "runtime", "ordinal", "status", "boundary", "model_build_sources", "bundle_digest", "cold_startup_seconds", "numerical_libraries")}
    proof.update(path=selected, **identity(data))
    for key in ("hardware", "workload", "environment", "evaluator_environment"):
        proof[key + "_sha256"] = identity(canonical(session[key]))["sha256"]
    proof["requests"] = [{key: gate.get(key) for key in ("label", "seconds", "canonical_report_sha256", "parity", "artifacts_verified")} for gate in [session["prerequisite"], *session["samples"], session["postrun"]]]
    proofs.append(proof)
print("PNMS_CFD_COMPARISON_BEGIN", flush=True)
print(json.dumps(dict(schema_version=1, comparison_base64=base64.b64encode(raw).decode(), sessions=proofs), sort_keys=True, allow_nan=False), flush=True)
print("PNMS_CFD_COMPARISON_END", flush=True)
"""


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate evidence field: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite evidence value: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def block(output, begin, end, *, start=0):
    offset = output.find(begin, start)
    finish = output.find(end, offset + len(begin))
    require(offset >= start and finish >= 0, "logs lack complete " + begin)
    require(
        output.find(begin, offset + len(begin)) < 0
        and output.find(end, finish + len(end)) < 0,
        "ambiguous evidence blocks: " + begin,
    )
    payload = output[offset + len(begin) : finish].strip()
    try:
        record = strict_json(payload)
    except json.JSONDecodeError:
        # The job emits each JSON record on one line. Lepton can interleave
        # stderr traceback lines with the surrounding stdout marker block.
        records = []
        for line in payload.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = strict_json(line)
            except json.JSONDecodeError:
                require(
                    not line.startswith(
                        ("{", "[", '"', "true", "false", "null", "NaN", "Infinity")
                    )
                    and not line[0].isdigit()
                    and not line.startswith("-"),
                    "ambiguous evidence JSON record: " + begin,
                )
                continue
            records.append(value)
        require(len(records) == 1, "ambiguous evidence JSON records: " + begin)
        record = records[0]
    return record, finish + len(end)


def extract_evidence(output):
    report, finish = block(output, job.BEGIN, job.END)
    require(isinstance(report, dict), "CFD job summary must be an object")
    if report.get("status") != "passed":
        return report, None
    transport, _ = block(output, COMPARISON_BEGIN, COMPARISON_END, start=finish)
    require(
        isinstance(transport, dict)
        and set(transport) == {"schema_version", "comparison_base64", "sessions"}
        and transport["schema_version"] == 1,
        "unsupported comparison transport",
    )
    encoded = transport["comparison_base64"]
    require(
        isinstance(encoded, str)
        and len(encoded) <= 4 * ((MAX_COMPARISON_BYTES + 2) // 3),
        "comparison transport exceeds its byte limit",
    )
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid comparison base64") from error
    require(len(raw) <= MAX_COMPARISON_BYTES, "comparison exceeds its byte limit")
    return report, {"raw": raw, "sessions": transport["sessions"]}


def source_files():
    selected = Path(job.__file__).resolve()
    require(
        identity(selected.read_bytes())["sha256"] == FROZEN_RUNNER,
        "local CFD job differs from the reviewed controller/job version",
    )
    return selected


def build_parser():
    parser = lifecycle.build_parser(profile_choices=("cfd",))
    parser.set_defaults(job_timeout=7200)
    parser.add_argument("--repository", default="/workspace")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--sdk-prefix", default="/opt/physicsnemo-inference")
    parser.add_argument("--source-metadata", default="/opt/physicsnemo-qa/source.json")
    parser.add_argument(
        "--source-inventory", default="/opt/physicsnemo-qa/source-inventory.json"
    )
    parser.add_argument("--expected-source-inventory-sha256", required=True)
    parser.add_argument("--reference-fields")
    parser.add_argument("--input-root")
    return parser


def remote_root(args):
    return f"{args.mount_target.rstrip('/')}/native-inference/{args.run_id}"


def build_job_command(args, name):
    command = lifecycle.build_job_command(args, name)
    mount = args.mount_target.rstrip("/")
    reference = (
        args.reference_fields
        or mount + "/native-inference/cfd-full-20261021t164815-r3/reference/fields.json"
    )
    inputs = (
        args.input_root
        or mount + "/native-inference/cfd-full-20261021t164815-r3/inputs/drivaerml"
    )
    runner = [
        "exec",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{int(args.job_timeout) - 60}s",
        "/opt/nvidia/nvidia_entrypoint.sh",
        args.python,
        "-B",
        "-c",
        WRAPPER,
        str(PurePosixPath(args.repository) / "qa/native_inference/pnms_cfd_job.py"),
        FROZEN_RUNNER,
        "--repository",
        args.repository,
        "--python",
        args.python,
        "--sdk-prefix",
        args.sdk_prefix,
        "--source-metadata",
        args.source_metadata,
        "--source-inventory",
        args.source_inventory,
        "--expected-source-inventory-sha256",
        args.expected_source_inventory_sha256,
        "--expected-runner-sha256",
        FROZEN_RUNNER,
        "--image",
        args.image,
        "--source-sha",
        args.expected_source_sha,
        "--run-id",
        args.run_id,
        "--output",
        remote_root(args),
        "--reference-fields",
        reference,
        "--input-root",
        inputs,
        "--checkpoint",
        mount + "/assets/transolver/15e1c59a166037e2/Transolver.0.501.mdlus",
        "--stats",
        mount + "/assets/transolver/15e1c59a166037e2/global_stats.json",
        "--timeout",
        str(int(args.job_timeout) - 120),
    ]
    command[command.index("--command") + 1] = shlex.join(runner)
    return command


def file_record(record):
    require(
        isinstance(record, dict)
        and re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", ""))
        and type(record.get("size_bytes")) is int
        and record["size_bytes"] > 0,
        "invalid evidence file identity",
    )


def validate_summary(
    report,
    *,
    run_id,
    source_sha,
    image_digest,
    profile,
    source_inventory_sha256,
    mount_target="/outputs",
):
    require(profile == "cfd", "unexpected CFD controller profile")
    for key, value in {
        "schema_version": 1,
        "qualification": "pnms-cfd-common-cli-http-session6-v1",
        "status": "passed",
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
    }.items():
        require(report.get(key) == value, "CFD job identity/status differs: " + key)
    require(
        report.get("recompiled") is False, "CFD import must not claim model compilation"
    )
    require(
        report.get("expected_cases") == job.CASES
        and isinstance(report.get("cases"), list)
        and [row.get("name") for row in report["cases"]] == job.CASES
        and all(
            row.get("status") == "passed"
            and type(row.get("seconds")) in (int, float)
            and math.isfinite(row["seconds"])
            and row["seconds"] >= 0
            for row in report["cases"]
        ),
        "CFD qualification requires all 16 ordered passed gates",
    )
    root = f"{mount_target.rstrip('/')}/native-inference/{run_id}"
    file_record(report.get("driver"))
    require(
        report["driver"] == identity(source_files().read_bytes()),
        "CFD job runner identity differs",
    )
    file_record(report.get("source_inventory"))
    require(
        report["source_inventory"]["sha256"] == source_inventory_sha256
        and report["source_inventory"].get("path")
        == root + "/verified-source-inventory.json",
        "CFD source inventory differs",
    )
    sources = {"full": job.IMPORT_ORIGIN, "tail": job.IMPORT_ORIGIN}
    require(
        isinstance(report.get("imports"), dict)
        and set(report["imports"]) == set(job.IMPORT_IMAGES)
        and isinstance(report.get("builds"), dict)
        and set(report["builds"]) == set(job.IMPORT_IMAGES),
        "CFD requires both native import/build receipts",
    )
    for backend, image in job.IMPORT_IMAGES.items():
        imported, built = report["imports"][backend], report["builds"][backend]
        require(
            imported.get("schema_version") == 1
            and imported.get("backend") == backend
            and imported.get("source_image") == image
            and imported.get("model_build_sources") == sources
            and imported.get("recompiled") is False,
            "CFD imported model provenance differs: " + backend,
        )
        for key, path in (
            (
                "original_deployment",
                f"/opt/pnms-qa/cfd-imports/{backend}/deployment.json",
            ),
            ("relocated_deployment", f"{root}/imports/{backend}/deployment.json"),
        ):
            file_record(imported.get(key))
            require(
                imported[key].get("path") == path,
                "CFD imported deployment path differs",
            )
        require(
            built.get("source_sha") == source_sha
            and built.get("model_build_sources") == sources
            and built.get("recompiled") is False
            and re.fullmatch(r"[a-f0-9]{64}", built.get("bundle_digest", ""))
            and re.fullmatch(
                re.escape(root + f"/applications/{backend}/.pnms/builds/")
                + r"[a-f0-9]{32}",
                built.get("root", ""),
            ),
            "CFD completed build provenance differs: " + backend,
        )
    sessions = [
        f"{root}/sessions/{ordinal}-{runtime}/session.json"
        for ordinal, runtime in enumerate(job.ORDER)
    ]
    require(report.get("sessions") == sessions, "CFD session paths/order differ")
    file_record(report.get("comparison"))
    require(
        report["comparison"].get("path") == root + "/comparison.json",
        "CFD comparison path differs",
    )


def distribution(values):
    require(
        values
        and all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in values
        ),
        "CFD timings must be finite positive observations",
    )
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def validate_comparison(raw, sessions, report):
    require(
        identity(raw)
        == {key: report["comparison"][key] for key in ("sha256", "size_bytes")},
        "returned comparison bytes differ from qualified file",
    )
    comparison = strict_json(raw)
    require(
        isinstance(comparison, dict)
        and comparison.get("schema_version") == 1
        and comparison.get("status") == "passed"
        and comparison.get("boundary") == BOUNDARY
        and comparison.get("execution_order") == list(job.ORDER)
        and comparison.get("kernel_throughput_measured") is False
        and comparison.get("tail_latency_estimated") is False,
        "CFD comparison status/boundary/order differs",
    )
    require(
        isinstance(sessions, list) and len(sessions) == 6,
        "six compact session proofs required",
    )
    receipts = comparison.get("session_receipts")
    require(
        isinstance(receipts, list) and len(receipts) == 6,
        "comparison lacks six session file identities",
    )
    libraries = None
    for ordinal, (proof, receipt, runtime, path) in enumerate(
        zip(sessions, receipts, job.ORDER, report["sessions"], strict=True)
    ):
        file_record(proof)
        file_record(receipt)
        require(
            {key: proof.get(key) for key in ("path", "sha256", "size_bytes")} == receipt
            and proof["path"] == path,
            "comparison session file identity differs",
        )
        for key, value in {
            "run_id": report["run_id"],
            "image_digest": report["image_digest"],
            "source_sha": report["source_sha"],
            "runtime": runtime,
            "ordinal": ordinal,
            "status": "passed",
            "boundary": BOUNDARY,
        }.items():
            require(proof.get(key) == value, "CFD session identity differs: " + key)
        for section in ("hardware", "workload", "environment"):
            require(
                isinstance(comparison.get(section), dict)
                and comparison[section]
                and proof.get(section + "_sha256")
                == identity(canonical(comparison[section]))["sha256"],
                "CFD comparison session data differs: " + section,
            )
        require(
            proof.get("evaluator_environment_sha256") == proof["environment_sha256"],
            "CFD evaluator environment differs",
        )
        current = proof.get("numerical_libraries")
        require(
            isinstance(current, dict)
            and {"libtorch_cpu.so", "libtorch_cuda.so"} <= current.keys(),
            "CFD numerical library proof is incomplete",
        )
        for value in current.values():
            file_record(value)
        require(
            libraries is None or libraries == current,
            "CFD runtime numerical libraries differ",
        )
        libraries = current
        expected_sources = (
            None
            if runtime == "python"
            else {"full": job.IMPORT_ORIGIN, "tail": job.IMPORT_ORIGIN}
        )
        require(
            proof.get("model_build_sources") == expected_sources
            and proof.get("bundle_digest")
            == (
                None
                if runtime == "python"
                else report["builds"][runtime]["bundle_digest"]
            ),
            "CFD session model/bundle provenance differs",
        )
        requests = proof.get("requests")
        require(
            isinstance(requests, list)
            and [row.get("label") for row in requests]
            == ["prerequisite", "first", "warm-1", "warm-2", "warm-3", "postrun"]
            and all(
                row.get("parity") == "passed" and row.get("artifacts_verified") is True
                for row in requests
            ),
            "CFD request qualification/timing gates differ",
        )
        distribution([row.get("seconds") for row in requests])
        hashes = set()
        for row in requests:
            require(
                re.fullmatch(r"[a-f0-9]{64}", row.get("canonical_report_sha256", "")),
                "CFD canonical report proof is invalid",
            )
            hashes.add(row["canonical_report_sha256"])
        require(len(hashes) == 1, "CFD canonical report changed within a session")
    hardware, workload = comparison["hardware"], comparison["workload"]
    require(
        hardware.get("uuid", "").startswith("GPU-")
        and "H100" in hardware.get("name", "")
        and hardware.get("compute_capability") == [9, 0],
        "CFD comparison lacks its observed H100 allocation",
    )
    for key, value in {
        "case": "run_1",
        "point_count": 8828095,
        "seed": 42,
        "block_size": 60000,
        "precision": "float32",
        "profile_sha256": identity(
            (ROOT / "qa/inference/cfd_parity_surface_run1_transolver.json").read_bytes()
        )["sha256"],
        "request_sha256": identity(
            (
                ROOT
                / "plugins/physicsnemo-cfd-surface-benchmark/examples/public_run_1_transolver_request.json"
            ).read_bytes()
        )["sha256"],
    }.items():
        require(workload.get(key) == value, "CFD comparison workload differs: " + key)
    statistics_by_runtime = {}
    for runtime in ("python", "aoti", "tensorrt"):
        rows = [proof for proof in sessions if proof["runtime"] == runtime]
        statistics_by_runtime[runtime] = {
            "cold_startup_seconds": distribution(
                [proof["cold_startup_seconds"] for proof in rows]
            ),
            "first_seconds": distribution(
                [proof["requests"][1]["seconds"] for proof in rows]
            ),
            "warm_seconds": distribution(
                [
                    request["seconds"]
                    for proof in rows
                    for request in proof["requests"][2:5]
                ]
            ),
        }
    require(
        comparison.get("runtime_statistics") == statistics_by_runtime,
        "CFD comparison statistics differ from actual session timings",
    )
    require(
        comparison.get("warm_speedup_vs_python")
        == {
            runtime: statistics_by_runtime["python"]["warm_seconds"]["median"]
            / statistics_by_runtime[runtime]["warm_seconds"]["median"]
            for runtime in ("aoti", "tensorrt")
        },
        "CFD reported speedups differ from actual timings",
    )
    return comparison


def run(args):
    source_files()
    require(not args.keep_job, "CFD qualification requires owned finite-job cleanup")
    require(not args.assets, "CFD uses the explicit pinned reference/input paths")
    require(
        math.isfinite(args.job_timeout) and 180 < args.job_timeout <= 7200,
        "CFD job timeout must exceed 180 and be at most 7200 seconds",
    )
    require(
        re.fullmatch(r"[a-f0-9]{64}", args.expected_source_inventory_sha256),
        "expected source inventory SHA256 required",
    )
    for value in (
        args.repository,
        args.sdk_prefix,
        args.source_metadata,
        args.source_inventory,
    ):
        path = PurePosixPath(value)
        require(
            path.is_absolute() and ".." not in path.parts,
            "CFD image paths must be absolute without traversal",
        )
    for value in (args.reference_fields, args.input_root):
        if value:
            path = PurePosixPath(value)
            require(
                path.is_absolute()
                and ".." not in path.parts
                and path.is_relative_to(PurePosixPath(args.mount_target)),
                "CFD data paths must be inside the mounted QA root",
            )
    captured = {}
    root = args.artifact_dir.resolve() / args.run_id

    def extract(output):
        report, transport = extract_evidence(output)
        captured.update(
            report_sha256=identity(canonical(report))["sha256"], transport=transport
        )
        return report

    def validate(report, **selected):
        validate_summary(
            report,
            **selected,
            source_inventory_sha256=args.expected_source_inventory_sha256,
            mount_target=args.mount_target,
        )
        require(
            captured.get("report_sha256") == identity(canonical(report))["sha256"]
            and captured.get("transport") is not None,
            "comparison transport belongs to another summary",
        )
        raw, sessions = captured["transport"]["raw"], captured["transport"]["sessions"]
        validate_comparison(raw, sessions, report)
        (root / "comparison.json").write_bytes(raw)
        lifecycle.write_json(
            root / "comparison-proof.json",
            {
                "schema_version": 1,
                "status": "passed",
                "image_digest": args.image,
                "source_sha": args.expected_source_sha,
                "source_inventory_sha256": args.expected_source_inventory_sha256,
                "run_id": args.run_id,
                "comparison": identity(raw),
                "sessions": sessions,
                "controller": identity(Path(__file__).read_bytes()),
                "runner_sha256": FROZEN_RUNNER,
            },
        )

    code = lifecycle.run(
        args,
        command_builder=build_job_command,
        summary_extractor=extract,
        summary_validator=validate,
        historical_first=True,
    )
    if root.exists():
        lifecycle.write_json(
            root / "controller-identity.json",
            {
                "controller": identity(Path(__file__).read_bytes()),
                "runner_sha256": FROZEN_RUNNER,
                "source_inventory_sha256": args.expected_source_inventory_sha256,
            },
        )
        proof_path = root / "comparison-proof.json"
        if proof_path.exists():
            proof = strict_json(proof_path.read_bytes())
            final = strict_json((root / "summary.json").read_bytes())
            proof.update(
                status="passed"
                if code == 0 and final.get("status") == "passed"
                else "failed",
                comparison_verification="passed",
                cleanup_errors=final.get("cleanup_errors", []),
            )
            lifecycle.write_json(proof_path, proof)
    return code


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
