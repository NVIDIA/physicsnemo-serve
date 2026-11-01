"""CPU contracts for public build packaging and exact-image Lepton release gates."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import run_pnms_release as release
from scripts import run_lepton_pnms_http_qa as http
from pnms_app import images


SOURCE = "a" * 40
BASE = "registry/runtime@sha256:" + "b" * 64
IMAGE = "registry/application@sha256:" + "c" * 64
CONFIG = "sha256:" + "d" * 64
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def identity(path):
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def pilot(tmp_path):
    repository = tmp_path / "packaging-repository"
    for relative, contents in {
        "pnms": "# frozen public CLI\n",
        "Cargo.toml": "# frozen workspace manifest\n",
        "Cargo.lock": "# frozen dependency lock\n",
        "crates/worker-runtime/src/main.rs": "// frozen worker source\n",
        "application-cli/docker/Dockerfile.native": "# frozen release Dockerfile\n",
    }.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (repository / "pnms").chmod(0o755)
    source_inventory = tmp_path / "source-inventory.json"
    write(
        source_inventory,
        {
            path.relative_to(repository).as_posix(): {
                **identity(path),
                "mode": path.stat().st_mode & 0o777,
                "symlink": None,
            }
            for path in sorted(repository.rglob("*"))
            if path.is_file()
        },
    )
    project = tmp_path / "portable/project"
    project.mkdir(parents=True)
    (project / "pnms.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "pilot",
                "runtime": "cpp",
                "models": {},
                "serve": {"image": BASE},
            }
        )
    )
    root = project / ".pnms/builds" / ("1" * 32)
    bundle = root / "bundle"
    descriptor = {
        "runtime": "cpp",
        "name": "pilot",
        "source_sha": SOURCE,
        "backend": "aoti",
        "models": {"primary": {}},
        "workflow": {"config": "workflow.json"},
    }
    write(bundle / "bundle.json", descriptor)
    write(
        bundle / "workflow.json",
        {"adapter": "tensor.v1", "tensor_adapter": {"inputs": {}}},
    )
    write(
        bundle / "fixtures/cases.json",
        {
            "schema_version": 1,
            "cases": [
                {
                    "name": "first",
                    "request": {"values": [1]},
                    "expected": {"values": [3]},
                },
                {
                    "name": "second",
                    "request": {"values": [2]},
                    "expected": {"values": [5]},
                },
            ],
        },
    )
    records = [
        {"path": p.relative_to(bundle).as_posix(), **identity(p)}
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write(
        bundle / "inventory.json",
        {"schema_version": 1, "bundle_digest": digest, "files": records},
    )
    built = {
        "schema_version": 1,
        "status": "complete",
        "runtime": "cpp",
        "name": "pilot",
        "source_sha": SOURCE,
        "build_id": "1" * 32,
        "bundle": "bundle",
        "bundle_digest": digest,
        "checks": {"model": "passed", "application": "passed"},
    }
    receipt = root / "build.json"
    write(receipt, built)
    inputs = tmp_path / "release-inputs.json"
    write(
        inputs,
        {
            "schema_version": 1,
            "status": "passed",
            "runtime": "cpp",
            "project": str(project),
            "build_receipt": str(receipt),
            "build_receipt_identity": identity(receipt),
            "source_sha": SOURCE,
            "source_inventory": str(source_inventory),
            "source_inventory_sha256": identity(source_inventory)["sha256"],
            "source_inventory_identity": identity(source_inventory),
            "build_id": built["build_id"],
            "bundle_digest": digest,
            "runtime_image": BASE,
            "http_runner": "/outputs/pilot/http-runner/pnms_http_job.py",
            "http_runner_sha256": http.job.sha256(http.job.__file__),
            "contract_sha256": release.contract.sha256(release.contract.__file__),
        },
    )
    args = release.build_parser().parse_args(
        [
            "--release-inputs",
            str(inputs),
            "--expected-source-sha",
            SOURCE,
            "--runtime-image",
            BASE,
            "--image",
            "registry/application:release",
            "--run-id",
            "pilot-release",
            "--artifact-dir",
            str(tmp_path / "evidence"),
            "--repository",
            str(repository),
            "--workspace-id",
            "workspace",
            "--node-group",
            "h100",
            "--pull-secret",
            "pull",
            "--nfs-mount-base",
            "/shared",
            "--promote",
            "stable",
        ]
    )
    return args, inputs, receipt, built


class Runner:
    def __init__(self, pilot, *, package_change=None, qa_change=None, failure=None):
        self.args, _, self.receipt, self.built = pilot
        self.commands = []
        self.package_change, self.qa_change, self.failure = (
            package_change,
            qa_change,
            failure,
        )

    def __call__(self, command, **kwargs):
        command = [str(x) for x in command]
        self.commands.append(command)
        output = ""
        if "--from-build" in command:
            if self.failure == "package":
                return subprocess.CompletedProcess(command, 1, "", "packaging failed")
            built = self.built
            image = {
                "status": "complete",
                "local_tag": "registry/application:release",
                "runtime_image": BASE,
                "registry_digest": IMAGE,
                "image_id": CONFIG,
                "image_config_digest": CONFIG,
                "bundle_digest": built["bundle_digest"],
                "checks": {
                    "contents": "passed",
                    "stack": "passed",
                    "native_load": "passed",
                    "gpu_startup": "not_run",
                    "http": "not_run",
                },
                "contents": {
                    "status": "passed",
                    "bundle_digest": built["bundle_digest"],
                    "source_sha": SOURCE,
                    "native_load": "passed",
                },
            }
            package = {
                "schema_version": 1,
                "status": "complete",
                "build_id": built["build_id"],
                "root": str(self.receipt.parent),
                "image": image,
            }
            if self.package_change:
                self.package_change(package)
            output = json.dumps(package)
        elif "--mode" in command:
            if self.failure == "qa":
                return subprocess.CompletedProcess(command, 1, "", "acceptance failed")
            selected = http.build_parser().parse_args(
                command[command.index("--image") :]
            )
            descriptor = json.loads(
                (self.receipt.parent / "bundle/bundle.json").read_text()
            )
            config = json.loads(
                (self.receipt.parent / "bundle/workflow.json").read_text()
            )
            template = "tensor" if config["adapter"] == "tensor.v1" else "custom"
            target = {
                "name": "native." + descriptor["backend"] + "." + template,
                "runtime": "cpp",
                "ingress": "application/json",
                "workflow": "pilot",
                "source_sha": SOURCE,
                "bundle": "/opt/physicsnemo-release",
                "bundle_digest": self.built["bundle_digest"],
                "runtime_identity": {"kind": "registry-image", "value": IMAGE},
            }
            fixture_path = self.receipt.parent / "bundle/fixtures/cases.json"
            fixtures = json.loads(fixture_path.read_text())["cases"]
            target["application_fixtures"] = {
                "path": "fixtures/cases.json",
                "sha256": identity(fixture_path)["sha256"],
                "case_count": len(fixtures),
            }
            if selected.release_acceptance:
                target["release_acceptance"] = {
                    "schema_version": 1,
                    "fixture_indices": list(range(len(fixtures))),
                }
                target["input_files"] = {}
            names = http.job.expected_cases([target])
            report = {
                "schema_version": 1,
                "status": "passed",
                "qa_scope": "image-http",
                "run_id": selected.run_id,
                "source_sha": SOURCE,
                "image_digest": IMAGE,
                "targets": [target],
                "expected_cases": names,
                "cases": [
                    {
                        "name": name,
                        "status": "passed",
                        "details": {
                            "runner_sha256": http.job.sha256(http.job.__file__)
                        },
                    }
                    for name in names
                ],
            }
            for case in report["cases"]:
                if ".fixture." in case["name"] or case["name"].endswith(
                    (".first_request", ".repeat_request")
                ):
                    index = (
                        int(case["name"].rsplit(".", 1)[-1])
                        if ".fixture." in case["name"]
                        else int(
                            case["name"].endswith(".repeat_request")
                            and len(fixtures) > 1
                        )
                    )
                    case["details"].update(
                        artifact_downloads=fixtures[index].get(
                            "expected_artifacts", {}
                        ),
                        output_paths=[],
                        native_attempt="attempt-" + str(index),
                    )
                    case["details"]["application_fixture"] = {
                        "sha256": identity(fixture_path)["sha256"],
                        "index": index,
                        "name": fixtures[index]["name"],
                    }
            if selected.release_acceptance:
                before = {"pid": 123, "worker_instance_id": "old"}
                after = {"pid": 456, "worker_instance_id": "new"}
                for case in report["cases"]:
                    op = case["name"].removeprefix(target["name"] + ".")
                    if op == "overlap":
                        case["details"] = {
                            "requests": [
                                {
                                    "run_id": "one",
                                    "native_attempt": "attempt-one",
                                    "output_paths": [],
                                },
                                {
                                    "run_id": "two",
                                    "native_attempt": "attempt-two",
                                    "output_paths": [],
                                },
                            ],
                            "both_admitted_before_poll": True,
                            "output_isolation": "passed",
                        }
                    elif op.startswith("startup_"):
                        case["details"] = {
                            "returncode": 1,
                            "rejected_before_admission": True,
                            "worker_before": before,
                            "worker_after": before,
                        }
                    elif op in ("request_deadline", "lease_cancellation"):
                        deadline = op == "request_deadline"
                        case["details"] = {
                            "kind": "deadline" if deadline else "lease_cancellation",
                            "bundle_digest": target["bundle_digest"],
                            "runtime_identity": target["runtime_identity"],
                            "no_late_success": True,
                            "controlled_input_started": True,
                            "returncode": 1,
                            "terminal_results": int(deadline),
                            "reservation_releases": int(deadline),
                            "pending_count": int(not deadline),
                        }
                    elif op == "worker_recovery":
                        case["details"] = {
                            "worker_before": before,
                            "worker_after": after,
                            "essential_processes_unchanged": {
                                "redis-server": [{"pid": 42, "start": "100"}]
                            },
                            "post_recovery_request": {"run_id": "recovered"},
                        }
            lifecycle = {
                "status": "passed",
                "run_id": selected.run_id,
                "image_digest": IMAGE,
                "source_sha": SOURCE,
                "job_state": "Completed",
                "job_id": "owned-job",
                "cleanup_errors": [],
            }
            if self.qa_change:
                self.qa_change(lifecycle, report)
            destination = selected.artifact_dir / selected.run_id
            write(destination / "summary.json", lifecycle)
            write(destination / "job-summary.json", report)
        elif "create" in command:
            if self.failure == "promote":
                return subprocess.CompletedProcess(command, 1, "", "promotion failed")
        elif "inspect" in command:
            output = json.dumps(
                "sha256:"
                + ("f" * 64 if self.failure == "promotion-digest" else "c" * 64)
            )
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, output, "")


def test_release_packages_public_completed_build_qualifies_exact_digest_then_promotes(
    pilot,
):
    args, _, receipt, built = pilot
    original = receipt.read_bytes()
    runner = Runner(pilot)
    result = release.run(args, runner=runner)
    assert result["status"] == "passed"
    assert result["bundle_digest"] == built["bundle_digest"]
    assert result["image"] == IMAGE
    assert result["build_receipt"] == identity(receipt)
    assert receipt.read_bytes() == original
    package, qa, promote, inspect = runner.commands
    snapshot = args.artifact_dir / args.run_id / "packaging-source"
    assert package[:4] == [sys.executable, "-B", str(snapshot / "pnms"), "build"]
    assert package[package.index("--repository") + 1] == str(snapshot)
    assert qa[1:3] == ["-B", str(snapshot / "scripts/run_lepton_pnms_http_qa.py")]
    assert package[package.index("--from-build") + 1] == str(receipt)
    assert package[package.index("--image") + 1] == "registry/application:release"
    assert "--push" in package
    assert qa[qa.index("--image") + 1] == IMAGE
    assert qa[qa.index("--mode") + 1] == "image"
    assert qa[qa.index("--runner") + 1] == "/outputs/pilot/http-runner/pnms_http_job.py"
    assert "--local-summary" not in qa and "--build-receipt" not in qa
    assert "--application-fixtures" in qa
    assert promote[-1] == IMAGE and "registry/application:stable" in promote
    assert inspect[-1] == "registry/application:stable"


@pytest.mark.parametrize(
    "destination,published",
    [
        ("docker.io/acme/app", "acme/app"),
        ("acme/app", "docker.io/acme/app"),
        ("index.docker.io/acme/app", "docker.io/acme/app"),
        ("acme/app", "index.docker.io/acme/app"),
        ("docker.io/library/app", "app"),
        ("app", "docker.io/library/app"),
        ("library/app", "app"),
        ("index.docker.io/library/app", "app"),
        ("app", "index.docker.io/library/app"),
    ],
)
def test_public_repository_alias_qualifies_and_promotes_the_exact_digest(
    pilot, destination, published
):
    args, _, receipt, _ = pilot
    args.image = destination + ":release"
    reference = published + "@sha256:" + "c" * 64
    original = receipt.read_bytes()
    assert images._repository(reference) == images._repository(args.image)

    def package_change(package):
        package["image"].update(local_tag=args.image, registry_digest=reference)

    def qa_change(lifecycle, report):
        lifecycle["image_digest"] = report["image_digest"] = reference
        report["targets"][0]["runtime_identity"]["value"] = reference

    runner = Runner(pilot, package_change=package_change, qa_change=qa_change)
    result = release.run(args, runner=runner)
    assert result["status"] == "passed"
    assert result["image"] == result["acceptance"]["image"] == reference
    assert result["promotion"]["image"] == reference
    assert receipt.read_bytes() == original
    package, qa, promote, inspect = runner.commands
    assert package[package.index("--image") + 1] == args.image
    assert qa[qa.index("--image") + 1] == reference
    assert promote[-1] == reference
    assert promote[promote.index("--tag") + 1] == destination + ":stable"
    assert inspect[-1] == destination + ":stable"


@pytest.mark.parametrize(
    "destination,published",
    [
        ("docker.io/acme/app", "registry.example/acme/app"),
        ("docker.io/acme/app", "docker.io/other/app"),
        ("docker.io/acme/app", "docker.io/acme/other"),
        ("docker.io/acme/app", "docker.io:443/acme/app"),
        ("registry.example:5000/acme/app", "registry.example:5001/acme/app"),
        ("docker.io/library/app", "docker.io/app"),
        ("localhost/acme/app", "acme/app"),
    ],
)
def test_foreign_repository_or_port_prevents_acceptance(pilot, destination, published):
    args = pilot[0]
    args.image = destination + ":release"
    reference = published + "@sha256:" + "c" * 64
    assert images._repository(reference) != images._repository(args.image)

    def change(package):
        package["image"].update(local_tag=args.image, registry_digest=reference)

    runner = Runner(pilot, package_change=change)
    with pytest.raises(ValueError, match="belongs to another destination"):
        release.run(args, runner=runner)
    assert len(runner.commands) == 1


def test_tensorrt_custom_uses_the_same_coordinator_and_its_own_sealed_bundle(pilot):
    args, inputs, receipt, built = pilot
    location = receipt.parent / "bundle"
    descriptor = json.loads((location / "bundle.json").read_text())
    descriptor["backend"] = "tensorrt"
    write(location / "bundle.json", descriptor)
    write(location / "workflow.json", {"adapter": "custom.v1"})
    records = [
        {"path": p.relative_to(location).as_posix(), **identity(p)}
        for p in sorted(location.rglob("*"))
        if p.is_file() and p.name != "inventory.json"
    ]
    built["bundle_digest"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write(
        location / "inventory.json",
        {
            "schema_version": 1,
            "bundle_digest": built["bundle_digest"],
            "files": records,
        },
    )
    write(receipt, built)
    handoff = json.loads(inputs.read_text())
    handoff.update(
        bundle_digest=built["bundle_digest"], build_receipt_identity=identity(receipt)
    )
    write(inputs, handoff)
    runner = Runner(pilot)
    result = release.run(args, runner=runner)
    assert (
        result["status"] == "passed"
        and result["bundle_digest"] == built["bundle_digest"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "status",
        "build-id",
        "runtime",
        "base",
        "hash",
        "bundle",
        "runner",
        "contract",
    ],
)
def test_invalid_handoff_cannot_start_packaging(pilot, mutation):
    args, inputs, receipt, built = pilot
    d = json.loads(inputs.read_text())
    if mutation == "source":
        d["source_sha"] = "f" * 40
    elif mutation == "status":
        d["status"] = "failed"
    elif mutation == "build-id":
        d["build_id"] = "2" * 32
    elif mutation == "runtime":
        d["runtime"] = "python"
    elif mutation == "base":
        d["runtime_image"] = "registry/other@sha256:" + "f" * 64
    elif mutation == "hash":
        d["build_receipt_identity"]["sha256"] = "f" * 64
    elif mutation == "bundle":
        (receipt.parent / "bundle/workflow.json").write_text("changed")
    elif mutation == "runner":
        d["http_runner_sha256"] = "f" * 64
    else:
        d["contract_sha256"] = "f" * 64
    write(inputs, d)
    runner = Runner(pilot)
    with pytest.raises(ValueError):
        release.run(args, runner=runner)
    assert not runner.commands


@pytest.mark.parametrize(
    "field,value", [("status", "failed"), ("build_id", "2" * 32), ("root", "/foreign")]
)
def test_wrong_packaging_result_prevents_acceptance(pilot, field, value):
    runner = Runner(pilot, package_change=lambda p: p.update({field: value}))
    with pytest.raises(ValueError):
        release.run(pilot[0], runner=runner)
    assert len(runner.commands) == 1


@pytest.mark.parametrize(
    "mutation", ["digest", "base", "bundle", "contents", "source", "config", "load"]
)
def test_wrong_image_binding_prevents_acceptance(pilot, mutation):
    def change(p):
        image = p["image"]
        if mutation == "digest":
            image["registry_digest"] = "registry/foreign@sha256:" + "f" * 64
        elif mutation == "base":
            image["runtime_image"] = "registry/foreign@sha256:" + "f" * 64
        elif mutation == "bundle":
            image["bundle_digest"] = "f" * 64
        elif mutation == "contents":
            image["checks"]["contents"] = "not_run"
        elif mutation == "source":
            image["contents"]["source_sha"] = "f" * 40
        elif mutation == "config":
            image["image_config_digest"] = "tag"
        else:
            image["checks"]["native_load"] = "failed"

    runner = Runner(pilot, package_change=change)
    with pytest.raises(ValueError):
        release.run(pilot[0], runner=runner)
    assert len(runner.commands) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "cleanup",
        "lifecycle",
        "state",
        "lifecycle-image",
        "lifecycle-source",
        "source",
        "digest",
        "bundle",
        "workflow",
        "fixtures",
        "missing",
        "failed",
        "duplicate",
        "target",
        "runner",
    ],
)
def test_unqualified_image_or_incomplete_cleanup_never_promotes(pilot, mutation):
    def change(lifecycle, report):
        if mutation == "cleanup":
            lifecycle["cleanup_errors"] = ["remove failed"]
        elif mutation == "lifecycle":
            lifecycle["status"] = "failed"
        elif mutation == "state":
            lifecycle["job_state"] = "Running"
        elif mutation == "lifecycle-image":
            lifecycle["image_digest"] = "registry/foreign@sha256:" + "f" * 64
        elif mutation == "lifecycle-source":
            lifecycle["source_sha"] = "f" * 40
        elif mutation == "source":
            report["source_sha"] = "f" * 40
        elif mutation == "digest":
            report["image_digest"] = "registry/foreign@sha256:" + "f" * 64
        elif mutation == "bundle":
            report["targets"][0]["bundle_digest"] = "f" * 64
        elif mutation == "workflow":
            report["targets"][0]["workflow"] = "foreign"
        elif mutation == "fixtures":
            report["targets"][0]["application_fixtures"]["sha256"] = "f" * 64
        elif mutation == "missing":
            report["cases"].pop()
        elif mutation == "failed":
            report["cases"][0]["status"] = "failed"
        elif mutation == "duplicate":
            report["cases"].append(deepcopy(report["cases"][0]))
        elif mutation == "target":
            report["targets"].append(deepcopy(report["targets"][0]))
        else:
            report["cases"][0]["details"]["runner_sha256"] = "f" * 64

    runner = Runner(pilot, qa_change=change)
    with pytest.raises(ValueError):
        release.run(pilot[0], runner=runner)
    assert len(runner.commands) == 2
    saved = json.loads(
        (pilot[0].artifact_dir / pilot[0].run_id / "release.json").read_text()
    )
    assert saved["status"] == "failed"
    assert saved["stage"] == "acceptance"


@pytest.mark.parametrize("failure", ["package", "qa", "promote", "promotion-digest"])
def test_subprocess_failures_leave_failed_receipt(pilot, failure):
    runner = Runner(pilot, failure=failure)
    with pytest.raises(ValueError):
        release.run(pilot[0], runner=runner)
    saved = json.loads(
        (pilot[0].artifact_dir / pilot[0].run_id / "release.json").read_text()
    )
    assert saved["status"] == "failed"


@pytest.mark.parametrize("mutation", ["fixture_name", "template_identity"])
def test_acceptance_must_describe_the_sealed_producer_contract(pilot, mutation):
    def change(lifecycle, report):
        if mutation == "fixture_name":
            report["cases"][3]["details"]["application_fixture"]["name"] = (
                "foreign-fixture"
            )
        else:
            old = report["targets"][0]["name"]
            new = "native.aoti.custom"
            report["targets"][0]["name"] = new
            report["expected_cases"] = [
                name.replace(old, new) for name in report["expected_cases"]
            ]
            for case in report["cases"]:
                case["name"] = case["name"].replace(old, new)

    runner = Runner(pilot, qa_change=change)
    with pytest.raises(ValueError):
        release.run(pilot[0], runner=runner)
    assert len(runner.commands) == 2


def test_promotion_is_optional_and_run_evidence_cannot_be_reused(pilot):
    args = pilot[0]
    args.promote = None
    runner = Runner(pilot)
    assert release.run(args, runner=runner)["status"] == "passed"
    assert len(runner.commands) == 2
    with pytest.raises((ValueError, FileExistsError)):
        release.run(args, runner=runner)
    assert len(runner.commands) == 2


def test_packaging_cannot_mutate_the_original_completed_build(pilot):
    receipt = pilot[2]

    def mutate(result):
        receipt.write_text('{"status":"changed"}')

    runner = Runner(pilot, package_change=mutate)
    with pytest.raises(ValueError, match="mutated"):
        release.run(pilot[0], runner=runner)
    assert len(runner.commands) == 1


def test_stale_http_evidence_is_rejected_before_publication(pilot):
    args = pilot[0]
    (args.artifact_dir / (args.run_id + "-http")).mkdir(parents=True)
    runner = Runner(pilot)
    with pytest.raises(ValueError, match="already exists"):
        release.run(args, runner=runner)
    assert not runner.commands


@pytest.mark.parametrize(
    "mutation",
    [
        "worker_same_git_sha",
        "mode",
        "missing",
        "inventory_hash",
        "inventory_identity",
        "unsafe_relative",
        "absolute",
        "record_symlink",
        "parent_symlink",
        "record_types",
    ],
)
def test_packaging_checkout_must_match_verified_frozen_source_bytes(pilot, mutation):
    args, inputs, _, _ = pilot
    handoff = json.loads(inputs.read_text())
    inventory_path = Path(handoff["source_inventory"])
    records = json.loads(inventory_path.read_text())
    worker = args.repository / "crates/worker-runtime/src/main.rs"
    if mutation == "worker_same_git_sha":
        worker.write_text("// changed worker, unchanged Git HEAD\n")
        assert handoff["source_sha"] == args.expected_source_sha
    elif mutation == "mode":
        (args.repository / "pnms").chmod(0o644)
    elif mutation == "missing":
        worker.unlink()
    elif mutation == "inventory_hash":
        handoff["source_inventory_sha256"] = "f" * 64
    elif mutation == "inventory_identity":
        handoff["source_inventory_identity"]["size_bytes"] += 1
    elif mutation == "parent_symlink":
        original = args.repository / "crates/worker-runtime"
        original.rename(args.repository / "different-worker-dir")
        original.symlink_to(
            args.repository / "different-worker-dir", target_is_directory=True
        )
    else:
        record = records.pop("crates/worker-runtime/src/main.rs")
        if mutation == "unsafe_relative":
            records["../worker.rs"] = record
        elif mutation == "absolute":
            records[str(worker)] = record
        elif mutation == "record_symlink":
            record["symlink"] = "other.rs"
            records["crates/worker-runtime/src/main.rs"] = record
        else:
            record["mode"] = True
            records["crates/worker-runtime/src/main.rs"] = record
        write(inventory_path, records)
        handoff.update(
            source_inventory_sha256=identity(inventory_path)["sha256"],
            source_inventory_identity=identity(inventory_path),
        )
    write(inputs, handoff)
    runner = Runner(pilot)
    with pytest.raises(ValueError):
        release.run(args, runner=runner)
    assert not runner.commands, (
        "source mismatch must fail before Docker packaging or promotion"
    )


def test_generated_packaging_outputs_are_not_part_of_frozen_source_inventory(pilot):
    args = pilot[0]
    generated = args.repository / "target/generated.bin"
    generated.parent.mkdir()
    generated.write_text("build output is not frozen source\n")
    assert release.run(args, runner=Runner(pilot))["status"] == "passed"


def test_added_sitecustomize_cannot_enter_public_build_context(pilot):
    args, inputs, _, _ = pilot
    injected = args.repository / "python/sitecustomize.py"
    injected.parent.mkdir()
    injected.write_text("raise RuntimeError('unverified packaging source')\n")
    records = json.loads(
        Path(json.loads(inputs.read_text())["source_inventory"]).read_text()
    )

    class ContextRunner(Runner):
        def __call__(self, command, **kwargs):
            if "--from-build" in command:
                context = Path(command[command.index("--repository") + 1])
                assert context != args.repository
                assert not (context / "python/sitecustomize.py").exists()
                assert {
                    path.relative_to(context).as_posix()
                    for path in context.rglob("*")
                    if path.is_file()
                } == set(records)
                for relative, record in records.items():
                    copied = context / relative
                    assert identity(copied) == {
                        "sha256": record["sha256"],
                        "size_bytes": record["size_bytes"],
                    }
                    assert copied.stat().st_mode & 0o777 == record["mode"]
            return super().__call__(command, **kwargs)

    assert release.run(args, runner=ContextRunner(pilot))["status"] == "passed"


def test_packaging_child_can_verify_the_gitless_source_snapshot(pilot, monkeypatch):
    args, inputs, _, _ = pilot
    handoff = json.loads(inputs.read_text())
    monkeypatch.setenv("PNMS_SOURCE_METADATA", "/unverified/source.json")
    monkeypatch.setenv("PNMS_RELEASE_TEST_ENV", "retained")
    captured = []

    class SourceContextRunner(Runner):
        def __call__(self, command, **kwargs):
            if "--from-build" in command:
                repository = Path(command[command.index("--repository") + 1])
                assert not (repository / ".git").exists()
                output = args.artifact_dir / "docker-source"
                captured.append(
                    images.source_context(
                        repository,
                        SOURCE,
                        output,
                        native=True,
                        metadata=kwargs.get("env", {}).get("PNMS_SOURCE_METADATA"),
                    )
                )
                assert kwargs["env"]["PNMS_RELEASE_TEST_ENV"] == "retained"
                assert (output / "Cargo.toml").read_bytes() == (
                    args.repository / "Cargo.toml"
                ).read_bytes()
            return super().__call__(command, **kwargs)

    assert release.run(args, runner=SourceContextRunner(pilot))["status"] == "passed"
    assert captured[0]["kind"] == "source-inventory"
    assert captured[0]["source_sha"] == SOURCE
    assert captured[0]["source_inventory_sha256"] == handoff["source_inventory_sha256"]


def test_source_change_after_initial_verification_cannot_enter_snapshot(
    pilot, monkeypatch
):
    args = pilot[0]
    original = release._verify_source
    changed = False

    def drift(*values, **options):
        nonlocal changed
        proof = original(*values, **options)
        if not changed:
            (args.repository / "crates/worker-runtime/src/main.rs").write_text(
                "// changed after first verification\n"
            )
            changed = True
        return proof

    monkeypatch.setattr(release, "_verify_source", drift)
    runner = Runner(pilot)
    with pytest.raises(ValueError):
        release.run(args, runner=runner)
    assert not runner.commands


def test_copied_source_bytes_are_rechecked_before_packaging(pilot, monkeypatch):
    original = release.shutil.copyfile

    def corrupt(source, destination):
        result = original(source, destination)
        if Path(destination).name == "main.rs":
            Path(destination).write_text("// corrupted during snapshot copy\n")
        return result

    monkeypatch.setattr(release.shutil, "copyfile", corrupt)
    runner = Runner(pilot)
    with pytest.raises(ValueError, match="Copied packaging source differs"):
        release.run(pilot[0], runner=runner)
    assert not runner.commands


@pytest.mark.parametrize("absolute", [False, True])
def test_invalid_run_id_cannot_write_evidence_outside_artifact_directory(
    pilot, tmp_path, absolute
):
    args = pilot[0]
    outside = tmp_path / "outside-release"
    args.run_id = str(outside) if absolute else "../outside-release"
    runner = Runner(pilot)
    with pytest.raises(ValueError):
        release.run(args, runner=runner)
    assert not outside.exists()
    assert not runner.commands


@pytest.mark.parametrize(
    "workflow_id, mutation",
    [
        ("pilot", None),
        ("baked-plugin", None),
        ("baked-plugin", "plugin-digest"),
        ("baked-plugin", "empty-workflow"),
        ("baked-plugin", "invalid-workflow"),
        ("baked-plugin", "http-workflow"),
        ("baked-plugin", "http-source"),
        ("baked-plugin", "http-image"),
        ("baked-plugin", "http-receipt"),
    ],
)
def test_python_completed_build_uses_its_existing_receipt_bound_image_qa(
    pilot, workflow_id, mutation
):
    args, inputs, receipt, built = pilot
    project = Path(json.loads(inputs.read_text())["project"])
    (project / "pnms.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "pilot",
                "runtime": "python",
                "plugin": ".",
                "serve": {"image": BASE},
            }
        )
    )
    plugin = receipt.parent / "python/plugin"
    plugin.mkdir(parents=True)
    (plugin / "workflow.py").write_text("# frozen Python plugin\n")
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"metadata": {"id": workflow_id}})
    )
    files = [{"path": path.name, **identity(path)} for path in sorted(plugin.iterdir())]
    built.update(
        runtime="python", python={"plugin_root": "python/plugin", "files": files}
    )
    write(receipt, built)
    handoff = json.loads(inputs.read_text())
    handoff.update(runtime="python", build_receipt_identity=identity(receipt))
    write(inputs, handoff)

    class PythonRunner(Runner):
        def __call__(self, command, **kwargs):
            if "--mode" not in command:
                result = super().__call__(command, **kwargs)
                if "--from-build" in command:
                    package = json.loads(result.stdout)
                    package["image"]["checks"]["python_base"] = "passed"
                    package["image"]["contents"] = {
                        "status": "passed",
                        "workflow_id": yaml.safe_load(
                            (plugin / "plugin.yaml").read_text()
                        )["metadata"]["id"],
                        "plugin_digest": release.digest(files),
                    }
                    if mutation == "plugin-digest":
                        package["image"]["contents"]["plugin_digest"] = "f" * 64
                    elif mutation in ("empty-workflow", "invalid-workflow"):
                        package["image"]["contents"]["workflow_id"] = (
                            "" if mutation == "empty-workflow" else 123
                        )
                    result.stdout = json.dumps(package)
                return result
            self.commands.append(command)
            selected = http.build_parser().parse_args(
                command[command.index("--image") :]
            )
            assert "--application-fixtures" not in command
            augmented = Path(selected.python_build_receipt)
            assert (
                json.loads(augmented.read_text())["image"]["registry_digest"] == IMAGE
            )
            binding = http._python_release_receipt(selected)
            assert binding["workflow"] == workflow_id
            target = {
                "name": "python.release",
                "runtime": "python",
                "workflow": binding["workflow"],
                "source_sha": SOURCE,
                "plugin": "/opt/pnms/plugin",
                "plugin_digest": binding["plugin_digest"],
                "build_receipt_sha256": binding["receipt_sha256"],
                "runtime_identity": {"kind": "registry-image", "value": IMAGE},
            }
            if mutation == "http-workflow":
                target["workflow"] = "foreign-plugin"
            elif mutation == "http-source":
                target["source_sha"] = "f" * 40
            elif mutation == "http-image":
                target["runtime_identity"]["value"] = (
                    "registry/application@sha256:" + "f" * 64
                )
            elif mutation == "http-receipt":
                target["build_receipt_sha256"] = "f" * 64
            names = http.job.expected_cases([target])
            report = {
                "schema_version": 1,
                "run_id": selected.run_id,
                "source_sha": SOURCE,
                "image_digest": IMAGE,
                "status": "passed",
                "qa_scope": "image-http",
                "targets": [target],
                "expected_cases": names,
                "cases": [
                    {
                        "name": name,
                        "status": "passed",
                        "details": {
                            "runner_sha256": http.job.sha256(http.job.__file__),
                            "python_release": target,
                        },
                    }
                    for name in names
                ],
            }
            destination = selected.artifact_dir / selected.run_id
            write(destination / "job-summary.json", report)
            write(
                destination / "summary.json",
                {
                    "status": "passed",
                    "run_id": selected.run_id,
                    "image_digest": IMAGE,
                    "source_sha": SOURCE,
                    "job_state": "Completed",
                    "job_id": "owned-python-job",
                    "cleanup_errors": [],
                },
            )
            return subprocess.CompletedProcess(command, 0, "", "")

    original = receipt.read_bytes()
    runner = PythonRunner(pilot)
    if mutation:
        with pytest.raises(ValueError):
            release.run(args, runner=runner)
        assert len(runner.commands) == (2 if mutation.startswith("http-") else 1)
    else:
        result = release.run(args, runner=runner)
        assert result["status"] == "passed"
        assert result["runtime"] == "python"
    assert receipt.read_bytes() == original


def workflow():
    return yaml.load(
        (ROOT / ".github/workflows/lepton_pnms_release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )


def test_release_workflow_builds_then_packages_qualifies_promotes_public_path():
    doc = workflow()
    assert set(doc["on"]) == {"workflow_dispatch", "workflow_call"}
    called = doc["on"]["workflow_call"]["inputs"]
    for field in ("image", "source_sha", "repository", "runtime_image"):
        assert called[field]["required"] == "true" and called[field]["type"] == "string"
    for field in ("backend", "template", "project", "promote"):
        assert called[field]["type"] == "string"
    inputs = doc["on"]["workflow_dispatch"]["inputs"]
    assert {
        "image",
        "source_sha",
        "backend",
        "template",
        "project",
        "repository",
        "runtime_image",
        "promote",
    } <= set(inputs)
    assert set(inputs["backend"]["options"]) == {"aoti", "tensorrt"}
    steps = doc["jobs"]["release"]["steps"]
    build = next(
        s for s in steps if "run_lepton_pnms_release_build.py" in s.get("run", "")
    )
    package = next(s for s in steps if "run_pnms_release.py" in s.get("run", ""))
    assert steps.index(build) < steps.index(package)
    for step in (build, package):
        assert "continue-on-error" not in step and "if" not in step
    assert (
        "--release-inputs" in package["run"] and "release-inputs.json" in package["run"]
    )
    assert "--runtime-image" in build["run"] and "--runtime-image" in package["run"]
    assert (
        "--expected-source-sha" in build["run"]
        and "--expected-source-sha" in package["run"]
    )
    assert any(
        s.get("uses", "").startswith("actions/upload-artifact@")
        and s.get("if") == "always()"
        for s in steps
    )
    body = json.dumps(doc)
    assert (
        "plugin_dev.py release" not in body
        and "docker.sock" not in body
        and "--privileged" not in body
    )


def test_called_release_backends_use_distinct_run_and_evidence_identities():
    doc = workflow()
    assert "${{ inputs.backend }}" in doc["concurrency"]["group"]
    release = doc["jobs"]["release"]
    assert release["runs-on"] == "linux-amd64-cpu8"
    assert "${{ inputs.backend }}" in release["env"]["PNMS_RUN"]
    upload = next(
        step
        for step in release["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert "${{ inputs.backend }}" in upload["with"]["name"]
    assert upload["if"] == "always()"


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "destination",
        "source",
        "build-image",
        "runtime-image",
        "project",
        "backend",
        "template",
    ],
)
def test_ci_selection_is_validated_before_allocating_a_gpu(invalid):
    steps = workflow()["jobs"]["release"]["steps"]
    step = next(
        (item for item in steps if item.get("name") == "Validate release inputs"), None
    )
    assert step is not None, (
        "invalid release selections must fail before GPU compilation"
    )
    build = next(
        item
        for item in steps
        if "run_lepton_pnms_release_build.py" in item.get("run", "")
    )
    assert steps.index(step) < steps.index(build)
    env = os.environ | {
        "PNMS_DESTINATION": "registry/application",
        "PNMS_SOURCE_SHA": SOURCE,
        "PNMS_BUILD_IMAGE": IMAGE,
        "PNMS_RUNTIME_IMAGE": BASE,
        "PNMS_PROJECT": "",
        "PNMS_BACKEND": "aoti",
        "PNMS_TEMPLATE": "tensor",
    }
    if invalid == "destination":
        env["PNMS_DESTINATION"] = "registry/application:tag"
    elif invalid == "source":
        env["PNMS_SOURCE_SHA"] = "short"
    elif invalid == "build-image":
        env["PNMS_BUILD_IMAGE"] = "registry/build:mutable"
    elif invalid == "runtime-image":
        env["PNMS_RUNTIME_IMAGE"] = "registry/runtime:mutable"
    elif invalid == "project":
        env["PNMS_PROJECT"] = "../outside"
    elif invalid == "backend":
        env["PNMS_BACKEND"] = "unsupported"
    elif invalid == "template":
        env["PNMS_TEMPLATE"] = "unsupported"
    result = subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True
    )
    assert (result.returncode == 0) is (invalid is None)


@pytest.mark.parametrize(
    "controller", ["run_lepton_pnms_release_build.py", "run_pnms_release.py"]
)
@pytest.mark.parametrize("optional", [False, True])
def test_ci_shell_forwards_release_handoff_and_inputs_literally(
    tmp_path, controller, optional
):
    steps = workflow()["jobs"]["release"]["steps"]
    step = next(item for item in steps if controller in item.get("run", ""))
    executable = tmp_path / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\npathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    executable.chmod(0o755)
    capture = tmp_path / "argv.json"
    literal = "project $(touch injected) with spaces"
    env = os.environ | {
        "CAPTURE": str(capture),
        "PNMS_BUILD_IMAGE": IMAGE,
        "PNMS_SOURCE_SHA": SOURCE,
        "PNMS_BACKEND": "tensorrt",
        "PNMS_TEMPLATE": "custom",
        "PNMS_RUNTIME_IMAGE": BASE,
        "PNMS_RUN": "release-ci",
        "PNMS_DESTINATION": "registry/application",
        "PNMS_PROJECT": literal if optional else "",
        "PNMS_PROMOTE": "stable$(touch injected)" if optional else "",
    }
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[0] == "scripts/" + controller
    assert argv[argv.index("--expected-source-sha") + 1] == SOURCE
    assert argv[argv.index("--runtime-image") + 1] == BASE
    assert not (tmp_path / "injected").exists()
    if controller == "run_lepton_pnms_release_build.py":
        assert argv[argv.index("--backend") + 1] == "tensorrt"
        assert argv[argv.index("--template") + 1] == "custom"
        assert argv[argv.index("--image") + 1] == IMAGE
        assert ("--project" in argv) is optional
        if optional:
            assert argv[argv.index("--project") + 1] == literal
    else:
        assert (
            argv[argv.index("--release-inputs") + 1]
            == "qa/artifacts/pnms-release/release-ci-build/release-inputs.json"
        )
        assert argv[argv.index("--image") + 1] == "registry/application:release-ci"
        assert ("--promote" in argv) is optional
        if optional:
            assert argv[argv.index("--promote") + 1] == "stable$(touch injected)"


def test_native_promotion_requires_expanded_release_acceptance(pilot):
    args, _, receipt, built = pilot
    runner = Runner(pilot)
    release.run(args, runner=runner)
    qa = next(command for command in runner.commands if "--mode" in command)
    assert "--release-acceptance" in qa


@pytest.mark.parametrize(
    "mutation",
    [
        "download-golden",
        "overlap-path",
        "recovery-stream",
        "startup-empty",
        "fixture-missing",
    ],
)
def test_expanded_promotion_rejects_forged_request_or_fault_evidence(pilot, mutation):
    args, _, _, _ = pilot

    def change(lifecycle, report):
        def detail(operation):
            return next(
                case["details"]
                for case in report["cases"]
                if case["name"].endswith("." + operation)
            )

        if mutation == "download-golden":
            detail("fixture.0")["artifact_downloads"] = {
                "unexpected": {"sha256": "f" * 64, "size_bytes": 1}
            }
        elif mutation == "overlap-path":
            for request in detail("overlap")["requests"]:
                request["output_paths"] = ["/outputs/shared/result.bin"]
        elif mutation == "recovery-stream":
            detail("worker_recovery")["worker_before"]["stream"] = "old"
            detail("worker_recovery")["worker_after"]["stream"] = "new"
        elif mutation == "startup-empty":
            detail("startup_gpu_rejection").update(worker_before={}, worker_after={})
        else:
            report["cases"] = [
                case
                for case in report["cases"]
                if not case["name"].endswith(".fixture.1")
            ]

    runner = Runner(pilot, qa_change=change)
    with pytest.raises(ValueError):
        release.run(args, runner=runner)
    assert not any("create" in command for command in runner.commands)
