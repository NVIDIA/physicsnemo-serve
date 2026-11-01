"""CPU contracts for the host-supplied final CFD image acceptance controller."""

import hashlib
import json
import shlex
from pathlib import Path, PureWindowsPath

import pytest

from scripts import run_lepton_pnms_cfd_image_qa as controller


def identity(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def remote():
    namespace = {"__name__": "cfd_image_unit"}
    exec(controller.REMOTE, namespace)  # noqa: S102 - Exercise the shipped wrapper.
    return namespace


@pytest.fixture
def args(tmp_path):
    return controller.build_parser().parse_args(
        [
            "--image",
            "nvcr.io/example/cfd@sha256:" + "a" * 64,
            "--expected-source-sha",
            "b" * 40,
            "--expected-source-inventory-sha256",
            "c" * 64,
            "--expected-bundle-digest",
            "d" * 64,
            "--backend",
            "aoti",
            "--run-id",
            "image-aoti-1",
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "workspace",
            "--node-group",
            "group",
            "--pull-secret",
            "pull",
            "--nfs-mount-base",
            "/qa",
            "--lustre-dir",
            "results",
        ]
    )


@pytest.fixture
def requests():
    rows = []
    for ordinal in (0, 1):
        rows.append(
            {
                "run_id": "request-" + str(ordinal),
                "seconds": 12.5,
                "parity": "passed",
                "artifacts_verified": True,
                "canonical_report_sha256": "e" * 64,
                "output": "/outputs/native-inference/image-aoti-1/request-"
                + str(ordinal),
                "files": {
                    name: identity(raw)
                    for name, raw in {
                        "physical_fields": b"physical",
                        "standardized_fields": b"standardized",
                        "native_metadata": b"metadata" + bytes([ordinal]),
                        "http-results.json": b"http" + bytes([ordinal]),
                    }.items()
                },
                "artifact_paths": {
                    name: "/service/attempt-" + str(ordinal) + "/" + name
                    for name in (
                        "physical_fields",
                        "standardized_fields",
                        "native_metadata",
                    )
                },
                "metadata": {"session_load_count": 2, "initialization_ms": 123.5},
            }
        )
    return rows


@pytest.fixture
def startup():
    return {
        "status": "passed",
        "policy": "expected-fp32-v1",
        "session_load_count": 2,
        "models": [
            {
                "point_count": points,
                "backend": "aoti",
                "shape": [1, points, 4],
                "dtype": "float32",
                "finite": True,
                "comparison": {
                    "rtol": 0.001,
                    "atol": 0.001,
                    "max_absolute_error": 0.00001,
                    "max_relative_error": 0.00001,
                },
            }
            for points in (60000, 8095)
        ],
    }


@pytest.fixture
def report(args, requests, startup):
    for index, row in enumerate(requests):
        row["provenance"] = {
            "source_sha": args.expected_source_sha,
            "backend": args.backend,
            "release_image": args.image,
            "model_build_sources": {
                "full": controller.ORIGIN,
                "tail": controller.ORIGIN,
            },
            "attempt": ("1" if index == 0 else "2") * 8
            + "-1111-1111-1111-111111111111",
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "profile": "cfd-image",
        "run_id": args.run_id,
        "source_sha": args.expected_source_sha,
        "image_digest": args.image,
        "backend": args.backend,
        "bundle_digest": args.expected_bundle_digest,
        "source_inventory_sha256": args.expected_source_inventory_sha256,
        "benchmark_sha256": controller.BENCHMARK_SHA256,
        "controller_sha256": identity(Path(controller.__file__).read_bytes())["sha256"],
        "expected_cases": controller.GATES,
        "cases": [{"name": name, "status": "passed"} for name in controller.GATES],
        "model_build_sources": {"full": controller.ORIGIN, "tail": controller.ORIGIN},
        "recompiled": False,
        "source_bindings": {
            "workspace_files": 950,
            "installed_files": 100,
            "installed_inventory_sha256": "1" * 64,
            "binaries": {
                name: identity(b"binary")
                for name in ("native-worker", "worker-runtime", "inference_server")
            },
        },
        "startup": startup,
        "requests": requests,
        "residency": [
            {
                "pid": 42,
                "start": "100",
                "worker_instance_id": "worker-uuid",
                "bundle_digest": args.expected_bundle_digest,
                "library": {
                    "path": "/opt/physicsnemo-release/workflow/lib/libtransolver_workflow.so",
                    **identity(b"library"),
                },
            }
            for _ in range(3)
        ],
        "hardware": {
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": "GPU-123",
            "compute_capability": [9, 0],
        },
        "service_command": ["python3", "-m", "pnms_app.image_entrypoint"],
        "shutdown": {"returncode": 0, "descendants_remaining": []},
    }


def validate(report, args):
    return controller.validate_summary(
        report,
        run_id=args.run_id,
        source_sha=args.expected_source_sha,
        image_digest=args.image,
        profile=args.profile,
        backend=args.backend,
        bundle_digest=args.expected_bundle_digest,
        source_inventory_sha256=args.expected_source_inventory_sha256,
    )


def test_image_controller_pins_the_current_benchmark_bytes():
    benchmark = controller.ROOT / "qa/scripts/benchmark_pnms_cfd.py"
    assert identity(benchmark.read_bytes())["sha256"] == controller.BENCHMARK_SHA256


def test_genuine_full_report_is_accepted(report, args):
    validate(report, args)


def test_linux_artifact_paths_are_validated_independently_of_host_path_rules(requests):
    module = remote()
    module["Path"] = PureWindowsPath
    module["validate_requests"](requests)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(status="prepared"),
        lambda r: r.update(backend="tensorrt"),
        lambda r: r.update(bundle_digest="9" * 64),
        lambda r: r.update(source_inventory_sha256="9" * 64),
        lambda r: r["model_build_sources"].update(tail="f" * 40),
        lambda r: r["cases"].pop(),
        lambda r: r["requests"][1].update(run_id=r["requests"][0]["run_id"]),
        lambda r: r["requests"][1]["metadata"].update(session_load_count=4),
        lambda r: r["requests"][1]["metadata"].update(initialization_ms=456),
        lambda r: r["residency"][2].update(start="101"),
        lambda r: r["requests"][1].update(
            artifact_paths=r["requests"][0]["artifact_paths"]
        ),
        lambda r: r["startup"]["models"][1].update(comparison=None),
        lambda r: r["shutdown"].update(descendants_remaining=[42]),
    ],
)
def test_stale_partial_foreign_or_nonresident_reports_fail(report, args, mutation):
    mutation(report)
    with pytest.raises(ValueError):
        validate(report, args)


def test_installed_source_maps_only_dockerfile_copied_subsets_and_detects_tampering(
    tmp_path,
):
    workspace, installed = tmp_path / "workspace", tmp_path / "installed"
    inventory = {}
    for relative in (
        "application-cli/src/pnms_app/image_entrypoint.py",
        "scripts/loader.py",
        "python/runtime.py",
        "crates/worker-runtime/config/config.json",
        "qa/only-in-workspace.py",
    ):
        source = workspace / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(relative.encode())
        inventory[relative] = identity(source.read_bytes())
        if not relative.startswith("qa/"):
            target = installed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps(inventory))
    meta = tmp_path / "source.json"
    meta.write_text(
        json.dumps(
            {
                "source_sha": "b" * 40,
                "source_inventory_sha256": identity(manifest.read_bytes())["sha256"],
            }
        )
    )
    result = remote()["bind_sources"](
        workspace,
        installed,
        meta,
        manifest,
        "b" * 40,
        identity(manifest.read_bytes())["sha256"],
    )
    assert result["workspace_files"] == 5
    assert result["installed_files"] == 4
    (installed / "scripts/loader.py").write_bytes(b"stale")
    with pytest.raises(ValueError, match="installed"):
        remote()["bind_sources"](
            workspace,
            installed,
            meta,
            manifest,
            "b" * 40,
            identity(manifest.read_bytes())["sha256"],
        )


def test_startup_requires_each_independent_golden(startup):
    module = remote()
    module["validate_startup"](startup, "aoti")
    startup["models"][1]["comparison"] = None
    with pytest.raises(ValueError, match="golden"):
        module["validate_startup"](startup, "aoti")


def test_public_shipped_entrypoint_is_the_service_command(args, tmp_path):
    config = {
        "image": args.image,
        "source_sha": args.expected_source_sha,
        "backend": args.backend,
        "run_id": args.run_id,
        "output": str(tmp_path / "run"),
        "mount": "/outputs",
        "timeout": 7000,
    }
    output = tmp_path / "run"
    output.mkdir()
    command, files = output / "command.json", output / "files.json"
    selected = remote()["session_arguments"](config, output, command, files)
    assert json.loads(command.read_text())[-2:] == ["-m", "pnms_app.image_entrypoint"]
    assert selected.native_bundle == Path("/opt/physicsnemo-release")
    assert selected.service_cwd == Path("/opt/pnms/source")


def test_job_command_is_finite_and_runs_host_wrapper_inside_exact_image(args):
    command = controller.build_job_command(args, "owned-job")
    assert command[command.index("--container-image") + 1] == args.image
    runner = shlex.split(command[command.index("--command") + 1])
    assert runner[:2] == ["exec", "/usr/bin/timeout"]
    assert controller.REMOTE in runner
    config = json.loads(runner[-1])
    assert config["backend"] == args.backend
    assert config["bundle_digest"] == args.expected_bundle_digest
    assert (
        config["controller_sha256"]
        == identity(Path(controller.__file__).read_bytes())["sha256"]
    )


def test_wrapper_uses_existing_http_constructor_and_waits_for_shipped_service(tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(url="http://127.0.0.1:8080", request_timeout=1800)
    process, stream = object(), object()
    observed = []

    class HTTP:
        def __init__(self, base_url):
            self.base_url = base_url

    benchmark = {
        "start_service": lambda selected, context: (process, stream),
        "HTTP": HTTP,
        "wait_ready": lambda selected, context, http, child: observed.append(
            (http.base_url, child)
        ),
    }
    actual = remote()["start_stack"](args, {}, benchmark)
    assert actual[:2] == (process, stream)
    assert observed == [(args.url, process)]


def test_request_record_uses_real_http_storage_paths_and_downloaded_metadata(requests):
    row = requests[0]
    actual = {
        "execution": {
            "outputs": [
                {"name": name, "storage_path": path, **row["files"][name]}
                for name, path in row["artifact_paths"].items()
            ]
        },
        "payload": {
            "native": {
                "source_sha": "b" * 40,
                "backend": "aoti",
                "release_image": "registry@sha256:" + "a" * 64,
                "model_build_sources": {
                    "full": controller.ORIGIN,
                    "tail": controller.ORIGIN,
                },
            }
        },
    }
    selected = {**row, "details": {"metadata": row["metadata"]}}
    record = remote()["request_record"](selected, Path(row["output"]), actual)
    assert record["artifact_paths"] == row["artifact_paths"]
    assert record["metadata"] == row["metadata"]


def test_failed_readiness_retires_the_started_service_and_closes_its_log():
    from types import SimpleNamespace

    process = object()
    observed = []

    class Stream:
        def close(self):
            observed.append("closed")

    def fail(*args):
        raise ValueError("readiness failed")

    benchmark = {
        "start_service": lambda *args: (process, Stream()),
        "HTTP": lambda *args: object(),
        "wait_ready": fail,
        "stop_service": lambda child, timeout: observed.append(child),
    }
    with pytest.raises(ValueError, match="readiness failed"):
        remote()["start_stack"](
            SimpleNamespace(url="http://local", request_timeout=10, shutdown_timeout=1),
            {},
            benchmark,
        )
    assert observed == [process, "closed"]


def encoded_report(report, raw=None):
    import base64

    raw = raw or json.dumps(report, sort_keys=True).encode()
    return (
        controller.BEGIN
        + "\n"
        + json.dumps({"report_base64": base64.b64encode(raw).decode(), **identity(raw)})
        + "\n"
        + controller.END
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "f" * 40),
        ("backend", "tensorrt"),
        ("release_image", "foreign@sha256:" + "f" * 64),
        ("model_build_sources", {"full": controller.ORIGIN, "tail": "f" * 40}),
    ],
)
def test_each_request_carries_actual_bound_source_backend_image_and_model_origin(
    report, args, field, value
):
    report["requests"][1]["provenance"][field] = value
    with pytest.raises(ValueError, match="request.*provenance"):
        validate(report, args)


@pytest.mark.parametrize(
    "raw", [b'{"status":"passed","status":"failed"}', b'{"duration":NaN}']
)
def test_transport_rejects_duplicate_and_nonfinite_json(raw):
    with pytest.raises(ValueError):
        controller.extract_summary(encoded_report({}, raw))


@pytest.mark.parametrize("failure", [None, "cleanup", "source"])
def test_actual_host_lifecycle_uses_history_preserves_raw_report_and_removes_owned_job(
    report, args, monkeypatch, failure
):
    if failure == "source":
        report["source_sha"] = "f" * 40
    raw = json.dumps(report, sort_keys=True, indent=1).encode()
    logs = encoded_report(report, raw)
    history = "\n".join(
        "2026-10-29 12:00:00.123456|" + json.dumps(line) for line in logs.splitlines()
    )
    calls = []
    removed = False

    def lep(command, **kwargs):
        nonlocal removed
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: owned-image-job\n"
        if command[:2] == ["job", "get"]:
            return (1, "404 Not Found") if removed else (0, '{"state":"Completed"}')
        if command[:2] == ["log", "get"]:
            return 0, history
        if command == ["job", "remove", "-i", "owned-image-job"]:
            removed = failure != "cleanup"
            return (7, "cleanup failed") if failure == "cleanup" else (0, "removed")
        pytest.fail("unexpected Lepton call: " + repr(command))

    monkeypatch.setattr(controller.lifecycle, "lep", lep)
    code = controller.run(args)
    root = args.artifact_dir / args.run_id
    assert code == (0 if failure is None else 1)
    assert ["job", "remove", "-i", "owned-image-job"] in calls
    if failure != "cleanup":
        assert calls[-1] == ["job", "get", "-i", "owned-image-job"]
    assert not any(call[:2] == ["job", "log"] for call in calls)
    proof = json.loads((root / "image-proof.json").read_bytes())
    assert proof["status"] == ("passed" if failure is None else "failed")
    if failure != "cleanup":
        assert proof["resource_cleanup"]["status"] == "absent-404"
    if failure == "source":
        assert not (root / "image-report.json").exists()
    else:
        assert (root / "image-report.json").read_bytes() == raw
        assert proof["report"] == identity(raw)


@pytest.mark.parametrize(
    "option,value",
    [
        ("job_timeout", float("inf")),
        ("job_timeout", 180),
        ("keep_job", True),
        ("expected_bundle_digest", "short"),
        ("reference_fields", "/etc/foreign.json"),
        ("input_root", "/etc/foreign"),
    ],
)
def test_invalid_or_unbounded_requests_fail_before_any_submission(
    args, monkeypatch, option, value
):
    setattr(args, option, value)
    monkeypatch.setattr(
        controller.lifecycle,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid image QA reached Lepton"),
    )
    with pytest.raises(ValueError):
        controller.run(args)
