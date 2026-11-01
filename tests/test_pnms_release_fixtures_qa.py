"""Sealed application fixtures over fake HTTP; no local acceptance services."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "application-cli/src"))
from pnms_app.common import digest, inventory  # noqa: E402
from qa.native_inference import pnms_http_job as job  # noqa: E402
from scripts import run_lepton_pnms_http_qa as controller  # noqa: E402
from test_pnms_http_qa import controller_args  # noqa: E402
from test_pnms_python_cli import executable as python_executable  # noqa: E402

SOURCE = "a" * 40
IMAGE = "registry/release@sha256:" + "b" * 64
CASES = [
    {
        "name": "gain",
        "request": {"samples": [2, 5], "gain": 3},
        "expected": {"score": 21.0, "label": "custom"},
        "atol": 0.01,
        "rtol": 0,
    },
    {
        "name": "negative",
        "request": {"samples": [-2, 1], "gain": 4},
        "expected": {"score": -4.0, "label": "custom"},
    },
]


def sealed_bundle(tmp_path, *, cases=None, template="custom", ingress=None):
    root = tmp_path / "bundle"
    root.mkdir()
    job.write_json(
        root / "bundle.json",
        {
            "source_sha": SOURCE,
            "name": "my-project",
            "backend": "tensorrt",
            "models": {"primary": "models/primary"},
            "workflow": {"config": "configuration/workflow.json"},
        },
    )
    job.write_json(
        root / "configuration/workflow.json",
        {
            "adapter": template + ".v1",
            "tensor_adapter": {
                "inputs": {"x": {"codec": "json-array-fp32"}},
            },
        },
    )
    job.write_json(
        root / "configuration/plugin/my-project/plugin.yaml",
        {"ingress": ingress or {"content_type": "application/json"}},
    )
    job.write_json(
        root / "fixtures/cases.json",
        {
            "schema_version": 1,
            "cases": copy.deepcopy(CASES if cases is None else cases),
        },
    )
    reseal(root)
    return root


def reseal(root):
    files = inventory(root, exclude=("inventory.json",))
    job.write_json(
        root / "inventory.json",
        {"schema_version": 1, "files": files, "bundle_digest": digest(files)},
    )


def selected(root):
    return job._bundle_target(
        root,
        source_sha=SOURCE,
        runtime_identity=job._expected_identity(IMAGE),
        application_fixtures=True,
    )


@pytest.mark.parametrize("template", ["tensor", "custom"])
def test_target_binds_real_sealed_fixture_bytes_and_count(tmp_path, template):
    root = sealed_bundle(tmp_path, template=template)
    target = selected(root)
    assert target["name"] == f"native.tensorrt.{template}"
    assert target["application_fixtures"] == {
        "path": "fixtures/cases.json",
        "sha256": job.sha256(root / "fixtures/cases.json"),
        "case_count": 2,
    }
    assert "application_fixtures" not in job._bundle_target(root, source_sha=SOURCE)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty",
        "no-assertion",
        "nonobject-request",
        "inputs",
        "negative-tolerance",
        "bool-tolerance",
        "multipart",
        "raw",
    ],
)
def test_fixture_scope_rejects_unsupported_or_vacuous_checks(tmp_path, mutation):
    cases = copy.deepcopy(CASES)
    ingress = None
    if mutation == "empty":
        cases = []
    elif mutation == "no-assertion":
        cases[0]["expected"] = {"nested": {}}
    elif mutation == "nonobject-request":
        cases[0]["request"] = [1, 2]
    elif mutation == "inputs":
        cases[0]["inputs"] = {"upload": "fixtures/input.bin"}
    elif mutation == "negative-tolerance":
        cases[0]["rtol"] = -1
    elif mutation == "bool-tolerance":
        cases[0]["atol"] = True
    elif mutation == "multipart":
        ingress = {"content_types": ["application/json", "multipart/form-data"]}
    root = sealed_bundle(
        tmp_path,
        cases=cases,
        ingress=ingress,
        template="tensor" if mutation == "raw" else "custom",
    )
    if mutation == "raw":
        config = root / "configuration/workflow.json"
        value = json.loads(config.read_text())
        value["tensor_adapter"]["inputs"]["x"]["codec"] = "raw-fp32-le"
        job.write_json(config, value)
        reseal(root)
    with pytest.raises(ValueError, match="fixture|JSON"):
        selected(root)


def result(target, payload):
    return {
        "payload": payload,
        "execution": {
            "status": "succeeded",
            "run_id": "request-1",
            "native": {
                "runtime": "cpp",
                "source_sha": SOURCE,
                "runtime_identity": target["runtime_identity"],
                "metadata": {"session_load_count": 1},
            },
        },
    }


@pytest.mark.parametrize(
    "mutation", [None, "value", "source", "identity", "session", "run", "status"]
)
def test_authored_numerics_do_not_bypass_canonical_native_identity(tmp_path, mutation):
    target = selected(sealed_bundle(tmp_path))
    report = result(target, {"score": 21.005, "label": "custom"})
    if mutation == "value":
        report["payload"]["score"] = 21.02
    elif mutation == "source":
        report["execution"]["native"]["source_sha"] = "f" * 40
    elif mutation == "identity":
        report["execution"]["native"]["runtime_identity"] = {}
    elif mutation == "session":
        report["execution"]["native"]["metadata"]["session_load_count"] = 2
    elif mutation == "run":
        report["execution"]["run_id"] = "foreign"
    elif mutation == "status":
        report["execution"]["status"] = "failed"
    if mutation is None:
        job.assert_result(report, run_id="request-1", target=target, fixture=CASES[0])
    else:
        with pytest.raises(ValueError):
            job.assert_result(
                report, run_id="request-1", target=target, fixture=CASES[0]
            )


@pytest.mark.parametrize("count", [1, 2])
def test_image_requests_use_declared_cases_and_nonobject_schema_rejection(
    tmp_path, monkeypatch, count
):
    root = sealed_bundle(tmp_path, cases=CASES[:count])
    target = selected(root)
    requests = []
    details = {}

    class HTTP:
        def __init__(self, *args, **kwargs):
            self.current = None

        def request(self, path, *, body=None):
            if path == "/readyz":
                return 200, {
                    "ready": True,
                    "native": {
                        "ready": True,
                        "workflow_id": target["workflow"],
                        "bundle_digest": target["bundle_digest"],
                        "runtime_identity": target["runtime_identity"],
                        "worker_instance_id": "resident",
                        "deployment_id": "deployment-1",
                    },
                }
            if "unknown-" in path:
                return 404, {}
            if body is not None:
                requests.append(body)
                if not isinstance(body["parameters"], dict):
                    return 422, {}
                self.current = next(
                    c for c in CASES if c["request"] == body["parameters"]
                )
                return 202, {
                    "workflow": target["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                    "run_id": "request-1",
                }
            if path.endswith("/status"):
                return 200, {"status": "succeeded"}
            return 200, result(target, self.current["expected"])

    def case(name, function):
        details[name] = function()
        return details[name]

    monkeypatch.setattr(job, "HTTP", HTTP)
    monkeypatch.setattr(job, "_port", lambda: 12345)
    ctx = SimpleNamespace(
        root=tmp_path,
        environment={},
        case=case,
        args=SimpleNamespace(
            base_url="https://unit.invalid",
            deployment_id="deployment-1",
            request_timeout=1,
            startup_timeout=1,
        ),
    )
    job.run_target(ctx, target)
    assert requests == [
        {"parameters": CASES[0]["request"]},
        {"parameters": CASES[min(1, count - 1)]["request"]},
        {"parameters": []},
    ]
    for operation, index in [
        ("first_request", 0),
        ("repeat_request", min(1, count - 1)),
    ]:
        assert details[target["name"] + "." + operation]["application_fixture"] == {
            "index": index,
            "name": CASES[index]["name"],
            "sha256": target["application_fixtures"]["sha256"],
        }


def test_fixture_file_changed_after_target_selection_is_rejected(tmp_path, monkeypatch):
    root = sealed_bundle(tmp_path)
    target = selected(root)
    job.write_json(
        root / "fixtures/cases.json", {"schema_version": 1, "cases": CASES[:1]}
    )
    with pytest.raises(ValueError, match="fixture"):
        job.application_fixture_cases(target)


def test_empty_expected_array_remains_a_real_length_assertion(tmp_path):
    cases = [{"name": "empty-output", "request": {}, "expected": {"tokens": []}}]
    target = selected(sealed_bundle(tmp_path, cases=cases))
    job.assert_result(
        result(target, {"tokens": []}),
        run_id="request-1",
        target=target,
        fixture=cases[0],
    )
    with pytest.raises(ValueError, match="array length"):
        job.assert_result(
            result(target, {"tokens": [1]}),
            run_id="request-1",
            target=target,
            fixture=cases[0],
        )


@pytest.mark.parametrize("mode,runtime", [("host", "cpp"), ("image", "python")])
def test_controller_rejects_fixture_opt_in_outside_native_image_scope(
    tmp_path, monkeypatch, mode, runtime
):
    _, args = controller_args(
        tmp_path,
        monkeypatch,
        "--mode",
        mode,
        "--image-runtime",
        runtime,
        "--application-fixtures",
    )
    with pytest.raises(ValueError, match="native.*image|image.*native"):
        controller.validate_args(args)


def test_controller_forwards_explicit_fixture_opt_in(tmp_path, monkeypatch):
    _, args = controller_args(
        tmp_path, monkeypatch, "--mode", "image", "--application-fixtures"
    )
    controller.validate_args(args)
    command = controller.build_job_command(args, "unit-job")
    remote = shlex.split(command[command.index("--command") + 1])
    assert remote.count("--application-fixtures") == 1


@pytest.mark.parametrize("mutation", [None, "missing", "hash", "index", "unsolicited"])
def test_controller_requires_bound_fixture_scope_and_request_evidence(
    tmp_path, monkeypatch, mutation
):
    target = selected(sealed_bundle(tmp_path))
    _, args = controller_args(
        tmp_path, monkeypatch, "--mode", "image", "--application-fixtures"
    )
    args.bundle_root = str(tmp_path / "bundle")
    args.image = IMAGE
    names = job.expected_cases([target])
    cases = [{"name": name, "status": "passed", "details": {}} for name in names]
    cases[0]["details"]["runner_sha256"] = job.sha256(job.__file__)
    for operation, index in [("first_request", 0), ("repeat_request", 1)]:
        next(c for c in cases if c["name"] == target["name"] + "." + operation)[
            "details"
        ]["application_fixture"] = {
            "index": index,
            "name": CASES[index]["name"],
            "sha256": target["application_fixtures"]["sha256"],
        }
    report = {
        "schema_version": 1,
        "status": "passed",
        "run_id": args.run_id,
        "source_sha": SOURCE,
        "image_digest": IMAGE,
        "qa_scope": "image-http",
        "targets": [target],
        "expected_cases": names,
        "cases": cases,
    }
    if mutation == "missing":
        target.pop("application_fixtures")
    elif mutation == "hash":
        cases[3]["details"]["application_fixture"]["sha256"] = "f" * 64
    elif mutation == "index":
        cases[4]["details"]["application_fixture"]["index"] = 0
    elif mutation == "unsolicited":
        args.application_fixtures = False
    if mutation is None:
        controller.validate_report(report, args=args)
    else:
        with pytest.raises(ValueError, match="fixture"):
            controller.validate_report(report, args=args)


def release_selected(root):
    target = job._bundle_target(
        root,
        source_sha=SOURCE,
        runtime_identity=job._expected_identity(IMAGE),
        release_acceptance=True,
    )
    target["release_acceptance"] = {
        "schema_version": 1,
        "fixture_indices": list(range(target["application_fixtures"]["case_count"])),
    }
    return target


@pytest.mark.parametrize("operation_key", ["operations", "operation"])
def test_promotable_fixture_posts_to_declared_default_operation(
    tmp_path, operation_key
):
    target = release_selected(
        sealed_bundle(
            tmp_path,
            ingress={
                "content_type": "application/json",
                operation_key: {"default": "predict", "allowed": ["predict"]},
            },
        )
    )
    posted = []

    class HTTP:
        def request(self, path, *, body=None):
            if body is not None:
                posted.append(path)
                return 202, {
                    "run_id": "request-1",
                    "workflow": target["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                }
            if path.endswith("/status"):
                return 200, {"status": "succeeded"}
            return 200, result(target, CASES[0]["expected"])

    job.exercise_request(
        HTTP(),
        target,
        fixture=job.application_fixture_cases(target)[0],
        fixture_index=0,
        output=tmp_path / "request",
        timeout=1,
    )
    assert posted == ["/v1/infer/my-project/predict"]


def test_promotable_inventory_executes_every_fixture_and_fault_gate(tmp_path):
    root = sealed_bundle(tmp_path, cases=CASES + [{**CASES[0], "name": "third"}])
    target = release_selected(root)
    names = job.expected_cases([target])
    for operation in (
        "fixture.0",
        "fixture.1",
        "fixture.2",
        "overlap",
        "worker_recovery",
        "request_deadline",
        "lease_cancellation",
        "startup_gpu_rejection",
        "startup_model_rejection",
    ):
        assert target["name"] + "." + operation in names
    assert target["name"] + ".first_request" not in names


@pytest.mark.parametrize("alias", ["features", "features.v1"])
def test_promotable_sealed_files_are_checked_and_sent_as_multipart(tmp_path, alias):
    root = sealed_bundle(tmp_path)
    payload = root / "fixtures/inputs/0/features/input.bin"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"golden-input")
    cases = copy.deepcopy(CASES)
    cases[0]["inputs"] = {alias: payload.relative_to(root).as_posix()}
    cases[0]["files"] = {
        alias: {
            "source": cases[0]["inputs"][alias],
            "media_type": "application/octet-stream",
            "sha256": job.sha256(payload),
            "size_bytes": payload.stat().st_size,
        }
    }
    job.write_json(root / "fixtures/cases.json", {"schema_version": 1, "cases": cases})
    descriptor = json.loads((root / "bundle.json").read_text())
    job.write_json(
        root / "configuration/plugin/my-project/plugin.yaml",
        {"ingress": {"content_types": ["application/json", "multipart/form-data"]}},
    )
    descriptor["input_files"] = {
        alias: {
            "required": False,
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 64,
        }
    }
    job.write_json(root / "bundle.json", descriptor)
    reseal(root)
    target = release_selected(root)
    target["input_files"] = {
        alias: {
            "required": False,
            "media_types": ["application/octet-stream"],
            "max_size_bytes": 64,
        }
    }
    target["ingress"] = "multipart/form-data"
    assert job.application_fixture_cases(target)[0]["files"][alias][
        "sha256"
    ] == job.sha256(payload)
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="fixture|input"):
        job.application_fixture_cases(target)


def test_multipart_encoder_preserves_dotted_fixture_alias(tmp_path, monkeypatch):
    from email.parser import BytesParser
    from email.policy import default

    payload = tmp_path / "input.bin"
    payload.write_bytes(b"golden-input")
    posted = []

    class Response:
        code = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _maximum):
            return b'{"run_id":"request-1"}'

    def urlopen(request, *, timeout):
        posted.append(request)
        return Response()

    monkeypatch.setattr(job, "urlopen", urlopen)
    status, _ = job.HTTP("http://127.0.0.1:8000").request(
        "/v1/infer/my-project/run",
        body={"parameters": CASES[0]["request"]},
        files={"features.v1": {"path": payload, "media_type": "application/x-custom"}},
    )
    assert status == 202
    request = posted[0]
    envelope = BytesParser(policy=default).parsebytes(
        ("Content-Type: " + request.get_header("Content-type") + "\r\n\r\n").encode()
        + request.data
    )
    parts = {
        part.get_param("name", header="content-disposition"): part
        for part in envelope.iter_parts()
    }
    assert set(parts) == {"parameters", "features.v1"}
    assert parts["features.v1"].get_payload(decode=True) == payload.read_bytes()
    assert parts["features.v1"].get_content_type() == "application/x-custom"


def test_promotable_artifact_requires_golden_identity_and_actual_download(tmp_path):
    target = release_selected(sealed_bundle(tmp_path))
    expected = b"output-bytes"
    identity = {
        "sha256": __import__("hashlib").sha256(expected).hexdigest(),
        "size_bytes": len(expected),
    }
    fixture = {**CASES[0], "expected_artifacts": {"answer.bin": identity}}
    report = result(target, fixture["expected"])
    report["execution"]["outputs"] = [
        {
            "name": "answer.bin",
            **identity,
            "storage_path": "/outputs/request-1/attempt-1/answer.bin",
        }
    ]

    class HTTP:
        def request(self, path, *, body=None, files=None):
            if body is not None:
                return 202, {
                    "run_id": "request-1",
                    "workflow": target["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                }
            if path.endswith("/status"):
                return 200, {"status": "succeeded"}
            return 200, report

        def download(self, path, destination, *, expected):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"bad-bytes")
            return {
                "sha256": job.sha256(destination),
                "size_bytes": destination.stat().st_size,
            }

    with pytest.raises(ValueError, match="artifact|download"):
        job.exercise_request(
            HTTP(),
            target,
            fixture=fixture,
            fixture_index=0,
            output=tmp_path / "request",
            timeout=1,
        )


def test_promotable_request_does_not_accept_unasserted_nonempty_artifacts(tmp_path):
    target = release_selected(sealed_bundle(tmp_path))
    report = result(target, CASES[0]["expected"])
    report["execution"]["outputs"] = [
        {"name": "unasserted", "sha256": "f" * 64, "size_bytes": 3}
    ]
    with pytest.raises(ValueError, match="artifact"):
        job.assert_result(report, run_id="request-1", target=target, fixture=CASES[0])


@pytest.mark.parametrize(
    "content_types", [["application/json"], ["multipart/form-data"]]
)
@pytest.mark.parametrize("explicit_empty_files", [False, True])
def test_expanded_runner_replays_all_cases_and_observes_overlap(
    tmp_path, monkeypatch, content_types, explicit_empty_files
):
    authored = CASES + [{**CASES[0], "name": "third-golden"}]
    if explicit_empty_files:
        authored = [{**case, "files": {}} for case in authored]
    target = release_selected(
        sealed_bundle(
            tmp_path, cases=authored, ingress={"content_types": content_types}
        )
    )
    admitted = {}
    schema_rejections = []
    lock = __import__("threading").Lock()
    details = {}

    class HTTP:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, path, *, body=None, files=None):
            if path == "/readyz":
                return 200, {
                    "ready": True,
                    "native": {
                        "ready": True,
                        "workflow_id": target["workflow"],
                        "bundle_digest": target["bundle_digest"],
                        "runtime_identity": target["runtime_identity"],
                        "worker_instance_id": "resident",
                        "deployment_id": "deployment-1",
                    },
                }
            if "unknown-" in path:
                return 404, {}
            if body is not None:
                content_type = (
                    "multipart/form-data" if files is not None else "application/json"
                )
                if content_type not in content_types:
                    return 415, {}
                if not isinstance(body["parameters"], dict):
                    schema_rejections.append(content_type)
                    return 422, {}
                fixture = next(
                    case for case in authored if case["request"] == body["parameters"]
                )
                with lock:
                    run_id = "request-" + str(len(admitted))
                    admitted[run_id] = fixture
                return 202, {
                    "run_id": run_id,
                    "workflow": target["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                }
            if path.endswith("/status"):
                return 200, {"status": "succeeded"}
            run_id = path.split("/")[-2]
            report = result(target, admitted[run_id]["expected"])
            report["execution"].update(run_id=run_id, outputs=[])
            report["execution"]["native"]["attempt"] = "attempt-" + run_id
            return 200, report

    monkeypatch.setattr(job, "HTTP", HTTP)
    monkeypatch.setattr(job, "_port", lambda: 12345)
    monkeypatch.setattr(job, "image_launch_plan", lambda *_args: (Path("/unused"), {}))
    monkeypatch.setattr(
        job, "reject_native_startup", lambda *_args, **kwargs: {"kind": kwargs["kind"]}
    )
    monkeypatch.setattr(
        job, "isolated_worker_fault", lambda *_args, **kwargs: {"kind": kwargs["kind"]}
    )
    monkeypatch.setattr(
        job, "recover_worker", lambda *_args, **_kwargs: {"recovered": True}
    )

    def case(name, function):
        details[name] = function()
        return details[name]

    context = SimpleNamespace(
        root=tmp_path,
        environment={},
        case=case,
        args=SimpleNamespace(
            base_url="https://unit.invalid",
            deployment_id="deployment-1",
            request_timeout=2,
            startup_timeout=1,
        ),
    )
    job.run_target(context, target)
    assert schema_rejections == [content_types[0]]
    assert set(details) == set(job.expected_cases([target])) - {"environment"}
    for index, fixture in enumerate(authored):
        assert (
            details[target["name"] + ".fixture." + str(index)]["application_fixture"][
                "name"
            ]
            == fixture["name"]
        )
    overlap = details[target["name"] + ".overlap"]
    assert overlap["both_admitted_before_poll"] is True
    assert len({item["native_attempt"] for item in overlap["requests"]}) == 2
    assert len(admitted) == len(authored) + 2


@pytest.mark.parametrize("kind", ["deadline", "lease_cancellation"])
@pytest.mark.parametrize(
    "mutation", [None, "late-success", "duplicate-release", "pending", "wrong-error"]
)
def test_internal_fault_completion_rejects_late_or_unfenced_publication(kind, mutation):
    releases = [{"status": "failed"}] if kind == "deadline" else []
    results = (
        [
            {
                "status": "failed",
                "execution": {
                    "status": "failed",
                    "error": "native request deadline exceeded",
                },
            }
        ]
        if kind == "deadline"
        else []
    )
    pending = int(kind != "deadline")
    stderr = "message lease ownership lost"
    if mutation == "late-success":
        results.append({"status": "completed", "execution": {"status": "succeeded"}})
    elif mutation == "duplicate-release":
        releases.append({"status": "failed"})
    elif mutation == "pending":
        pending = int(kind == "deadline")
    elif mutation == "wrong-error":
        if results:
            results[0]["execution"]["error"] = "unrelated failure"
        stderr = "unrelated failure"
    if mutation:
        with pytest.raises(ValueError):
            job.validate_fault_completion(
                kind, releases=releases, results=results, pending=pending, stderr=stderr
            )
    else:
        assert job.validate_fault_completion(
            kind, releases=releases, results=results, pending=pending, stderr=stderr
        )["no_late_success"]


def test_release_mode_is_explicit_owned_native_image_scope(tmp_path, monkeypatch):
    _, args = controller_args(
        tmp_path,
        monkeypatch,
        "--mode",
        "image",
        "--application-fixtures",
        "--release-acceptance",
    )
    controller.validate_args(args)
    command = controller.build_job_command(args, "unit-job")
    remote = shlex.split(command[command.index("--command") + 1])
    assert "--release-acceptance" in remote
    args.application_fixtures = False
    with pytest.raises(ValueError, match="Release acceptance"):
        controller.validate_args(args)


@pytest.fixture
def native_probe_plan(tmp_path, monkeypatch):
    for key in (
        "RELEASE_IMAGE_DIGEST",
        "PNMS_IMAGE_CONFIG_DIGEST",
        "PNMS_BUNDLE_DIGEST",
        "NATIVE_DEPLOYMENT_OPERATION_ID",
        "NATIVE_WORKFLOW_ONLY",
    ):
        monkeypatch.setenv(key, "inherited-unrelated-value")
    monkeypatch.setenv("PNMS_QA_PASSTHROUGH", "retained")
    config = tmp_path / "original-deployment.json"
    job.write_json(
        config,
        {
            "target": {"compilation": {"gpu_product": "supported-product"}},
            "model_assets": [{"sha256": "f" * 64}],
            "input_artifacts": {},
        },
    )
    capture = tmp_path / "child-environment.json"
    executable = python_executable(
        tmp_path / "native-worker",
        "import json, os, pathlib, sys\n"
        "identities = ('RELEASE_IMAGE_DIGEST', 'PNMS_IMAGE_CONFIG_DIGEST', 'PNMS_BUNDLE_DIGEST')\n"
        "keys = (*identities, 'NATIVE_DEPLOYMENT_OPERATION_ID', 'NATIVE_WORKFLOW_ONLY', 'PNMS_QA_PASSTHROUGH')\n"
        "pathlib.Path(os.environ['QA_PROBE_CAPTURE']).write_text(json.dumps({key: os.environ[key] for key in keys if key in os.environ}))\n"
        "if sum(bool(os.environ.get(name)) for name in identities) != 1:\n"
        "    print('Error: set exactly one of RELEASE_IMAGE_DIGEST, PNMS_IMAGE_CONFIG_DIGEST or PNMS_BUNDLE_DIGEST', file=sys.stderr)\n"
        "else:\n"
        "    config = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "    error = 'GPU product mismatch' if config['target']['compilation']['gpu_product'].startswith('qa-unsupported-') else 'Model SHA256 integrity mismatch'\n"
        "    print(error, file=sys.stderr)\n"
        "raise SystemExit(1)\n",
    )
    client = SimpleNamespace(scan_iter=lambda **_kwargs: [])
    monkeypatch.setattr(job, "_redis", lambda _plan: client)
    monkeypatch.setattr(
        job,
        "_registry",
        lambda _client: {
            "owned": {
                "deployment_id": "owned-deployment",
                "pid": 101,
                "worker_instance_id": "resident-worker",
            }
        },
    )
    plan = {
        "deployment_id": "owned-deployment",
        "processes": [
            {
                "name": "native-worker",
                "argv": [str(executable), str(config)],
                "env": {
                    "RELEASE_IMAGE_DIGEST": IMAGE,
                    "QA_PROBE_CAPTURE": str(capture),
                },
            }
        ],
    }
    return plan, capture


@pytest.mark.parametrize("kind", ["gpu", "model"])
@pytest.mark.parametrize(
    "identity_key,identity",
    [
        ("RELEASE_IMAGE_DIGEST", IMAGE),
        ("PNMS_IMAGE_CONFIG_DIGEST", "sha256:" + "d" * 64),
        ("PNMS_BUNDLE_DIGEST", "e" * 64),
    ],
)
def test_startup_negative_uses_launch_plan_identity_under_contaminated_parent(
    native_probe_plan, tmp_path, kind, identity_key, identity
):
    plan, capture = native_probe_plan
    environment = plan["processes"][0]["env"]
    environment.pop("RELEASE_IMAGE_DIGEST")
    environment[identity_key] = identity

    evidence = job.reject_native_startup(plan, {}, kind=kind, output=tmp_path)

    child_environment = json.loads(capture.read_text())
    assert {
        key: child_environment[key]
        for key in (
            "RELEASE_IMAGE_DIGEST",
            "PNMS_IMAGE_CONFIG_DIGEST",
            "PNMS_BUNDLE_DIGEST",
        )
        if key in child_environment
    } == {identity_key: identity}
    assert "NATIVE_DEPLOYMENT_OPERATION_ID" not in child_environment
    assert "NATIVE_WORKFLOW_ONLY" not in child_environment
    assert child_environment["PNMS_QA_PASSTHROUGH"] == "retained"
    assert evidence["returncode"] == 1
    assert evidence["worker_before"] == evidence["worker_after"]
    assert ("GPU product" if kind == "gpu" else "SHA256 integrity") in evidence[
        "stderr"
    ]


def test_isolated_fault_uses_plan_identity_without_inherited_runtime_flags(
    native_probe_plan, tmp_path, monkeypatch
):
    import http.server
    import ssl

    plan, _capture = native_probe_plan
    captured = {}

    class Server:
        server_port = 8443
        socket = object()

        def __init__(self, *_args):
            pass

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    class TLS:
        def __init__(self, *_args):
            pass

        def load_cert_chain(self, *_args):
            pass

        def wrap_socket(self, connection, **_kwargs):
            return connection

    class LaunchCaptured(Exception):
        pass

    def launch(_argv, **kwargs):
        keys = (
            "RELEASE_IMAGE_DIGEST",
            "PNMS_IMAGE_CONFIG_DIGEST",
            "PNMS_BUNDLE_DIGEST",
            "NATIVE_DEPLOYMENT_OPERATION_ID",
            "NATIVE_WORKFLOW_ONLY",
            "PNMS_QA_PASSTHROUGH",
            "NATIVE_DEPLOYMENT_ID",
            "REDIS_STREAM_PREFIX",
            "SSL_CERT_FILE",
            "NO_PROXY",
        )
        captured.update(
            {key: kwargs["env"][key] for key in keys if key in kwargs["env"]}
        )
        raise LaunchCaptured

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(ssl, "SSLContext", TLS)
    monkeypatch.setattr(job.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(job.subprocess, "Popen", launch)
    with pytest.raises(LaunchCaptured):
        job.isolated_worker_fault(
            plan,
            {"bundle_digest": "f" * 64},
            CASES[0],
            kind="deadline",
            output=tmp_path / "fault",
            timeout=1,
        )

    assert captured["RELEASE_IMAGE_DIGEST"] == IMAGE
    assert "PNMS_IMAGE_CONFIG_DIGEST" not in captured
    assert "PNMS_BUNDLE_DIGEST" not in captured
    assert "NATIVE_DEPLOYMENT_OPERATION_ID" not in captured
    assert "NATIVE_WORKFLOW_ONLY" not in captured
    assert captured["PNMS_QA_PASSTHROUGH"] == "retained"
    assert captured["NATIVE_DEPLOYMENT_ID"].startswith("fault-")
    assert captured["REDIS_STREAM_PREFIX"].startswith("pnms-qa-")
    assert captured["SSL_CERT_FILE"] == str(tmp_path / "fault/ca.pem")
    assert captured["NO_PROXY"] == "localhost,127.0.0.1"


def test_isolated_fault_queues_the_compiled_workflow_identity(
    native_probe_plan, tmp_path, monkeypatch
):
    import http.server
    import ssl

    plan, _capture = native_probe_plan
    configuration = Path(plan["processes"][0]["argv"][1])
    config = json.loads(configuration.read_text())
    config.update(
        workflow_id="compiled-workflow",
        executor_class="native.compiled",
        manifest_version="1.2.3",
    )
    job.write_json(configuration, config)
    captured = {}

    class Submitted(Exception):
        pass

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 1

        def wait(self, **_kwargs):
            return self.returncode

    class Server:
        server_port = 8443
        socket = object()

        def __init__(self, *_args):
            pass

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    class TLS:
        def __init__(self, *_args):
            pass

        def load_cert_chain(self, *_args):
            pass

        def wrap_socket(self, connection, **_kwargs):
            return connection

    def launch(_argv, **kwargs):
        captured["environment"] = kwargs["env"]
        return Child()

    def enqueue(stream, fields):
        captured.update(stream=stream, fields=fields)
        raise Submitted

    client = SimpleNamespace(scan_iter=lambda **_kwargs: [], xadd=enqueue)
    monkeypatch.setattr(job, "_redis", lambda _plan: client)
    monkeypatch.setattr(
        job,
        "_registry",
        lambda _client: {
            "worker": {
                "deployment_id": captured["environment"]["NATIVE_DEPLOYMENT_ID"],
                "status": "available",
                "stream": "owned-execute-stream",
            },
        },
    )
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(ssl, "SSLContext", TLS)
    monkeypatch.setattr(job.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(job.subprocess, "Popen", launch)
    with pytest.raises(Submitted):
        job.isolated_worker_fault(
            plan,
            {"bundle_digest": "f" * 64},
            CASES[0],
            kind="deadline",
            output=tmp_path / "fault",
            timeout=1,
        )
    payload = json.loads(captured["fields"]["payload"])
    assert payload.get("workflow_id") == "compiled-workflow"
    assert payload["runtime"] == {"kind": "cpp", "executor_class": "native.compiled"}
    assert payload["parameters"] == CASES[0]["request"]
    assert payload["manifest_version"] == "1.2.3"
    assert captured["fields"]["stage"] == "execute"
    assert captured["stream"] == "owned-execute-stream"
