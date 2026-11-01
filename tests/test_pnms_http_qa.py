"""Unit evidence contracts for the finite Lepton HTTP runner (no live HTTP)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shlex
import signal
import subprocess
import struct
import sys
from types import SimpleNamespace

import pytest

from qa.native_inference import pnms_http_job as job


TARGET = {
    "name": "native.aoti.tensor",
    "runtime": "cpp",
    "workflow": "aoti-tensor",
    "source_sha": "a" * 40,
    "bundle_digest": "b" * 64,
    "model_count": 1,
    "runtime_identity": {"kind": "bundle", "value": "b" * 64},
}


@pytest.fixture
def isolated_fault_probe(tmp_path, monkeypatch):
    import http.server
    import ssl

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "application-cli/src")
    )
    configuration = tmp_path / "original-deployment.json"
    job.write_json(
        configuration,
        {
            "workflow_id": TARGET["workflow"],
            "executor_class": "native.compiled",
            "input_artifacts": {},
            "ingress_root": str(tmp_path / "ingress"),
        },
    )
    state = SimpleNamespace(
        retired_change=lambda worker: None,
        registry_change=lambda registry: registry,
        owner_lease=None,
        returncode=1,
        environment={},
        lease_queries=[],
    )

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 1

        def wait(self, **_kwargs):
            return self.returncode

    child = Child()

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

    class Event:
        def set(self):
            pass

        def wait(self, *_args):
            return True

    class Thread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def join(self, **_kwargs):
            pass

    def launch(_argv, **kwargs):
        state.environment = kwargs["env"]
        kwargs["stdout"].write("message lease ownership lost\n")
        return child

    def enqueue(_stream, _fields):
        child.returncode = state.returncode
        return "1-0"

    def registry(_client):
        worker = {
            "deployment_id": state.environment["NATIVE_DEPLOYMENT_ID"],
            "worker_instance_id": "original-worker",
            "worker_slot": "0",
            "pid": 101,
            "workflow_id": TARGET["workflow"],
            "executor_class": "native.compiled",
            "device_uuid": "GPU-owned",
            "device_index": 0,
            "runtime_identity": TARGET["runtime_identity"],
            "bundle_digest": TARGET["bundle_digest"],
            "deployment_sha256": "d" * 64,
            "stream": "owned-execute-stream",
            "status": "available",
        }
        if child.poll() is not None:
            worker.update(
                status="warming",
                model_cache={
                    "schema_version": 1,
                    "scope": "process",
                    "entries": [],
                    "total_entries": 0,
                    "warmup": {
                        "workflow_id": TARGET["workflow"],
                        "status": "warming",
                        "reason": "restart_pending",
                    },
                },
            )
            state.retired_change(worker)
            return state.registry_change({"original-slot": worker})
        return {"original-slot": worker}

    def owner(key):
        state.lease_queries.append(key)
        return state.owner_lease

    def events(_client, stream, _run):
        if state.kind == "lease_cancellation":
            return []
        if stream.endswith("release"):
            return [{"status": "failed"}]
        return [
            {
                "status": "failed",
                "execution": {
                    "status": "failed",
                    "error": "native request deadline exceeded",
                },
            }
        ]

    client = SimpleNamespace(
        scan_iter=lambda **_kwargs: [],
        xadd=enqueue,
        xinfo_groups=lambda _stream: [{"name": "group"}],
        xpending_range=lambda *_args: [{"consumer": "original"}],
        xpending=lambda *_args: {"pending": int(state.kind != "deadline")},
        xclaim=lambda *_args, **_kwargs: None,
        get=owner,
    )
    monkeypatch.setattr(job, "_redis", lambda _plan: client)
    monkeypatch.setattr(job, "_registry", registry)
    monkeypatch.setattr(job, "_event_payloads", events)
    monkeypatch.setattr(job, "fixture_files", lambda *_args: {})
    monkeypatch.setattr(http.server, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(ssl, "SSLContext", TLS)
    monkeypatch.setattr(job, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    monkeypatch.setattr(job.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(job.subprocess, "Popen", launch)
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    plan = {
        "processes": [
            {
                "name": "native-worker",
                "argv": ["mock-native-worker", str(configuration)],
                "env": {"PNMS_BUNDLE_DIGEST": TARGET["bundle_digest"]},
            }
        ]
    }

    def run(kind):
        state.kind = kind
        return job.isolated_worker_fault(
            plan,
            TARGET,
            {"request": {}, "files": {}},
            kind=kind,
            output=tmp_path / "fault",
            timeout=1,
        )

    return run, state, tmp_path / "fault"


@pytest.mark.parametrize("kind", ["deadline", "lease_cancellation"])
def test_isolated_fault_requires_retained_warming_slot_and_released_owner(
    isolated_fault_probe, kind
):
    run, state, output = isolated_fault_probe
    evidence = run(kind)
    assert evidence["returncode"] == 1
    assert evidence["no_late_success"] is True
    snapshot = json.loads((output / "fault-registry.json").read_text())
    assert snapshot["registry_field"] == "original-slot"
    assert snapshot["selected_registry"]["original-slot"]["status"] == "warming"
    assert snapshot["owner_lease"] is None
    assert state.lease_queries == [
        "native:worker-lease:" + job.hashlib.sha256(b"original-slot").hexdigest()
    ]


@pytest.mark.parametrize("kind", ["deadline", "lease_cancellation"])
@pytest.mark.parametrize("inventory", ["absent", "different-slot", "duplicate"])
def test_isolated_fault_rejects_lost_or_changed_durable_slot(
    isolated_fault_probe, kind, inventory
):
    run, state, output = isolated_fault_probe

    def change(registry):
        worker = registry["original-slot"]
        if inventory == "absent":
            return {}
        if inventory == "different-slot":
            return {"replacement-slot": worker}
        return {**registry, "duplicate-slot": worker}

    state.registry_change = change
    with pytest.raises(ValueError):
        run(kind)
    assert (output / "fault-registry.json").is_file()


@pytest.mark.parametrize("kind", ["deadline", "lease_cancellation"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "available"),
        ("status", "busy"),
        ("worker_instance_id", "replacement-worker"),
        ("pid", 102),
        ("worker_slot", "1"),
        ("deployment_id", "foreign"),
        ("stream", "foreign"),
        ("workflow_id", "foreign"),
        ("executor_class", "foreign"),
        ("device_uuid", "foreign"),
        ("device_index", 1),
        ("runtime_identity", {"kind": "bundle", "value": "e" * 64}),
        ("bundle_digest", "e" * 64),
        ("deployment_sha256", "e" * 64),
        ("model_cache.entries", [{"status": "loaded"}]),
        ("model_cache.total_entries", 1),
        ("model_cache.warmup.status", "succeeded"),
        ("model_cache.warmup.reason", "startup"),
        ("model_cache.warmup.workflow_id", "foreign"),
    ],
)
def test_isolated_fault_rejects_admission_or_changed_identity(
    isolated_fault_probe, kind, field, value
):
    run, state, output = isolated_fault_probe

    def change(worker):
        keys = field.split(".")
        for key in keys[:-1]:
            worker = worker[key]
        worker[keys[-1]] = value

    state.retired_change = change
    with pytest.raises(ValueError):
        run(kind)
    snapshot = json.loads((output / "fault-registry.json").read_text())
    assert "original-slot" in snapshot["selected_registry"]


@pytest.mark.parametrize("kind", ["deadline", "lease_cancellation"])
@pytest.mark.parametrize("failure", ["live-lease", "successful-exit", "completion"])
def test_isolated_fault_saves_registry_before_other_fault_assertions(
    isolated_fault_probe, monkeypatch, kind, failure
):
    run, state, output = isolated_fault_probe
    if failure == "live-lease":
        state.owner_lease = "original-worker"
    elif failure == "successful-exit":
        state.returncode = 0
    else:
        monkeypatch.setattr(
            job, "_event_payloads", lambda *_args: [{"status": "succeeded"}]
        )
    with pytest.raises(ValueError):
        run(kind)
    snapshot = json.loads((output / "fault-registry.json").read_text())
    assert snapshot["owner_lease"] == state.owner_lease


@pytest.mark.parametrize("lost_slot", [True, False])
def test_worker_recovery_requires_durable_slot_throughout_restart(
    tmp_path, monkeypatch, lost_slot
):
    before = {
        "deployment_id": "owned-deployment",
        "worker_instance_id": "original-worker",
        "pid": 101,
        "stream": "owned-execute-stream",
        "status": "available",
    }
    after = {**before, "worker_instance_id": "replacement-worker", "pid": 102}
    restarting = {**before, "status": "warming"}
    snapshots = iter(
        [
            {"original-slot": before},
            {} if lost_slot else {"original-slot": restarting},
            {"original-slot": after},
        ]
    )
    monkeypatch.setattr(job, "_redis", lambda _plan: object())
    monkeypatch.setattr(job, "_registry", lambda _client: next(snapshots))
    monkeypatch.setattr(
        job,
        "_owned_processes",
        lambda _pid: {
            "native-worker": [{"pid": 101, "start": 1}],
            "redis-server": [{"pid": 99, "start": 1}],
        },
    )
    monkeypatch.setattr(job.os, "readlink", lambda _path: "/mock/native-worker")
    # Model the Linux runner's signal capability on every CPU test host.
    monkeypatch.setattr(job.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(job.os, "kill", lambda *_args: None)
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        job, "exercise_request", lambda *_args, **_kwargs: {"run_id": "replay"}
    )
    report = ready()
    report["native"]["worker_instance_id"] = "replacement-worker"
    http = SimpleNamespace(request=lambda _path: (200, report))

    def recover():
        return job.recover_worker(
            SimpleNamespace(pid=100, poll=lambda: None),
            http,
            TARGET,
            ready(),
            {"deployment_id": "owned-deployment"},
            {},
            output=tmp_path,
            startup_timeout=1,
            request_timeout=1,
        )

    if lost_slot:
        with pytest.raises(ValueError, match="durable.*slot"):
            recover()
    else:
        assert recover()["worker_after"]["worker_instance_id"] == "replacement-worker"


def ready():
    return {
        "ready": True,
        "native": {
            "ready": True,
            "workflow_id": TARGET["workflow"],
            "bundle_digest": TARGET["bundle_digest"],
            "deployment_id": "owned-deployment",
            "runtime_identity": TARGET["runtime_identity"],
            "worker_instance_id": "worker-1",
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("bundle_digest", "foreign"),
        ("deployment_id", "foreign"),
        ("workflow_id", "foreign"),
        ("runtime_identity", {"kind": "registry-image", "value": "foreign"}),
    ],
)
def test_readiness_rejects_other_bundle_deployment_workflow_or_identity(field, value):
    report = copy.deepcopy(ready())
    report["native"][field] = value
    with pytest.raises(ValueError):
        job.assert_readiness(report, TARGET, deployment_id="owned-deployment")


def result():
    return {
        "payload": {"values": [3, -3, 1.5, 9]},
        "execution": {
            "run_id": "run-1",
            "status": "succeeded",
            "native": {
                "runtime": "cpp",
                "source_sha": TARGET["source_sha"],
                "runtime_identity": TARGET["runtime_identity"],
                "metadata": {"session_load_count": 1},
            },
        },
    }


@pytest.mark.parametrize(
    "mutation", ["source", "run-id", "value", "nan", "session", "identity"]
)
def test_result_requires_independent_numerical_and_execution_evidence(mutation):
    report = copy.deepcopy(result())
    if mutation == "source":
        report["execution"]["native"]["source_sha"] = "f" * 40
    elif mutation == "run-id":
        report["execution"]["run_id"] = "another-run"
    elif mutation == "value":
        report["payload"]["values"][0] = 99
    elif mutation == "nan":
        report["payload"]["values"][0] = float("nan")
    elif mutation == "session":
        report["execution"]["native"]["metadata"]["session_load_count"] = 2
    else:
        report["execution"]["native"]["runtime_identity"] = {
            "kind": "image-config",
            "value": "sha256:" + "b" * 64,
        }
    with pytest.raises(ValueError):
        job.assert_result(
            report, run_id="run-1", target=TARGET, values=[1, -2, 0.25, 4]
        )


def summary():
    return {
        "schema_version": 1,
        "run_id": "unit-run",
        "source_sha": "a" * 40,
        "image_digest": "registry/qa@sha256:" + "c" * 64,
        "status": "passed",
        "qa_scope": "host-http",
        "targets": [TARGET],
        "expected_cases": ["environment"]
        + [TARGET["name"] + "." + op for op in job.OPERATIONS],
        "cases": [
            {"name": name, "status": "passed"}
            for name in ["environment"]
            + [TARGET["name"] + "." + op for op in job.OPERATIONS]
        ],
    }


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "skipped", "foreign-image", "foreign-bundle"]
)
def test_http_summary_cannot_pass_with_incomplete_or_foreign_evidence(mutation):
    report = copy.deepcopy(summary())
    if mutation == "missing":
        report["cases"].pop()
    elif mutation == "duplicate":
        report["cases"].append(report["cases"][0])
    elif mutation == "skipped":
        report["cases"][0]["status"] = "skipped"
    elif mutation == "foreign-image":
        report["image_digest"] = "registry/other@sha256:" + "d" * 64
    else:
        report["targets"][0]["bundle_digest"] = "f" * 64
    with pytest.raises(ValueError):
        job.validate_summary(
            report,
            run_id="unit-run",
            source_sha="a" * 40,
            image_digest="registry/qa@sha256:" + "c" * 64,
            targets=[TARGET],
        )


def test_owned_service_shutdown_waits_for_controlled_exit():
    events = []

    class Process:
        pid = 123456789

        def poll(self):
            return None

        def send_signal(self, value):
            events.append(("signal", value))

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 0

    report = job.OwnedService(Process(), timeout=7).close()
    assert events == [("signal", signal.SIGINT), ("wait", 7)]
    assert report["stopped"] is True


def test_good_native_evidence_is_accepted():
    job.assert_readiness(ready(), TARGET, deployment_id="owned-deployment")
    job.assert_result(result(), run_id="run-1", target=TARGET, values=[1, -2, 0.25, 4])
    job.validate_summary(
        summary(),
        run_id="unit-run",
        source_sha="a" * 40,
        image_digest="registry/qa@sha256:" + "c" * 64,
        targets=[TARGET],
    )


def controller_args(tmp_path, monkeypatch, *extra):
    from scripts import run_lepton_pnms_http_qa as controller

    monkeypatch.delenv("LEPTON_WORKSPACE_TOKEN", raising=False)
    args = controller.build_parser().parse_args(
        [
            "--image",
            "registry/qa@sha256:" + "c" * 64,
            "--expected-source-sha",
            "a" * 40,
            "--run-id",
            "unit-run",
            "--artifact-dir",
            str(tmp_path),
            "--workspace-id",
            "unit-workspace",
            "--node-group",
            "unit-group",
            "--pull-secret",
            "unit-secret",
            "--resource-shape",
            "gpu.h100-sxm",
            "--nfs-mount-base",
            "/mnt/shared",
            *extra,
        ]
    )
    return controller, args


def test_http_controller_forwards_verified_local_handoff_without_nested_docker(
    tmp_path, monkeypatch
):
    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--local-summary",
        "/outputs/native-inference/local-run/summary.json",
        "--include-python",
    )
    command = controller.build_job_command(args, "owned-job")
    remote = shlex.split(command[command.index("--command") + 1])
    assert remote[:3] == ["exec", "/opt/nvidia/nvidia_entrypoint.sh", "python3"]
    assert remote[3] == "-B"
    assert remote[4] == "/workspace/qa/native_inference/pnms_http_job.py"
    assert remote[remote.index("--local-summary") + 1] == args.local_summary
    assert remote[remote.index("--mount-root") + 1] == "/outputs"
    assert remote[remote.index("--output") + 1] == "/outputs/native-inference/unit-run"
    assert (
        "--include-python" in remote and "docker" not in remote and "pip" not in remote
    )
    assert command[command.index("--num-workers") + 1] == "1"
    assert command[command.index("--max-failure-retry") + 1] == "0"


def test_owned_service_timeout_kills_its_process_without_using_an_unowned_group(
    monkeypatch,
):
    events = []

    class Process:
        pid = 123456789

        def poll(self):
            return None

        def send_signal(self, value):
            events.append(("signal", value))

        def wait(self, timeout):
            events.append(("wait", timeout))
            if timeout == 7:
                raise subprocess.TimeoutExpired("owned-service", timeout)
            return -9

        def kill(self):
            events.append(("kill", self.pid))

    monkeypatch.setattr(
        job.os,
        "killpg",
        lambda *a: pytest.fail("Unowned process group was targeted"),
        raising=False,
    )
    with pytest.raises(ValueError, match="shutdown deadline"):
        job.OwnedService(Process(), timeout=7).close()
    assert ("kill", Process.pid) in events


def test_public_request_posts_once_while_waiting_for_the_same_run_result(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            (
                202,
                {
                    "workflow": TARGET["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                    "run_id": "run-1",
                },
            ),
            (200, {"status": "succeeded"}),
            (404, {}),
            (200, {"status": "succeeded"}),
            (200, result()),
        ]
    )
    calls = []

    class HTTP:
        def request(self, path, *, body=None):
            calls.append((path, body))
            return next(responses)

    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    evidence = job.exercise_request(
        HTTP(), TARGET, values=[1, -2, 0.25, 4], output=tmp_path, timeout=1
    )
    assert evidence["run_id"] == "run-1"
    assert sum(body is not None for path, body in calls) == 1
    assert [path for path, body in calls if body is None] == [
        "/v1/infer/aoti-tensor/run-1/status",
        "/v1/infer/aoti-tensor/run-1/results",
        "/v1/infer/aoti-tensor/run-1/status",
        "/v1/infer/aoti-tensor/run-1/results",
    ]


def python_provisional():
    return {
        "run_id": "python-run",
        "workflow": "http-python",
        "status": "succeeded",
        "output_path": None,
        "error": None,
        "execution_time_seconds": 0.003,
    }


def python_result():
    return {
        "request": {"parameters": {"value": 17}},
        "execution": {"run_id": "python-run", "status": "succeeded"},
        "payload": {"value": 17, "doubled": 34},
    }


def test_python_http_waits_for_canonical_result_after_legacy_metadata(
    tmp_path, monkeypatch
):
    responses = iter(
        [
            (202, {"workflow": "http-python", "run_id": "python-run"}),
            (200, {"status": "succeeded"}),
            (200, python_provisional()),
            (200, {"status": "succeeded"}),
            (200, python_result()),
        ]
    )
    calls = []

    class HTTP:
        def request(self, path, *, body=None):
            calls.append((path, body))
            return next(responses)

    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    evidence = job.exercise_request(
        HTTP(),
        {"runtime": "python", "workflow": "http-python"},
        values=17,
        output=tmp_path,
        timeout=1,
    )
    assert evidence["run_id"] == "python-run"
    assert sum(body is not None for _, body in calls) == 1
    assert len(calls) == 5
    assert json.loads((tmp_path / "result.json").read_text()) == python_result()
    assert (
        json.loads((tmp_path / "provisional-result.json").read_text())
        == python_provisional()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "provisional-run",
        "provisional-workflow",
        "provisional-status",
        "canonical-run",
        "canonical-status",
        "canonical-value",
    ],
)
def test_python_result_polling_rejects_foreign_metadata_and_wrong_final_results(
    tmp_path, monkeypatch, mutation
):
    provisional, canonical = python_provisional(), python_result()
    if mutation == "provisional-run":
        provisional["run_id"] = "foreign-run"
    elif mutation == "provisional-workflow":
        provisional["workflow"] = "foreign-workflow"
    elif mutation == "provisional-status":
        provisional["status"] = "failed"
    elif mutation == "canonical-run":
        canonical["execution"]["run_id"] = "foreign-run"
    elif mutation == "canonical-status":
        canonical["execution"]["status"] = "failed"
    else:
        canonical["payload"]["doubled"] = 35
    responses = iter(
        [
            (202, {"workflow": "http-python", "run_id": "python-run"}),
            (200, {"status": "succeeded"}),
            (200, provisional),
            (200, {"status": "succeeded"}),
            (200, canonical),
        ]
    )

    class HTTP:
        def request(self, path, *, body=None):
            return next(responses)

    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    with pytest.raises(ValueError):
        job.exercise_request(
            HTTP(),
            {"runtime": "python", "workflow": "http-python"},
            values=17,
            output=tmp_path,
            timeout=1,
        )


def test_python_provisional_metadata_alone_cannot_pass_acceptance(
    tmp_path, monkeypatch
):
    times = iter([0, 0.1, 2])
    posts = []

    class HTTP:
        def request(self, path, *, body=None):
            if body is not None:
                posts.append(body)
                return 202, {"workflow": "http-python", "run_id": "python-run"}
            return 200, python_provisional() if path.endswith("/results") else {
                "status": "succeeded"
            }

    monkeypatch.setattr(job.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="POST was not replayed"):
        job.exercise_request(
            HTTP(),
            {"runtime": "python", "workflow": "http-python"},
            values=17,
            output=tmp_path,
            timeout=1,
        )
    assert len(posts) == 1


@pytest.mark.parametrize(
    "value",
    ["relative/build.json", "/outputs/../secret/build.json", "/elsewhere/build.json"],
)
def test_http_controller_rejects_handoff_inputs_outside_artifact_mount(
    tmp_path, monkeypatch, value
):
    controller, args = controller_args(tmp_path, monkeypatch, "--build-receipt", value)
    with pytest.raises(ValueError, match="mount-target"):
        controller.validate_args(args)


def test_controller_rejects_non_baked_bundle_for_owned_image_mode(
    tmp_path, monkeypatch
):
    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--mode",
        "image",
        "--bundle-root",
        "/outputs/foreign-bundle",
    )
    with pytest.raises(ValueError, match="baked bundle"):
        controller.validate_args(args)


def test_mounted_input_cannot_escape_through_a_symlink(tmp_path):
    mount = tmp_path / "outputs"
    mount.mkdir()
    (mount / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="artifact mount"):
        job.mounted_path(mount / "escape/secret.json", mount)


@pytest.mark.parametrize("bundle", ["../../outside", "/elsewhere/bundle"])
def test_host_receipt_cannot_select_a_bundle_outside_its_build(
    tmp_path, monkeypatch, bundle
):
    receipt = tmp_path / "outputs/app/.pnms/builds/build-id/build.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "status": "complete",
                "runtime": "cpp",
                "checks": {"application": "passed", "model": "passed"},
                "source_sha": "a" * 40,
                "bundle": bundle,
            }
        )
    )
    called = []
    monkeypatch.setattr(job, "_bundle_target", lambda *a, **k: called.append(a) or {})
    with pytest.raises(ValueError, match="relative.*bundle"):
        job.load_host_target(receipt, source_sha="a" * 40)
    assert not called


def test_shutdown_timeout_stops_owned_descendants_without_killing_reused_pids(
    monkeypatch,
):
    events = []

    class Process:
        pid = 123456789

        def poll(self):
            return None

        def wait(self, timeout):
            if timeout == 7:
                raise subprocess.TimeoutExpired("owned-service", timeout)
            return -9

    monkeypatch.setattr(
        job, "_descendants", lambda _: {41: "old-start", 42: "owned-start"}
    )
    monkeypatch.setattr(
        job,
        "_process_record",
        lambda pid: {"start": "reused-start" if pid == 41 else "owned-start"},
    )
    monkeypatch.setattr(
        job.os,
        "killpg",
        lambda pid, sig: events.append(("group", pid, sig)),
        raising=False,
    )
    # Model the Linux runner's signal capability on every CPU test host.
    monkeypatch.setattr(job.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        job.os, "kill", lambda pid, sig: events.append(("pid", pid, sig))
    )
    with pytest.raises(ValueError, match="shutdown deadline"):
        job.OwnedService(Process(), timeout=7, process_group=True).close()
    assert ("pid", 42, signal.SIGKILL) in events
    assert not any(event[:2] == ("pid", 41) for event in events)
    assert ("group", Process.pid, signal.SIGINT) in events
    assert ("group", Process.pid, signal.SIGKILL) in events


def test_owned_descendant_inventory_does_not_adopt_other_services(tmp_path):
    def process(pid, parent, started):
        root = tmp_path / str(pid)
        root.mkdir()
        (root / "stat").write_text(
            f"{pid} (name has spaces) S {parent} " + "0 " * 17 + str(started)
        )

    process(11, 1, 101)
    process(12, 11, 102)
    process(13, 12, 103)
    process(14, 1, 104)
    assert job._descendants(11, tmp_path) == {12: "102", 13: "103"}


def test_release_image_driver_is_standalone_and_does_not_need_the_qa_package(tmp_path):
    runner = tmp_path / "pnms_http_job.py"
    runner.write_text(Path(job.__file__).read_text())
    completed = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        "--mode {host,image}" in completed.stdout
        and "--build-receipt" in completed.stdout
    )


def receipt_report(receipt="/outputs/app/.pnms/builds/build-id/build.json"):
    report = copy.deepcopy(summary())
    report["targets"][0].update(
        build_receipt=receipt,
        build_id="d" * 32,
        bundle="/outputs/app/.pnms/builds/build-id/bundle",
    )
    report["cases"][0]["details"] = {"runner_sha256": job.sha256(job.__file__)}
    return report


def test_http_controller_submits_and_cleans_its_finite_job(tmp_path, monkeypatch):
    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--build-receipt",
        "/outputs/app/.pnms/builds/build-id/build.json",
    )
    calls = []

    def lep(command, **kwargs):
        calls.append(command)
        if command[:2] == ["job", "create"]:
            return 0, "ID: pnms-unit-http-job\n"
        if command[:2] == ["job", "get"]:
            return 0, '{"state":"Completed"}'
        if command[:2] == ["log", "get"]:
            lines = [job.SUMMARY_BEGIN, json.dumps(receipt_report()), job.SUMMARY_END]
            return 0, "\n".join(
                "2026-10-28 16:00:00.000001|" + json.dumps(line) for line in lines
            )
        return 0, ""

    monkeypatch.setattr(controller.lifecycle, "lep", lep)
    assert controller.run(args) == 0
    assert calls[-1] == ["job", "remove", "-i", "pnms-unit-http-job"]
    assert (
        json.loads((tmp_path / args.run_id / "summary.json").read_text())["status"]
        == "passed"
    )


@pytest.mark.parametrize(
    "mutation", ["wrong-receipt", "wrong-runner", "missing-case", "unexpected-python"]
)
def test_http_controller_rejects_complete_looking_but_foreign_evidence(
    tmp_path, monkeypatch, mutation
):
    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--build-receipt",
        "/outputs/app/.pnms/builds/build-id/build.json",
    )
    report = receipt_report()
    if mutation == "wrong-receipt":
        report["targets"][0]["build_receipt"] = "/outputs/other/build.json"
    elif mutation == "wrong-runner":
        report["cases"][0]["details"]["runner_sha256"] = "f" * 64
    elif mutation == "missing-case":
        report["cases"].pop()
    else:
        report["targets"].append(
            {"runtime": "python", "name": "python.generated", "source_sha": "a" * 40}
        )
    with pytest.raises(ValueError):
        controller.validate_report(report, args=args)


def test_http_controller_dry_run_uses_no_cloud_or_gpu(tmp_path, monkeypatch):
    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--build-receipt",
        "/outputs/app/.pnms/builds/build-id/build.json",
    )
    args.dry_run = True
    monkeypatch.setattr(
        controller.lifecycle,
        "lep",
        lambda *a, **k: pytest.fail("Dry run contacted Lepton"),
    )
    assert controller.run(args) == 0
    assert (
        json.loads((tmp_path / args.run_id / "summary.json").read_text())["status"]
        == "dry-run"
    )


def local_report(mount):
    from qa.native_inference import pnms_local_job as local

    report = {
        "schema_version": 1,
        "run_id": "local-run",
        "source_sha": "a" * 40,
        "image_digest": "registry/qa@sha256:" + "c" * 64,
        "profile": "local",
        "status": "passed",
        "qa_scope": "local-workflow",
        "http_qa": "not_run",
        "expected_cases": local.expected_cases(),
        "cases": [],
        "release_candidates": [],
    }
    for name in report["expected_cases"]:
        details = {}
        if name.startswith("native.") and name.endswith(".build"):
            details = {
                "build_id": "d" * 32,
                "bundle_digest": "b" * 64,
                "model_identity": "e" * 64,
                "native_checks": {"verified": True, "case_count": 3},
            }
        elif name.endswith(".held_out"):
            details = {
                "case_count": 3,
                "parity": "passed",
                "input_kind": "raw-fp32-le" if ".files." in name else "json-array-fp32",
            }
        elif (
            name.startswith("native.") and name.endswith(".cache_reuse")
        ) or name == "environment_lock.refresh":
            details = {
                "model_reused": True,
                "workflow_reused": True,
                "environment_lock_digest": "1" * 64,
                "build_id": "d" * 32,
                "bundle_digest": "b" * 64,
            }
            if name == "environment_lock.refresh":
                details.update(
                    drift_rejected=True,
                    prior_success_preserved=True,
                    refresh_passed=True,
                )
        report["cases"].append({"name": name, "status": "passed", "details": details})
    for backend, template, ingress in local.NATIVE_TARGETS:
        selection = {"backend": backend, "template": template, "ingress": ingress}
        project = (
            local.candidate_name(selection).removeprefix("native.").replace(".", "-")
        )
        report["release_candidates"].append(
            {
                **selection,
                "build_id": "d" * 32,
                "bundle_digest": "b" * 64,
                "build_receipt": str(
                    mount
                    / f"native-inference/local-run/relocated/{project}/.pnms/builds/build-id/build.json"
                ),
            }
        )
    return report


def test_http_handoff_resolves_all_five_candidates_from_verified_local_summary(
    tmp_path,
):
    mount = tmp_path / "outputs"
    path = mount / "native-inference/local-run/summary.json"
    job.write_json(path, local_report(mount))
    candidates, evidence = job.local_candidates(
        path,
        mount_root=mount,
        source_sha="a" * 40,
        image_digest="registry/qa@sha256:" + "c" * 64,
    )
    assert len(candidates) == 5 and evidence["sha256"] == job.sha256(path)
    assert {item["backend"] for item in candidates} == {"aoti", "tensorrt"}
    assert {item["template"] for item in candidates} == {"tensor", "custom"}


@pytest.mark.parametrize(
    "mutation",
    ["foreign-source", "foreign-image", "failed", "escaped-receipt", "wrong-run-path"],
)
def test_http_handoff_rejects_unqualified_or_escaped_local_candidates(
    tmp_path, mutation
):
    mount = tmp_path / "outputs"
    path = mount / "native-inference/local-run/summary.json"
    report = local_report(mount)
    if mutation == "foreign-source":
        report["source_sha"] = "f" * 40
    elif mutation == "foreign-image":
        report["image_digest"] = "registry/other@sha256:" + "f" * 64
    elif mutation == "failed":
        report["status"] = "failed"
    elif mutation == "escaped-receipt":
        report["release_candidates"][0]["build_receipt"] = "/etc/secret.json"
    else:
        report["run_id"] = "other-run"
    job.write_json(path, report)
    with pytest.raises(ValueError):
        job.local_candidates(
            path,
            mount_root=mount,
            source_sha="a" * 40,
            image_digest="registry/qa@sha256:" + "c" * 64,
        )


def test_file_target_sends_binary_multipart_and_keeps_independent_parity(tmp_path):
    target = {**TARGET, "name": "native.aoti.files", "ingress": "multipart/form-data"}
    calls = []
    responses = iter(
        [
            (
                202,
                {
                    "workflow": target["workflow"],
                    "pipeline": ["schedule", "execute", "results"],
                    "run_id": "run-1",
                },
            ),
            (200, {"status": "succeeded"}),
            (200, result()),
        ]
    )

    class HTTP:
        def request(self, path, *, body=None, files=None):
            calls.append((path, body, files))
            return next(responses)

    evidence = job.exercise_request(
        HTTP(), target, values=[1, -2, 0.25, 4], output=tmp_path, timeout=1
    )
    files = calls[0][2]
    assert files is not None, "raw target must send a multipart binary file"
    assert Path(files["features"]).read_bytes() == struct.pack("<4f", 1, -2, 0.25, 4)
    assert calls[0][1]["values"]["artifact"] == "features"
    assert evidence["transport"] == "multipart/form-data"
    assert evidence["input_sha256"] == job.sha256(files["features"])
    assert evidence["expected"] == [3, -3, 1.5, 9]


def test_multipart_encoder_preserves_route_fields_and_exact_binary(
    tmp_path, monkeypatch
):
    from email.parser import BytesParser
    from email.policy import default

    raw = tmp_path / "input.f32"
    raw.write_bytes(struct.pack("<4f", 1, -2, 0.25, 4))
    calls = []

    class Response:
        code = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, maximum):
            return b'{"run_id":"run-1"}'

    def urlopen(request, *, timeout):
        calls.append(request)
        return Response()

    monkeypatch.setattr(job, "urlopen", urlopen)
    status, _ = job.HTTP("http://127.0.0.1:8000").request(
        "/v1/infer/raw/run",
        body={"values": {"artifact": "features"}},
        files={"features": raw},
    )
    request = calls[0]
    assert (
        status == 202 and request.full_url == "http://127.0.0.1:8000/v1/infer/raw/run"
    )
    envelope = BytesParser(policy=default).parsebytes(
        ("Content-Type: " + request.get_header("Content-type") + "\r\n\r\n").encode()
        + request.data
    )
    parts = {
        part.get_param("name", header="content-disposition"): part
        for part in envelope.iter_parts()
    }
    assert json.loads(parts["values"].get_payload(decode=True)) == {
        "artifact": "features"
    }
    assert parts["features"].get_payload(decode=True) == raw.read_bytes()
    assert parts["features"].get_content_type() == "application/octet-stream"


def test_raw_http_summary_requires_binary_transport_evidence():
    report = summary()
    target = {**TARGET, "name": "native.aoti.files", "ingress": "multipart/form-data"}
    report["targets"] = [target]
    report["expected_cases"] = job.expected_cases([target])
    report["cases"] = [
        {
            "name": name,
            "status": "passed",
            "details": {
                "transport": "multipart/form-data",
                "input_size_bytes": 16,
                "input_sha256": "e" * 64,
            },
        }
        for name in report["expected_cases"]
    ]
    identity = {
        "run_id": "unit-run",
        "source_sha": "a" * 40,
        "image_digest": "registry/qa@sha256:" + "c" * 64,
        "targets": [target],
    }
    job.validate_summary(report, **identity)
    next(case for case in report["cases"] if case["name"].endswith(".first_request"))[
        "details"
    ].pop("input_sha256")
    with pytest.raises(ValueError, match="Raw-file HTTP"):
        job.validate_summary(report, **identity)


def test_controller_requires_the_exact_five_qualified_bundles(tmp_path, monkeypatch):
    from qa.native_inference import pnms_local_job as local

    controller, args = controller_args(
        tmp_path,
        monkeypatch,
        "--local-summary",
        "/outputs/native-inference/local-run/summary.json",
    )
    candidates = local_report(Path("/outputs"))["release_candidates"]
    report = summary()
    report["targets"] = [
        {
            **TARGET,
            "name": local.candidate_name(candidate),
            "ingress": candidate["ingress"],
            "build_id": candidate["build_id"],
            "build_receipt": candidate["build_receipt"],
        }
        for candidate in candidates
    ]
    report["input_local_summary"] = {
        "path": args.local_summary,
        "source_sha": args.expected_source_sha,
        "image_digest": args.image,
        "sha256": "e" * 64,
        "candidates": candidates,
    }
    report["expected_cases"] = job.expected_cases(report["targets"])
    report["cases"] = [
        {
            "name": name,
            "status": "passed",
            "details": {
                "runner_sha256": job.sha256(job.__file__),
                "transport": "multipart/form-data",
                "input_sha256": "e" * 64,
                "input_size_bytes": 16,
            },
        }
        for name in report["expected_cases"]
    ]
    controller.validate_report(report, args=args)

    swapped = copy.deepcopy(report)
    swapped["targets"][-1]["bundle_digest"] = "f" * 64
    with pytest.raises(ValueError, match="candidate"):
        controller.validate_report(swapped, args=args)

    relabeled = copy.deepcopy(report)
    relabeled["targets"][-1]["ingress"] = "application/json"
    with pytest.raises(ValueError, match="candidate"):
        controller.validate_report(relabeled, args=args)

    report["targets"].pop()
    report["input_local_summary"]["candidates"].pop()
    report["expected_cases"] = job.expected_cases(report["targets"])
    report["cases"] = [
        case for case in report["cases"] if case["name"] in report["expected_cases"]
    ]
    with pytest.raises(ValueError, match="all five"):
        controller.validate_report(report, args=args)


@pytest.fixture
def inherited_repository_override(tmp_path, monkeypatch):
    alternate = tmp_path / "alternate-checkout"
    (alternate / "scripts").mkdir(parents=True)
    (alternate / "scripts/plugin_dev.py").write_text("# alternate source helper\n")
    (alternate / "native-inference").mkdir()
    monkeypatch.setenv("PNMS_REPOSITORY", str(alternate))
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "application-cli/src")
    )
    return alternate


@pytest.mark.parametrize("runtime", ["cpp", "python"])
def test_host_serve_binds_the_selected_repository(
    tmp_path, monkeypatch, inherited_repository_override, runtime
):
    from pnms_app import cli, common

    repository = Path(__file__).resolve().parents[1]
    args = job.parser().parse_args(
        [
            "--repository",
            str(repository),
            "--output",
            str(tmp_path / "qa"),
            "--run-id",
            "repository-unit",
            "--expected-source-sha",
            "a" * 40,
            "--image-digest",
            "registry/qa@sha256:" + "b" * 64,
        ]
    )
    context = job.Context(args)
    captured = {}

    def launch(argv, **options):
        captured.update(argv=argv, environment=options["env"])
        return SimpleNamespace(pid=123)

    def case(name, function):
        if name.endswith((".start", ".shutdown")):
            return function()
        if name.endswith(".readiness"):
            return {"native": {"deployment_id": "owned"}}
        return {}

    context.case = case
    monkeypatch.setattr(job, "_port", lambda: 18082)
    monkeypatch.setattr(job, "HTTP", lambda *_args, **_options: SimpleNamespace())
    monkeypatch.setattr(job.subprocess, "Popen", launch)
    monkeypatch.setattr(
        job,
        "OwnedService",
        lambda *_args, **_options: SimpleNamespace(close=lambda: {}),
    )
    job.run_target(
        context,
        {**TARGET, "runtime": runtime, "project": str(tmp_path / "app")},
    )
    child_args = cli.parser().parse_args(captured["argv"][2:])
    monkeypatch.setenv("PNMS_REPOSITORY", captured["environment"]["PNMS_REPOSITORY"])
    assert common.repository_root() == inherited_repository_override
    assert common.repository_root(child_args.repository) == repository


@pytest.fixture
def setup_http_job(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    commands = []
    served = []
    monkeypatch.setenv("PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE", "/other/python")
    monkeypatch.setenv("PYO3_PYTHON", "/other/python")
    monkeypatch.setattr(job, "load_host_target", lambda *a, **k: copy.deepcopy(TARGET))
    monkeypatch.setattr(
        job,
        "_bundle_target",
        lambda *a, **k: {
            **copy.deepcopy(TARGET),
            "runtime_identity": k["runtime_identity"],
        },
    )

    def serve(ctx, target):
        served.append(target["runtime"])
        for operation in job.OPERATIONS:
            ctx.case(target["name"] + "." + operation, lambda: {})

    monkeypatch.setattr(job, "run_target", serve)

    def run(
        *, mode="host", include_python=False, bootstrap_code=0, bootstrap_timeout=False
    ):
        def command(argv, **kwargs):
            commands.append((argv, kwargs))
            bootstrap = argv[-1] == "bootstrap-python"
            if bootstrap and bootstrap_timeout:
                raise subprocess.TimeoutExpired(
                    argv,
                    kwargs["timeout"],
                    output=b"building bindings\n",
                    stderr=b"compiler diagnostic\n",
                )
            return subprocess.CompletedProcess(
                argv,
                bootstrap_code if bootstrap else 0,
                stdout='Building editable bindings\n{"status":"bootstrapped"}\n'
                if bootstrap
                else '{"status":"created"}\n',
                stderr="bootstrap diagnostic\n" if bootstrap else "",
            )

        monkeypatch.setattr(job.subprocess, "run", command)
        argv = [
            "--mode",
            mode,
            "--repository",
            str(repository),
            "--python",
            "/selected/python",
            "--mount-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "qa"),
            "--run-id",
            "unit-run",
            "--expected-source-sha",
            "a" * 40,
            "--image-digest",
            "registry/qa@sha256:" + "c" * 64,
            "--startup-timeout",
            "321",
        ]
        if mode == "host":
            argv += ["--build-receipt", str(tmp_path / "native/build.json")]
        if include_python:
            argv.append("--include-python")
        return job.main(argv), commands, served, tmp_path / "qa", repository

    return run


def test_host_python_init_binds_the_selected_repository(
    tmp_path, monkeypatch, inherited_repository_override, setup_http_job
):
    from pnms_app import cli, common

    repository = tmp_path / "repository"
    (repository / "scripts").mkdir(parents=True)
    (repository / "scripts/plugin_dev.py").write_text("# selected source helper\n")
    (repository / "native-inference").mkdir()
    code, commands, _, _, selected = setup_http_job(include_python=True)
    assert code == 0
    argv, options = commands[1]
    child_args = cli.parser().parse_args(argv[2:])
    monkeypatch.setenv("PNMS_REPOSITORY", options["env"]["PNMS_REPOSITORY"])
    assert common.repository_root() == inherited_repository_override
    assert common.repository_root(child_args.repository) == selected


def test_host_python_http_qa_bootstraps_selected_interpreter_and_retains_logs(
    setup_http_job,
):
    code, commands, served, output, repository = setup_http_job(include_python=True)
    assert code == 0
    selected_python = str(Path("/selected/python"))
    assert commands[0][0] == [
        selected_python,
        str(repository / "scripts/plugin_dev.py"),
        "bootstrap-python",
    ]
    environment = commands[0][1]["env"]
    assert environment["PYO3_PYTHON"] == selected_python
    assert environment["PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE"] == selected_python
    assert commands[0][1]["timeout"] == 321
    assert (
        "Building editable bindings"
        in (output / "python-bootstrap.stdout.log").read_text()
    )
    assert (
        output / "python-bootstrap.stderr.log"
    ).read_text() == "bootstrap diagnostic\n"
    assert served == ["cpp", "python"]


def test_failed_python_bootstrap_fails_qa_before_service_start(setup_http_job):
    code, commands, served, output, _ = setup_http_job(
        include_python=True, bootstrap_code=9
    )
    assert code == 1
    assert not served
    assert len(commands) == 1
    assert (
        json.loads((output / "python-bootstrap.command.json").read_text())["returncode"]
        == 9
    )
    assert (
        "bootstrap diagnostic" in (output / "python-bootstrap.stderr.log").read_text()
    )
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "failed"
    assert "python-bootstrap" in report["error"]


def test_python_bootstrap_timeout_preserves_partial_setup_logs(setup_http_job):
    code, _, served, output, _ = setup_http_job(
        include_python=True, bootstrap_timeout=True
    )
    assert code == 1
    assert not served
    assert (output / "python-bootstrap.stdout.log").read_text() == "building bindings\n"
    assert (
        output / "python-bootstrap.stderr.log"
    ).read_text() == "compiler diagnostic\n"
    assert (
        json.loads((output / "python-bootstrap.command.json").read_text())["timed_out"]
        is True
    )
    assert "timed out" in json.loads((output / "summary.json").read_text())["error"]


@pytest.mark.parametrize("mode", ["host", "image"])
def test_native_only_http_qa_needs_no_python_bootstrap(setup_http_job, mode):
    code, commands, served, _, _ = setup_http_job(mode=mode)
    assert code == 0
    assert not commands
    assert served == ["cpp"]
