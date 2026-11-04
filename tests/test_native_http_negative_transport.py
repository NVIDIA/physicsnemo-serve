"""Exercise negative QA requests through their real HTTP representation."""

from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release_faults as faults  # noqa: E402
import native_release_http_job as qa  # noqa: E402
from qa.scripts.run_native_cfd_http_job import HTTP  # noqa: E402


@pytest.fixture
def malformed_probe(tmp_path, monkeypatch):
    data = bytes(range(16))
    release_root = tmp_path / "release"
    fixture = release_root / "qualification/fixtures/features.bin"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(data)
    rule = {
        "required": True,
        "media_types": ["application/octet-stream"],
        "max_size_bytes": 16,
    }
    case = {
        "name": "default",
        "request": {
            "values": {"artifact": "features", "shape": [4], "dtype": "float32"}
        },
        "expected": {"values": [1, 3, -1, 9]},
        "files": {
            "features": {
                "source": "fixtures/features.bin",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "media_type": "application/octet-stream",
            }
        },
    }
    release = {
        "workflow": "workflow",
        "qualification": {
            "required_cases": ["default"],
            "cases_document": {"cases": [case]},
        },
    }
    captured = []
    options = {"multipart": True, "status": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            content_type = self.headers["Content-Type"]
            captured.append((content_type, body))
            is_multipart = content_type.startswith("multipart/form-data;")
            status = options["status"] or (
                422 if is_multipart == options["multipart"] else 415
            )
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"{}")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Job(qa.ServiceJob):
        def __init__(self, args, release, deployment):
            self.args, self.release, self.deployment = args, release, deployment
            self.root, self.workflow = args.output, "workflow"
            self.stack, self.monitor = None, None
            self.summary = {"cases": [], "expected_cases": []}

        def case(self, name, function):
            # Isolate the production run() malformed gate; other gates have
            # their own component tests and require the actual GPU service.
            result = function() if name == "malformed_request" else None
            self.summary["expected_cases"].append(name)
            self.summary["cases"].append({"name": name, "status": "passed"})
            return {"worker": {"pid": 123}} if name == "startup" else result

        def stack_tail(self):
            return ""

        def save(self):
            pass

    class Monitor:
        def start(self):
            pass

        def stop(self):
            return {"samples": []}

    monkeypatch.setattr(qa, "ServiceJob", Job)
    monkeypatch.setattr(qa, "MemoryMonitor", Monitor)
    monkeypatch.setattr(qa, "RedisProbe", lambda: None)
    monkeypatch.setattr(
        qa, "HTTP", lambda _: HTTP(f"http://127.0.0.1:{server.server_port}")
    )
    monkeypatch.setattr(qa, "verify_bundle", lambda _: release)
    monkeypatch.setattr(qa, "native_process_snapshot", lambda _: {"pid": 123})
    monkeypatch.setattr(qa, "seal_evidence", lambda *args: None)

    def run(*, multipart=True, status=None):
        options.update(multipart=multipart, status=status)
        if not multipart:
            case.pop("files")
        (release_root / "deployment.json").write_text(
            json.dumps({"input_files": {"features": rule} if multipart else {}})
        )
        args = SimpleNamespace(
            release_root=release_root,
            output=tmp_path / "job",
            image="registry/image@sha256:" + "b" * 64,
        )
        return qa.run(args)

    yield run, captured, data
    server.shutdown()
    server.server_close()
    thread.join()


def test_multipart_malformed_gate_sends_invalid_parameters_with_valid_sealed_files(
    malformed_probe,
):
    run, captured, data = malformed_probe
    assert run()["status"] == "passed"
    assert len(captured) == 1
    content_type, body = captured[0]
    message = BytesParser(policy=default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    assert message.is_multipart()
    parts = {
        part.get_param("name", header="content-disposition"): part
        for part in message.iter_parts()
    }
    assert json.loads(parts["parameters"].get_payload(decode=True)) == []
    assert parts["features"].get_payload(decode=True) == data
    assert parts["features"].get_content_type() == "application/octet-stream"


def test_json_malformed_gate_keeps_existing_json_representation(malformed_probe):
    run, captured, _ = malformed_probe
    assert run(multipart=False)["status"] == "passed"
    assert len(captured) == 1
    content_type, body = captured[0]
    assert content_type == "application/json"
    assert json.loads(body) == {"parameters": []}


@pytest.mark.parametrize("status", [415, 500])
def test_multipart_malformed_gate_rejects_wrong_media_or_server_error(
    malformed_probe, status
):
    run, captured, _ = malformed_probe
    with pytest.raises(ValueError, match="malformed request failed unexpectedly"):
        run(status=status)
    assert len(captured) == 1


def test_multipart_malformed_gate_cannot_accept_a_success_response(malformed_probe):
    run, captured, _ = malformed_probe
    with pytest.raises(ValueError, match="malformed native request was accepted"):
        run(status=202)
    assert len(captured) == 1


@pytest.mark.parametrize("kind", ["gpu", "model"])
def test_startup_negative_probes_preserve_multipart_contract_and_input_fixtures(
    tmp_path, monkeypatch, kind
):
    deployment = {
        "schema_version": 3,
        "manifest_version": "0.1.0",
        "public_contract_digest": "a" * 64,
        "target": {"compilation": {"gpu_product": "NVIDIA H100 80GB HBM3"}},
        "workflow": {"schema_version": 2, "adapter": "tensor.v1"},
        "model_assets": [
            {"sha256": "b" * 64, "path": "/opt/model.pt2", "size_bytes": 1}
        ],
        "input_files": {
            "features": {
                "required": True,
                "media_types": ["application/octet-stream"],
                "max_size_bytes": 16,
            }
        },
        "startup_cases": [
            {
                "name": "default",
                "request": {"values": {"artifact": "features"}},
                "input_artifacts": {
                    "features": {
                        "path": "/opt/startup/features",
                        "sha256": "c" * 64,
                        "size_bytes": 16,
                        "media_type": "application/octet-stream",
                    }
                },
            }
        ],
    }
    captured = []

    def worker(argv, **kwargs):
        captured.append(json.loads(Path(argv[1]).read_text()))
        return SimpleNamespace(
            returncode=1,
            stderr="GPU product differs" if kind == "gpu" else "model SHA256 differs",
        )

    monkeypatch.setattr(faults.subprocess, "run", worker)
    job = SimpleNamespace(
        root=tmp_path, args=SimpleNamespace(image="registry/image@sha256:" + "b" * 64)
    )
    registry = {
        "worker": {
            "pid": 123,
            "worker_instance_id": "instance",
            "registered_at": "time",
            "stream": "execute",
        }
    }
    qa.reject_startup(
        job, SimpleNamespace(registry=lambda: registry), deployment, kind=kind
    )
    changed = captured[0]
    for key in (
        "workflow",
        "schema_version",
        "manifest_version",
        "public_contract_digest",
        "input_files",
        "startup_cases",
    ):
        assert changed[key] == deployment[key]
    if kind == "gpu":
        assert (
            changed["target"]["compilation"]["gpu_product"]
            == "unsupported-other-product-same-sm"
        )
        assert changed["model_assets"] == deployment["model_assets"]
    else:
        assert changed["model_assets"][0]["sha256"] == "0" * 64
        assert changed["target"] == deployment["target"]
