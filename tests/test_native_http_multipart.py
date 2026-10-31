"""Real multipart bytes pass through the same HTTP qualification submission path."""

from email.parser import BytesParser
from email.policy import default
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
from native_release_http_job import ServiceJob
from qa.scripts.run_native_cfd_http_job import HTTP


def test_qualification_submits_real_binary_multipart_and_checks_fixture_identity(
    tmp_path,
):
    data = b"\x00\xff\x01\x80binary\r\ninput"
    fixture = tmp_path / "release" / "qualification" / "fixtures" / "features.bin"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(data)
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append((self.headers["Content-Type"], body))
            self.send_response(202)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "workflow": "workflow",
                        "pipeline": ["schedule", "execute", "results"],
                        "run_id": "run-1",
                    }
                ).encode()
            )

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path.endswith("/status"):
                value = {"status": "succeeded"}
            else:
                value = {
                    "payload": {"answer": 3},
                    "execution": {
                        "run_id": "run-1",
                        "status": "succeeded",
                        "outputs": [],
                        "native": {
                            "source_sha": "a" * 40,
                            "release_image": "registry.example/native@sha256:"
                            + "b" * 64,
                            "runtime": "cpp",
                            "metadata": {"session_load_count": 1},
                        },
                    },
                }
            self.wfile.write(json.dumps(value).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        job = object.__new__(ServiceJob)
        job.root = tmp_path / "evidence"
        job.root.mkdir()
        job.workflow = "workflow"
        job.runs = []
        job.results = {}
        job.args = SimpleNamespace(
            image="registry.example/native@sha256:" + "b" * 64,
            request_timeout=1,
            release_root=tmp_path / "release",
        )
        job.release = {"source_sha": "a" * 40, "models": {"primary": {}}}
        job.deployment = {
            "input_files": {
                "features": {
                    "required": True,
                    "media_types": ["application/octet-stream"],
                    "max_size_bytes": 1024,
                }
            }
        }
        case = {
            "request": {"scale": 2},
            "expected": {"answer": 3},
            "files": {
                "features": {
                    "source": "fixtures/features.bin",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "media_type": "application/octet-stream",
                }
            },
        }
        assert (
            job.request(HTTP(f"http://127.0.0.1:{server.server_port}"), "upload", case)[
                0
            ]
            == "run-1"
        )
        content_type, body = captured[0]
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
        )
        assert message.is_multipart(), "qualification posted JSON instead of multipart"
        parts = {
            part.get_param("name", header="content-disposition"): part
            for part in message.iter_parts()
        }
        assert json.loads(parts["parameters"].get_payload(decode=True)) == {"scale": 2}
        assert parts["features"].get_payload(decode=True) == data
        assert parts["features"].get_content_type() == "application/octet-stream"
        fixture.write_bytes(b"tampered")
        import pytest

        with pytest.raises(ValueError, match="fixture"):
            job.request(
                HTTP(f"http://127.0.0.1:{server.server_port}"), "tampered", case
            )
        assert len(captured) == 1, "changed fixture was submitted"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
