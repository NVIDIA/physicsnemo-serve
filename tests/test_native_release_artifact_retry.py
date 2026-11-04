"""Exercise private archive handoff retries without network calls or real sleeps."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "native-inference/deploy"))
import native_release_artifacts as artifacts  # noqa: E402
import native_release_lepton as lepton  # noqa: E402


class Transfer:
    def __init__(self, root, monkeypatch, responses):
        self.root = root
        self.now = 0.0
        self.sleeps = []
        self.responses = list(responses)
        self.downloads = []
        self.removed = False
        self.tampered = False
        self.duration = 0
        self.image = "registry/image@sha256:" + "b" * 64
        self.expected = {
            "source_sha": "a" * 40,
            "run_id": "completed-job",
            "image_digest": self.image,
        }
        source = root / "source"
        source.mkdir()
        (source / "proof.json").write_bytes(b'{"status":"passed"}\n')
        self.archive = root / "sealed.tar"
        self.manifest = artifacts.seal(source, self.archive, **self.expected)
        self.output = root / "transfer"
        monkeypatch.setattr(lepton.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(lepton.time, "sleep", self.sleep)
        monkeypatch.delenv("NATIVE_RELEASE_ARTIFACT_PROXY", raising=False)

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds

    def run(self, argv, **kwargs):
        if argv[:3] == ["lep", "workspace", "url"]:
            return SimpleNamespace(returncode=0, stdout="https://workspace.example")
        if argv[0] == "lep":
            if argv[2] == "get":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "status": {
                                "endpoint": {
                                    "external_endpoint": "https://private.example"
                                }
                            }
                        }
                    ),
                )
            if argv[2] == "remove":
                self.removed = True
            return SimpleNamespace(returncode=0, stdout="")
        assert argv[0] == "curl"
        assert kwargs["sensitive"] is True
        output = Path(argv[argv.index("--output") + 1])
        if argv[-1].endswith("/manifest.json"):
            output.write_text(json.dumps(self.manifest))
            return SimpleNamespace(
                returncode=0, stdout="200" if "--header" in argv else "403"
            )
        assert argv[-1].endswith("/bundle.tar")
        self.downloads.append((list(argv), kwargs))
        self.now += self.duration
        action = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(action, Exception):
            raise action
        if isinstance(action, tuple):
            code, status = action
            output.write_bytes(b"partial archive")
        else:
            code, status = 0, action
            output.write_bytes(
                self.archive.read_bytes()
                if status == 200 and not self.tampered
                else b"no healthy upstream"
            )
        return SimpleNamespace(returncode=code, stdout=str(status))

    def fetch(self, timeout=30):
        args = SimpleNamespace(
            workspace_id="workspace",
            workspace_url=None,
            nfs_mount_base="/PhysicsNeMo/platform",
            lustre_dir="qa",
            mount_target="/outputs",
            lustre_storage="lustre",
            node_group="nodes",
            pull_secret="pull",
        )
        return lepton.fetch_artifact(
            image=self.image,
            job={
                "args": args,
                "controller": {
                    "job_id": "job-1",
                    "node_id": "node-1",
                    "remote_artifacts": "/outputs/completed-job",
                },
            },
            expected=self.expected,
            output=self.output,
            runner=self,
            timeout=timeout,
        )


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_archive_transient_http_response_retries_and_verifies_completed_bytes(
    tmp_path, monkeypatch, status
):
    transfer = Transfer(tmp_path, monkeypatch, [status, 200])
    result = transfer.fetch()
    assert result["status"] == "passed"
    assert len(transfer.downloads) == 2
    assert transfer.sleeps
    assert transfer.removed
    assert (
        transfer.output / "extracted/proof.json"
    ).read_bytes() == b'{"status":"passed"}\n'
    assert (
        transfer.output / "bundle.tar"
    ).read_bytes() == transfer.archive.read_bytes()


@pytest.mark.parametrize("code", [5, 6, 7, 18, 28, 35, 52, 55, 56, 92])
def test_archive_transient_curl_failure_restarts_download(tmp_path, monkeypatch, code):
    transfer = Transfer(tmp_path, monkeypatch, [(code, "000"), 200])
    assert transfer.fetch()["status"] == "passed"
    assert len(transfer.downloads) == 2
    assert transfer.removed


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired("curl", 1),
        TimeoutError("network timeout"),
        RuntimeError("sensitive release command timed out"),
    ],
)
def test_archive_retry_handles_transport_timeouts(tmp_path, monkeypatch, error):
    transfer = Transfer(tmp_path, monkeypatch, [error, 200])
    assert transfer.fetch()["status"] == "passed"
    assert len(transfer.downloads) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 500])
def test_archive_permanent_response_fails_closed_without_retry(
    tmp_path, monkeypatch, status
):
    transfer = Transfer(tmp_path, monkeypatch, [status, 200])
    with pytest.raises(ValueError, match="artifact download failed"):
        transfer.fetch()
    assert len(transfer.downloads) == 1
    assert not transfer.sleeps
    assert transfer.removed
    assert not (transfer.output / "extracted").exists()


def test_archive_certificate_failure_is_not_retried(tmp_path, monkeypatch):
    transfer = Transfer(tmp_path, monkeypatch, [(60, "000"), 200])
    with pytest.raises(ValueError, match="unreachable"):
        transfer.fetch()
    assert len(transfer.downloads) == 1
    assert not transfer.sleeps
    assert transfer.removed


def test_archive_retries_share_original_deadline_and_bound_process_timeout(
    tmp_path, monkeypatch
):
    transfer = Transfer(tmp_path, monkeypatch, [503])
    with pytest.raises(TimeoutError, match="archive.*timed out"):
        transfer.fetch(timeout=12)
    assert 1 < len(transfer.downloads) <= 4
    assert transfer.now == 12
    assert sum(transfer.sleeps) == 12
    assert all(options["timeout"] <= 12 for _, options in transfer.downloads)
    assert transfer.removed


def test_archive_cannot_accept_success_after_original_deadline(tmp_path, monkeypatch):
    transfer = Transfer(tmp_path, monkeypatch, [200])
    transfer.duration = 4
    with pytest.raises(TimeoutError, match="archive.*timed out"):
        transfer.fetch(timeout=3)
    assert len(transfer.downloads) == 1
    assert not (transfer.output / "extracted").exists()
    assert transfer.removed


def test_archive_retry_does_not_weaken_sha256_or_size_checks(tmp_path, monkeypatch):
    transfer = Transfer(tmp_path, monkeypatch, [503, 200])
    transfer.tampered = True
    with pytest.raises(ValueError, match="SHA256 or size differs"):
        transfer.fetch()
    assert len(transfer.downloads) == 2
    assert not (transfer.output / "extracted").exists()
    assert transfer.removed
