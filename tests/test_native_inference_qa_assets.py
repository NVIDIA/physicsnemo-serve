"""Pinned asset download, cache repair, and publication contracts."""

import hashlib
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from urllib.error import HTTPError

import pytest

from qa.native_inference import assets


@pytest.fixture
def pinned(monkeypatch):
    payloads = {}
    descriptors = {}
    for name, filename in (
        ("checkpoint", "model.mdlus"),
        ("stats", "stats.json"),
        ("vtp", "surface.vtp"),
        ("stl", "vehicle.stl"),
    ):
        payload = f"trusted {name}".encode()
        url = f"https://example.invalid/pinned/{filename}"
        payloads[url] = payload
        descriptors[name] = {
            "filename": filename,
            "url": url,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    monkeypatch.setattr(assets, "ASSETS", descriptors)
    monkeypatch.setattr(assets.time, "sleep", lambda _: None)
    return payloads


def install_downloads(monkeypatch, payloads):
    requested = []

    def open_url(request, *, timeout):
        requested.append(request.full_url)
        assert timeout > 0
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_url)
    return requested


def test_cold_cache_publishes_verified_files_and_relative_manifest(
    tmp_path, monkeypatch, pinned
):
    requested = install_downloads(monkeypatch, pinned)
    manifest = assets.prepare_assets(tmp_path)
    document = json.loads(manifest.read_text())
    assert set(requested) == set(pinned)
    assert document["format_version"] == 1
    assert manifest.parent.parent == tmp_path
    for name, descriptor in assets.ASSETS.items():
        entry = document[name]
        assert not Path(entry["path"]).is_absolute()
        payload = (manifest.parent / entry["path"]).read_bytes()
        assert payload == pinned[descriptor["url"]]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert not list(tmp_path.rglob("*.part"))


def test_verified_warm_cache_works_offline(tmp_path, monkeypatch, pinned):
    install_downloads(monkeypatch, pinned)
    original = assets.prepare_assets(tmp_path)

    def offline(*args, **kwargs):
        pytest.fail("verified cache must not access the network")

    monkeypatch.setattr(assets.urllib.request, "urlopen", offline)
    assert assets.prepare_assets(tmp_path) == original


def test_concurrent_callers_share_one_verified_cache(tmp_path, monkeypatch, pinned):
    requested = install_downloads(monkeypatch, pinned)
    ready = threading.Barrier(2)

    def prepare():
        ready.wait(timeout=5)
        return assets.prepare_assets(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(prepare) for _ in range(2)]
        manifests = [future.result(timeout=10) for future in futures]
    assert manifests[0] == manifests[1]
    assert len(requested) == len(pinned)
    assert set(requested) == set(pinned)
    document = json.loads(manifests[0].read_text())
    for name, descriptor in assets.ASSETS.items():
        assert (manifests[0].parent / document[name]["path"]).read_bytes() == pinned[
            descriptor["url"]
        ]


@pytest.mark.parametrize("payload", [b"x" * len(b"trusted checkpoint"), b"short"])
def test_invalid_download_does_not_publish_payload_or_manifest(
    tmp_path, monkeypatch, pinned, payload
):
    descriptor = assets.ASSETS["checkpoint"]
    pinned[descriptor["url"]] = payload
    requested = install_downloads(monkeypatch, pinned)
    with pytest.raises(ValueError, match="SHA256|size"):
        assets.prepare_assets(tmp_path)
    assert requested == [descriptor["url"]]
    assert not list(tmp_path.rglob(descriptor["filename"]))
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("*.part"))


def test_corrupt_cache_repairs_only_changed_asset(tmp_path, monkeypatch, pinned):
    install_downloads(monkeypatch, pinned)
    manifest = assets.prepare_assets(tmp_path)
    descriptor = assets.ASSETS["vtp"]
    target = manifest.parent / descriptor["filename"]
    target.write_bytes(b"x" * descriptor["size"])
    requested = install_downloads(monkeypatch, pinned)
    assert assets.prepare_assets(tmp_path) == manifest
    assert requested == [descriptor["url"]]
    assert target.read_bytes() == pinned[descriptor["url"]]


def test_failed_repair_removes_stale_manifest(tmp_path, monkeypatch, pinned):
    install_downloads(monkeypatch, pinned)
    manifest = assets.prepare_assets(tmp_path)
    descriptor = assets.ASSETS["vtp"]
    (manifest.parent / descriptor["filename"]).write_bytes(b"corrupt")
    pinned[descriptor["url"]] = b"x" * descriptor["size"]
    install_downloads(monkeypatch, pinned)
    with pytest.raises(ValueError, match="SHA256"):
        assets.prepare_assets(tmp_path)
    assert not manifest.exists()
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("status,attempts", [(503, 3), (403, 1)])
def test_only_transient_http_failures_are_retried(
    tmp_path, monkeypatch, pinned, status, attempts
):
    requested = []

    def fail(request, *, timeout):
        requested.append(request.full_url)
        raise HTTPError(request.full_url, status, "test failure", {}, None)

    monkeypatch.setattr(assets.urllib.request, "urlopen", fail)
    with pytest.raises(HTTPError):
        assets.prepare_assets(tmp_path)
    assert len(requested) == attempts
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("*.part"))


def test_partial_transfer_is_discarded_before_transient_retry(
    tmp_path, monkeypatch, pinned
):
    checkpoint = assets.ASSETS["checkpoint"]
    attempts = 0

    class Interrupted(io.BytesIO):
        def read(self, size):
            if self.tell():
                raise ConnectionResetError("interrupted transfer")
            return super().read(4)

    def open_url(request, *, timeout):
        nonlocal attempts
        if request.full_url == checkpoint["url"]:
            attempts += 1
            if attempts == 1:
                return Interrupted(pinned[request.full_url])
        return io.BytesIO(pinned[request.full_url])

    monkeypatch.setattr(assets.urllib.request, "urlopen", open_url)
    manifest = assets.prepare_assets(tmp_path)
    assert attempts == 2
    assert (manifest.parent / checkpoint["filename"]).read_bytes() == pinned[
        checkpoint["url"]
    ]
    assert not list(tmp_path.rglob("*.part"))
