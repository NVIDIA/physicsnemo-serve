"""Sealed, byte-verified native build transport; no models or framework imports."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import tarfile
import time

from native_release import (
    contained,
    file_identity,
    inventory,
    read_json,
    records,
    relative,
    require,
    verify_inventory,
    write_json,
)

MAX_ARCHIVE_BYTES = 16 * 1024**3


def seal(directory, archive, **provenance):
    directory, archive = Path(directory), Path(archive)
    expected = inventory(directory)
    require(
        not archive.exists() and not archive.is_symlink(),
        "archive output already exists",
    )
    require(
        not archive.absolute().is_relative_to(directory.absolute()),
        "archive cannot be inside its own input",
    )
    try:
        with tarfile.open(archive, "x", format=tarfile.PAX_FORMAT) as bundle:
            for item in expected:
                bundle.add(
                    contained(directory, item["path"]),
                    arcname=item["path"],
                    recursive=False,
                )
        verify_inventory(directory, expected, exact=True)
        return {
            "schema_version": 1,
            "status": "complete",
            **provenance,
            "archive": file_identity(archive),
            "files": expected,
        }
    except BaseException:
        archive.unlink(missing_ok=True)
        raise


def seal_evidence(directory, summary, *, maximum_file_bytes=16 * 1024**2):
    """Export original reports plus hashes of large data retained on node-NFS.

    Field arrays and compiled payloads are already independently inventoried by
    their correctness receipts. Avoid duplicating gigabytes in report recovery.
    JSON/text reports and logs up to64MiB are retained byte-for-byte.
    """
    directory = Path(directory)
    staging = directory / "evidence"
    require(not staging.exists(), "qualification evidence already staged")
    files = inventory(directory)
    staging.mkdir()
    omitted = []
    for item in files:
        source = contained(directory, item["path"])
        report = source.suffix in {".json", ".log", ".txt", ".md"}
        limit = 64 * 1024**2 if report else maximum_file_bytes
        if item["size_bytes"] > limit:
            omitted.append(item)
            continue
        destination = staging / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(
            file_identity(destination)
            == {key: item[key] for key in ("sha256", "size_bytes")},
            "qualification evidence changed during capture",
        )
    write_json(
        staging / "retained-files.json",
        {
            "schema_version": 1,
            "scope": "Original reports/logs and small files; large data remain at the recorded node-NFS job path with these exact byte identities",
            "original_job_path": str(directory),
            "original_files": files,
            "omitted_large_files": omitted,
        },
    )
    manifest = seal(
        staging,
        directory / "bundle.tar",
        **{key: summary[key] for key in ("run_id", "source_sha", "image_digest")},
        qualification="exact-image-service-evidence-v1",
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def extract(archive, manifest, output, *, maximum_bytes=MAX_ARCHIVE_BYTES):
    archive, output = Path(archive), Path(output)
    require(
        manifest.get("schema_version") == 1 and manifest.get("status") == "complete",
        "completed artifact transport manifest required",
    )
    require(
        file_identity(archive) == manifest.get("archive"),
        "artifact archive SHA256 or size differs",
    )
    require(
        archive.stat().st_size <= maximum_bytes, "artifact archive exceeds byte limit"
    )
    expected = records(manifest["files"])
    require(
        sum(item["size_bytes"] for item in expected) <= maximum_bytes,
        "artifact contents exceed byte limit",
    )
    require(
        not output.exists() and not output.is_symlink(),
        "artifact extraction output already exists",
    )
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        names = [str(relative(member.name)) for member in members]
        require(len(set(names)) == len(names), "archive contains duplicate file paths")
        require(
            set(names) == {item["path"] for item in expected},
            "archive members differ from manifest",
        )
        by_name = {item["path"]: item for item in expected}
        for member, name in zip(members, names, strict=True):
            require(
                member.isfile() and member.size == by_name[name]["size_bytes"],
                "archive contains a link, special file or wrong size",
            )
        try:
            output.mkdir(parents=True, exist_ok=False)
            for member, name in zip(members, names, strict=True):
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                require(source is not None, "archive member has no file data")
                with source, destination.open("xb") as stream:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
            verify_inventory(output, expected, exact=True)
        except BaseException:
            shutil.rmtree(output)
            raise
    return output


def handler(archive, manifest):
    """A private ingress protects this allowlisted, read-only HTTP surface."""
    archive = Path(archive)
    require(file_identity(archive) == manifest["archive"], "export archive changed")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/manifest.json":
                body, media = json.dumps(manifest).encode(), "application/json"
            elif self.path == "/bundle.tar":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Content-Length", str(archive.stat().st_size))
                self.end_headers()
                with archive.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile, 1024 * 1024)
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass  # Ingress credentials and request headers never enter evidence.

    return Handler


def serve(archive, manifest_path, *, port=8080, deadline=1200):
    require(0 < deadline <= 1800, "CPU exporter deadline must be at most 1800 seconds")
    manifest = read_json(manifest_path)
    with ThreadingHTTPServer(("0.0.0.0", port), handler(archive, manifest)) as server:
        server.timeout = 1
        until = time.monotonic() + deadline
        while time.monotonic() < until:
            server.handle_request()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--deadline", type=int, default=1200)
    args = parser.parse_args()
    serve(args.archive, args.manifest, port=args.port, deadline=args.deadline)


if __name__ == "__main__":
    main()
