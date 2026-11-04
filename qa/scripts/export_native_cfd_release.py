#!/usr/bin/env python3
"""Export only a passed native CFD job's qualified release inputs.

Run this on a short CPU-only Lepton deployment sharing the job's node-NFS mount.
Require explicit endpoint authentication in the deployment configuration. The
HTTP handler serves only the verified archive and its digest, never a directory.
"""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def inventory(package: Path) -> list[dict]:
    result = []
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError("package symlinks cannot be exported")
        if path.is_file():
            result.append(
                {
                    "path": path.relative_to(package).as_posix(),
                    "sha256": digest(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return result


def build_bundle(job_root: Path, output: Path, source_sha: str) -> dict:
    job_root = job_root.resolve(strict=True)
    receipt_path, summary_path = (
        job_root / "build-receipt.json",
        job_root / "summary.json",
    )
    receipt, summary = read(receipt_path), read(summary_path)
    if receipt.get("status") != "passed" or summary.get("status") != "passed":
        raise ValueError("only passed qualification artifacts may be exported")
    if receipt.get("qualification") != "full-run1-fields-and-metrics-v1":
        raise ValueError("full run_1 field and metric qualification is required")
    if (
        receipt.get("source_sha") != source_sha
        or summary.get("source_sha") != source_sha
    ):
        raise ValueError("qualification source SHA differs")
    if digest(summary_path) != receipt.get("summary_sha256"):
        raise ValueError("qualification summary changed after its receipt")
    declared = receipt.get("packages", [])
    if len(declared) != 2:
        raise ValueError("full and exact tail packages are required")
    selected: list[tuple[Path, str]] = []
    points = set()
    for record in declared:
        matches = [
            path.parent
            for path in (job_root / "build").rglob("model.json")
            if digest(path) == record["manifest_sha256"]
        ]
        if len(matches) != 1:
            raise ValueError("package receipt must identify exactly one package")
        package = matches[0]
        files = inventory(package)
        if files != sorted(record["files"], key=lambda item: item["path"]):
            raise ValueError("compiled package file inventory changed")
        manifest = read(package / "model.json")
        count = next(
            item["shape"][1] for item in manifest["inputs"] if item["name"] == "fx"
        )
        if count not in {60000, 8095} or count in points:
            raise ValueError("unexpected full/tail point dimension")
        points.add(count)
        for item in files:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe package-relative path")
            selected.append(
                (package / item["path"], f"packages/points-{count}/{item['path']}")
            )
    stats = job_root / "native/global_stats.json"
    if stats.is_symlink() or digest(stats) != receipt.get("stats_sha256"):
        raise ValueError("normalization statistics changed")
    selected.extend(
        [
            (receipt_path, "build-receipt.json"),
            (summary_path, "summary.json"),
            (stats, "global_stats.json"),
            (job_root / "environment.json", "environment.json"),
        ]
    )
    startup = receipt.get("startup_fixture")
    if startup:
        for record in startup["files"]:
            relative = PurePosixPath(record["path"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "startup"
            ):
                raise ValueError("unsafe startup fixture path")
            source = job_root / relative
            if (
                source.is_symlink()
                or not source.is_file()
                or digest(source) != record["sha256"]
                or source.stat().st_size != record["size_bytes"]
            ):
                raise ValueError("trusted startup fixture changed")
            selected.append((source, record["path"]))
    for source, _ in selected:
        if (
            not source.is_file()
            or source.is_symlink()
            or not source.resolve().is_relative_to(job_root)
        ):
            raise ValueError("export file must remain inside this qualification job")
    with tarfile.open(output, "x") as archive:
        for source, name in selected:
            archive.add(source, arcname=name, recursive=False)
    # Recheck source inventories to detect concurrent modification during export.
    for record in declared:
        package = next(
            path
            for path, _ in selected
            if path.name == "model.json" and digest(path) == record["manifest_sha256"]
        ).parent
        if inventory(package) != sorted(record["files"], key=lambda item: item["path"]):
            output.unlink(missing_ok=True)
            raise ValueError("package changed during export")
    return {
        "schema_version": 1,
        "source_sha": source_sha,
        "builder_image": receipt["builder_image"],
        "sha256": digest(output),
        "size_bytes": output.stat().st_size,
        "files": [name for _, name in selected],
    }


def handler(archive: Path, manifest: dict):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                body, media = b"ready", "text/plain"
            elif self.path == "/manifest.json":
                body, media = json.dumps(manifest).encode(), "application/json"
            elif self.path == "/release.tar":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Content-Length", str(archive.stat().st_size))
                self.end_headers()
                with archive.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", media)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--archive", type=Path, default=Path("/tmp/native-cfd-release.tar")
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    manifest = build_bundle(args.job_root, args.archive, args.source_sha)
    ThreadingHTTPServer(
        ("0.0.0.0", args.port), handler(args.archive, manifest)
    ).serve_forever()


if __name__ == "__main__":
    main()
