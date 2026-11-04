#!/usr/bin/env python3
"""Finite Lepton acceptance of pnms host HTTP and exact release-image serving.

Run this acceptance driver on Lepton. Its unit tests use in-memory responses
and fake owned processes; they never start HTTP or GPU services locally.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import threading
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import uuid

SUMMARY_BEGIN = "PNMS_HTTP_QA_SUMMARY_BEGIN"
SUMMARY_END = "PNMS_HTTP_QA_SUMMARY_END"
OPERATIONS = (
    "start",
    "readiness",
    "first_request",
    "repeat_request",
    "schema_rejection",
    "unknown_workflow",
    "resident_worker",
    "shutdown",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def expected_cases(targets):
    return ["environment"] + [
        target["name"] + "." + operation
        for target in targets
        for operation in target_operations(target)
    ]


def target_operations(target):
    if not target.get("release_acceptance"):
        return OPERATIONS
    count = target["application_fixtures"]["case_count"]
    return (
        "start",
        "readiness",
        *(f"fixture.{index}" for index in range(count)),
        "overlap",
        "schema_rejection",
        "unknown_workflow",
        "resident_worker",
        "startup_gpu_rejection",
        "startup_model_rejection",
        "request_deadline",
        "lease_cancellation",
        "worker_recovery",
        "shutdown",
    )


def verify_artifacts(descriptors, expected):
    """Same authored identity contract as native_release_http_job's gate."""
    require(
        isinstance(descriptors, list) and isinstance(expected, dict),
        "artifact descriptors/goldens must be an inventory",
    )
    actual = {item.get("name"): item for item in descriptors}
    require(
        len(actual) == len(descriptors) and set(actual) == set(expected),
        "artifact inventory differs from authored fixture",
    )
    for name, identity in expected.items():
        require(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name)
            and isinstance(identity, dict)
            and set(identity) == {"sha256", "size_bytes"}
            and re.fullmatch(r"[a-f0-9]{64}", str(identity.get("sha256", "")))
            and type(identity.get("size_bytes")) is int
            and identity["size_bytes"] >= 0,
            "artifact golden requires SHA256 and byte size",
        )
        require(
            all(actual[name].get(key) == value for key, value in identity.items()),
            "artifact identity differs from authored fixture",
        )
    return actual


def fixture_files(target, case):
    from pnms_app.common import contained

    files = case.get("files", {})
    aliases = case.get("inputs", {})
    rules = target.get("input_files", {})
    require(
        isinstance(files, dict)
        and isinstance(aliases, dict)
        and set(files) == set(aliases),
        "Sealed fixture inputs/files alias maps differ",
    )
    require(
        set(files) <= set(rules)
        and all(not rule["required"] or name in files for name, rule in rules.items()),
        "Fixture input violates declared ingress fields",
    )
    result = {}
    for name, descriptor in files.items():
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
            and isinstance(descriptor, dict)
            and descriptor.get("source") == aliases[name],
            "Invalid fixture input identity",
        )
        source = contained(target["bundle"], descriptor["source"])
        require(
            source.is_file()
            and not source.is_symlink()
            and descriptor.get("sha256") == sha256(source)
            and type(descriptor.get("size_bytes")) is int
            and descriptor["size_bytes"] == source.stat().st_size
            and descriptor["size_bytes"] <= rules[name]["max_size_bytes"]
            and descriptor.get("media_type") in rules[name]["media_types"]
            and not any(c in descriptor["media_type"] for c in "\r\n"),
            "Sealed fixture input bytes/media/size differ",
        )
        result[name] = {"path": source, "media_type": descriptor["media_type"]}
    return result


def assert_readiness(report, target, *, deployment_id=None):
    if target["runtime"] == "python":
        require(report.get("status") == "ready", "Python service is not ready")
        return {}
    native = report.get("native", {})
    require(
        report.get("ready") is True and native.get("ready") is True,
        "Native service is not ready",
    )
    for key, value in (
        ("bundle_digest", target["bundle_digest"]),
        ("workflow_id", target["workflow"]),
        ("runtime_identity", target["runtime_identity"]),
    ):
        require(
            native.get(key) == value,
            f"Native readiness {key} differs from selected release",
        )
    require(
        isinstance(native.get("worker_instance_id"), str)
        and native["worker_instance_id"],
        "Native readiness lacks a resident worker identity",
    )
    if deployment_id is not None:
        require(
            native.get("deployment_id") == deployment_id,
            "Native readiness belongs to another deployment",
        )
    return native


def assert_result(report, *, run_id, target, values=None, fixture=None):
    execution = report.get("execution", {})
    require(
        execution.get("status") == "succeeded" and execution.get("run_id") == run_id,
        "Canonical result execution identity/status differs",
    )
    payload = report.get("payload", {})
    if target["runtime"] == "python":
        require(
            payload.get("value") == values and payload.get("doubled") == 2 * values,
            "Independent Python HTTP result differs",
        )
        return
    native = execution.get("native", {})
    require(
        native.get("runtime") == "cpp"
        and native.get("source_sha") == target["source_sha"],
        "Native HTTP result source/runtime differs",
    )
    require(
        native.get("runtime_identity") == target["runtime_identity"],
        "Native result execution source differs",
    )
    require(
        native.get("metadata", {}).get("session_load_count") == target["model_count"],
        "Native model session inventory differs",
    )
    if fixture is not None:
        from pnms_app.bundle import compare

        if target.get("release_acceptance"):
            verify_artifacts(
                execution.get("outputs", []), fixture.get("expected_artifacts", {})
            )
        compare(
            payload,
            fixture["expected"],
            rtol=fixture.get("rtol", 1e-5),
            atol=fixture.get("atol", 1e-6),
            where=f"HTTP application fixture {fixture['name']}",
        )
        return
    actual = payload.get("values")
    expected = [2 * value + 1 for value in values]
    require(
        isinstance(actual, list) and len(actual) == len(expected),
        "Native output shape differs",
    )
    require(
        all(
            type(a) in (int, float)
            and math.isfinite(a)
            and math.isclose(a, e, rel_tol=1e-5, abs_tol=1e-6)
            for a, e in zip(actual, expected, strict=True)
        ),
        "Independent native HTTP numerical result differs",
    )


def validate_summary(report, *, run_id, source_sha, image_digest, targets):
    require(
        re.fullmatch(r"[a-f0-9]{40}", source_sha),
        "HTTP QA requires a full source revision",
    )
    require(
        re.fullmatch(r"(?:[^\s@]+@)?sha256:[a-f0-9]{64}", image_digest),
        "HTTP QA requires an immutable job image identity",
    )
    require(
        targets and len({target["name"] for target in targets}) == len(targets),
        "HTTP QA target inventory is empty or duplicated",
    )
    scope = (
        "image-http"
        if any(
            target.get("runtime_identity", {}).get("kind")
            in {"registry-image", "image-config"}
            for target in targets
        )
        else "host-http"
    )
    for key, value in {
        "schema_version": 1,
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "status": "passed",
        "qa_scope": scope,
        "targets": targets,
    }.items():
        require(report.get(key) == value, f"HTTP QA {key} identity/status differs")
    names = expected_cases(targets)
    require(
        report.get("expected_cases") == names, "HTTP QA declared case inventory differs"
    )
    cases = report.get("cases")
    require(
        isinstance(cases, list)
        and sorted(case.get("name", "") for case in cases) == sorted(names),
        "HTTP QA executed case inventory differs",
    )
    require(
        all(case.get("status") == "passed" for case in cases),
        "Every HTTP QA case must execute and pass",
    )
    for target in targets:
        if "application_fixtures" in target:
            fixtures = target["application_fixtures"]
            require(
                target.get("runtime") == "cpp"
                and (
                    target.get("release_acceptance")
                    and target.get("ingress")
                    in {"application/json", "multipart/form-data"}
                    or target.get("ingress") == "application/json"
                )
                and isinstance(fixtures, dict)
                and fixtures.get("path") == "fixtures/cases.json"
                and re.fullmatch(r"[a-f0-9]{64}", str(fixtures.get("sha256", "")))
                and type(fixtures.get("case_count")) is int
                and fixtures["case_count"] > 0,
                "Application fixture identity/count is invalid",
            )
            operations = (
                [(f"fixture.{index}", index) for index in range(fixtures["case_count"])]
                if target.get("release_acceptance")
                else [
                    ("first_request", 0),
                    ("repeat_request", min(1, fixtures["case_count"] - 1)),
                ]
            )
            if target.get("release_acceptance"):
                require(
                    target["release_acceptance"]
                    == {
                        "schema_version": 1,
                        "fixture_indices": list(range(fixtures["case_count"])),
                    },
                    "Expanded release fixture inventory differs",
                )
                validate_release_evidence(target, cases)
            for operation, index in operations:
                evidence = next(
                    case.get("details", {}).get("application_fixture", {})
                    for case in cases
                    if case["name"] == target["name"] + "." + operation
                )
                require(
                    evidence.get("sha256") == fixtures["sha256"]
                    and type(evidence.get("index")) is int
                    and evidence["index"] == index
                    and isinstance(evidence.get("name"), str)
                    and evidence["name"],
                    "Application fixture request evidence differs",
                )
        if target.get("ingress") == "multipart/form-data" and not target.get(
            "release_acceptance"
        ):
            for operation in ("first_request", "repeat_request"):
                evidence = next(
                    case.get("details", {})
                    for case in cases
                    if case["name"] == target["name"] + "." + operation
                )
                require(
                    evidence.get("transport") == "multipart/form-data"
                    and evidence.get("input_size_bytes") == 16
                    and re.fullmatch(
                        r"[a-f0-9]{64}", str(evidence.get("input_sha256", ""))
                    ),
                    "Raw-file HTTP evidence is missing",
                )


def _process_record(pid, proc_root=Path("/proc")):
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"parent": int(fields[1]), "start": fields[19]}
    except (OSError, IndexError, ValueError):
        return None


def _descendants(parent, proc_root=Path("/proc")):
    """Remember exact owned descendants, including children in new sessions."""
    if not proc_root.is_dir():
        return {}
    records = {
        int(item.name): _process_record(int(item.name), proc_root)
        for item in proc_root.iterdir()
        if item.name.isdecimal()
    }
    selected = {parent}
    while True:
        children = {
            pid
            for pid, record in records.items()
            if record and record["parent"] in selected
        }
        expanded = selected | children
        if expanded == selected:
            return {
                pid: records[pid]["start"]
                for pid in selected
                if pid != parent and records.get(pid)
            }
        selected = expanded


class OwnedService:
    def __init__(self, process, *, timeout=180, process_group=False):
        self.process = process
        self.timeout = timeout
        self.process_group = process_group

    def close(self):
        descendants = _descendants(self.process.pid) if self.process_group else {}
        if self.process.poll() is None:
            try:
                if self.process_group:
                    os.killpg(self.process.pid, signal.SIGINT)
                else:
                    self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            # The supervisor's children use their own sessions. Killing its
            # process group alone would leak Redis and resident GPU workers.
            if self.process_group:
                descendants.update(_descendants(self.process.pid))
            for pid, start in descendants.items():
                current = _process_record(pid)
                if current and current["start"] == start:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                if self.process_group:
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
            raise ValueError("Owned service exceeded its shutdown deadline") from error
        require(code in (0, 130), f"Owned service exited unsuccessfully with {code}")
        return {"owned": True, "stopped": True, "returncode": code}


class HTTP:
    """One POST per call; polling retries GET only under the original deadline."""

    def __init__(self, base_url, timeout=15):
        parsed = urlparse(base_url)
        require(
            parsed.scheme == "http"
            and parsed.hostname in {"127.0.0.1", "localhost"}
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment,
            "HTTP QA uses a local service URL inside its Lepton job",
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path, *, body=None, files=None):
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        content_type = "application/json"
        if files is not None:
            boundary = "pnms-qa-" + uuid.uuid4().hex
            chunks = []
            for name, value in (body or {}).items():
                chunks.append(
                    (
                        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                        + json.dumps(value, allow_nan=False)
                        + "\r\n"
                    ).encode()
                )
            for name, supplied in files.items():
                input_path = (
                    supplied["path"] if isinstance(supplied, dict) else supplied
                )
                media = (
                    supplied["media_type"]
                    if isinstance(supplied, dict)
                    else "application/octet-stream"
                )
                require(
                    re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
                    and not any(c in media for c in "\r\n"),
                    "Invalid multipart field/media",
                )
                chunks.append(
                    (
                        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                        f'filename="{name}.f32"\r\nContent-Type: {media}\r\n\r\n'
                    ).encode()
                )
                chunks.extend([Path(input_path).read_bytes(), b"\r\n"])
            chunks.append(f"--{boundary}--\r\n".encode())
            data = b"".join(chunks)
            content_type = "multipart/form-data; boundary=" + boundary
        request = Request(
            self.base_url + path,
            data=data,
            method="POST" if data is not None else "GET",
            headers={"Content-Type": content_type},
        )
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(1024 * 1024 + 1)
            require(len(raw) <= 1024 * 1024, "HTTP QA response exceeded its size bound")
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = {"raw": raw.decode(errors="replace")}
            return response.code, payload

    def download(self, path, destination, *, expected):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(self.base_url + path, timeout=self.timeout) as response:
            require(response.status == 200, "artifact download failed")
            data = response.read(expected["size_bytes"] + 1)
        require(
            len(data) == expected["size_bytes"]
            and hashlib.sha256(data).hexdigest() == expected["sha256"],
            "artifact download bytes differ from authored golden",
        )
        destination.write_bytes(data)
        return {"sha256": sha256(destination), "size_bytes": destination.stat().st_size}


def _port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _expected_identity(image):
    if re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        return {"kind": "image-config", "value": image}
    require(
        re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", image),
        "Immutable image digest required",
    )
    return {"kind": "registry-image", "value": image}


def _python_image_target(
    location, *, source_sha, runtime_identity, workflow, plugin_digest, receipt_sha256
):
    import yaml

    require(
        os.environ.get("SOURCE_SHA") == source_sha,
        "Python image source differs from its build",
    )
    require(
        re.fullmatch(r"[a-f0-9]{64}", plugin_digest or ""),
        "Python plugin digest required",
    )
    require(
        re.fullmatch(r"[a-f0-9]{64}", receipt_sha256 or ""),
        "Python build receipt digest required",
    )
    root = Path(location).resolve()
    plugin = root / "plugin"
    files = []
    for path in sorted(plugin.rglob("*")):
        require(not path.is_symlink(), "Python image plugin contains a symlink")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(plugin).as_posix(),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    actual = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        files and actual == plugin_digest,
        "Python image plugin differs from public build receipt",
    )
    manifest = yaml.safe_load((plugin / "plugin.yaml").read_text())
    launch = json.loads((root / "python-launch.json").read_text())
    require(
        workflow
        and launch.get("workflow_id") == workflow == manifest["metadata"]["id"],
        "Python image workflow differs from public build receipt",
    )
    return {
        "name": "python.release",
        "runtime": "python",
        "workflow": workflow,
        "source_sha": source_sha,
        "runtime_identity": runtime_identity,
        "plugin": str(plugin),
        "plugin_digest": actual,
        "build_receipt_sha256": receipt_sha256,
    }


def application_fixture_cases(target):
    """Read the selected sealed JSON fixtures again before submitting requests."""
    from pnms_app.common import contained, read_json

    identity = target["application_fixtures"]
    require(identity.get("path") == "fixtures/cases.json", "Invalid fixture path")
    path = contained(target["bundle"], identity["path"])
    require(sha256(path) == identity["sha256"], "Application fixture bytes changed")
    document = read_json(path)
    require(
        isinstance(document, dict), "Application fixture document must be an object"
    )
    cases = document.get("cases")
    require(
        document.get("schema_version") == 1
        and isinstance(cases, list)
        and cases
        and len(cases) == identity["case_count"],
        "Application fixture count/schema differs",
    )

    def has_assertion(value):
        if isinstance(value, dict):
            return any(has_assertion(item) for item in value.values())
        return True

    names = set()
    for case in cases:
        require(
            isinstance(case, dict)
            and isinstance(case.get("name"), str)
            and case["name"]
            and case["name"] not in names
            and isinstance(case.get("request"), dict)
            and "expected" in case
            and has_assertion(case["expected"]),
            "Application fixtures require unique names, JSON object requests and independent expected assertions",
        )
        names.add(case["name"])
        require(
            target.get("release_acceptance")
            or (not case.get("inputs") and not case.get("files")),
            "Application fixtures support JSON only; file/multipart fixtures are unsupported",
        )
        if target.get("release_acceptance"):
            fixture_files(target, case)
            verify_artifacts(
                [
                    {"name": name, **identity}
                    for name, identity in case.get("expected_artifacts", {}).items()
                ],
                case.get("expected_artifacts", {}),
            )
        for key, default in (("rtol", 1e-5), ("atol", 1e-6)):
            value = case.get(key, default)
            require(
                type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Application fixture tolerances must be finite nonnegative numbers",
            )
    return cases


def _bundle_target(
    location,
    *,
    source_sha,
    name=None,
    runtime_identity=None,
    application_fixtures=False,
    release_acceptance=False,
):
    from pnms_app import bundle

    location = Path(location).resolve()
    seal = bundle.verify(location)
    descriptor = json.loads((location / "bundle.json").read_text())
    require(
        descriptor.get("source_sha") == source_sha,
        "HTTP target source differs from selected source",
    )
    configuration = json.loads(
        (location / descriptor["workflow"]["config"]).read_text()
    )
    adapter = configuration["adapter"]
    template = (
        "tensor"
        if adapter == "tensor.v1"
        else "custom"
        if adapter == "custom.v1"
        else None
    )
    require(
        template is not None, "HTTP affine QA supports tensor/custom native workflows"
    )
    files = template == "tensor" and any(
        rule.get("codec") == "raw-fp32-le"
        for rule in configuration["tensor_adapter"]["inputs"].values()
    )
    require(
        release_acceptance or not application_fixtures or not files,
        "Application fixtures support JSON only; raw/multipart contracts are unsupported",
    )
    if files and not release_acceptance:
        require(
            descriptor.get("input_files", {}).keys() == {"features"},
            "Raw affine HTTP QA requires the sealed features upload policy",
        )
    target = {
        "name": name
        or f"native.{descriptor['backend']}."
        + ("files" if files and not release_acceptance else template),
        "ingress": "multipart/form-data" if files else "application/json",
        "runtime": "cpp",
        "workflow": descriptor["name"],
        "source_sha": source_sha,
        "bundle_digest": seal["bundle_digest"],
        "model_count": len(descriptor["models"]),
        "bundle": str(location),
        "runtime_identity": runtime_identity
        or {"kind": "bundle", "value": seal["bundle_digest"]},
    }
    if application_fixtures or release_acceptance:
        from pnms_app.common import contained, read_json

        manifest = read_json(
            contained(
                location, f"configuration/plugin/{descriptor['name']}/plugin.yaml"
            )
        )
        ingress = manifest.get("ingress", {})
        content_types = ingress.get("content_types") or [ingress.get("content_type")]
        require(
            release_acceptance
            and set(content_types) <= {"application/json", "multipart/form-data"}
            or not files
            and not descriptor.get("input_files")
            and not ingress.get("files")
            and content_types == ["application/json"],
            "Application fixtures support JSON only; raw/multipart contracts are unsupported",
        )
        operation = ingress.get("operations", ingress.get("operation", {})).get(
            "default", "run"
        )
        require(isinstance(operation, str) and operation, "Invalid fixture operation")
        fixture_path = contained(location, "fixtures/cases.json")
        document = read_json(fixture_path)
        cases = document.get("cases")
        target.update(
            operation=operation,
            application_fixtures={
                "path": "fixtures/cases.json",
                "sha256": sha256(fixture_path),
                "case_count": len(cases) if isinstance(cases, list) else 0,
            },
        )
        if release_acceptance:
            require(
                not descriptor.get("input_files")
                or "multipart/form-data" in content_types,
                "Sealed input files require a multipart ingress contract",
            )
            target.update(
                content_types=content_types,
                release_acceptance={
                    "schema_version": 1,
                    "fixture_indices": list(range(len(cases))),
                },
                input_files=descriptor.get("input_files", {}),
                ingress="multipart/form-data"
                if descriptor.get("input_files")
                else "application/json",
            )
        application_fixture_cases(target)
    return target


def load_host_target(receipt, *, source_sha):
    receipt = Path(receipt).resolve()
    document = json.loads(receipt.read_text())
    require(
        document.get("status") == "complete"
        and document.get("runtime") == "cpp"
        and document.get("checks", {}).get("application") == "passed"
        and document.get("checks", {}).get("model") == "passed",
        "HTTP QA needs a completed checked native build",
    )
    require(document.get("source_sha") == source_sha, "Build receipt source differs")
    relative = Path(document["bundle"])
    require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and (receipt.parent / relative).resolve().is_relative_to(receipt.parent),
        "Build receipt must select a relative contained bundle",
    )
    target = _bundle_target(receipt.parent / relative, source_sha=source_sha)
    require(
        target["bundle_digest"] == document["bundle_digest"],
        "Build receipt and compiled bundle differ",
    )
    state = next((parent for parent in receipt.parents if parent.name == ".pnms"), None)
    require(
        state is not None,
        "Host build receipt must belong to its application .pnms directory",
    )
    from pnms_app.build import last_build

    selected = last_build(
        {"root": state.parent, "name": document["name"], "runtime": "cpp"}
    )
    require(
        selected["bundle_digest"] == target["bundle_digest"],
        "Public serve selects another build; use the latest verified candidate receipt",
    )
    return {
        **target,
        "project": str(state.parent),
        "build_receipt": str(receipt),
        "build_id": document["build_id"],
    }


def mounted_path(path, mount_root):
    value, root = Path(path), Path(mount_root).resolve()
    require(
        value.is_absolute()
        and ".." not in value.parts
        and value.resolve().is_relative_to(root),
        "QA input must be an absolute path inside its artifact mount",
    )
    return value.resolve()


def local_candidates(path, *, mount_root, source_sha, image_digest):
    path = mounted_path(path, mount_root)
    if __package__:
        from . import pnms_local_job
    else:
        import pnms_local_job
    document = json.loads(path.read_text())
    require(
        path.name == "summary.json"
        and document.get("run_id") == path.parent.name
        and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", document.get("run_id", "")),
        "Local QA summary path/run identity differs",
    )
    profile = document.get("profile")
    if profile == "fresh-venv":
        if __package__:
            from . import pnms_venv_job as profile_job
        else:
            import pnms_venv_job as profile_job
    else:
        require(profile == "local", "Unsupported local QA handoff profile")
        profile_job = pnms_local_job
    profile_job.validate_summary(
        document,
        run_id=document.get("run_id"),
        source_sha=source_sha,
        image_digest=image_digest,
        profile=profile,
    )
    selected_python = None
    if profile == "fresh-venv":
        check = next(
            case["details"]
            for case in document["cases"]
            if case["name"] == "venv.verify"
        )
        expected_prefix = path.parent / "venv"
        selected_python = str(expected_prefix / "bin/python")
        require(
            check.get("python") == selected_python
            and check.get("prefix") == str(expected_prefix)
            and check.get("base_prefix") != check.get("prefix"),
            "Fresh venv handoff selected Python/prefix differs from its completed setup",
        )
    candidates = document["release_candidates"]
    receipts = [mounted_path(item["build_receipt"], mount_root) for item in candidates]
    require(
        len(set(receipts)) == len(receipts),
        "Local QA candidate receipts are duplicated",
    )
    return candidates, {
        "path": str(path),
        "sha256": sha256(path),
        "run_id": document["run_id"],
        "source_sha": source_sha,
        "image_digest": image_digest,
        "candidates": candidates,
        "profile": profile,
        **({"python": selected_python} if selected_python else {}),
    }


def local_target_inventory(profile):
    """Keep each acceptance profile's exact native inventory independent."""
    if __package__:
        from . import pnms_local_job, pnms_venv_job
    else:
        import pnms_local_job
        import pnms_venv_job
    if profile == "fresh-venv":
        return {
            (backend, template, "application/json")
            for backend, template in pnms_venv_job.NATIVE_TARGETS
        }
    require(profile == "local", "Unsupported local QA handoff profile")
    return set(pnms_local_job.NATIVE_TARGETS)


def configure_local_environment(ctx, evidence):
    if evidence.get("profile") != "fresh-venv":
        return
    require(
        ctx.args.service_bin_dir is None,
        "Fresh venv HTTP QA must rebuild serving binaries; prebuilt binaries are not allowed",
    )
    if __package__:
        from .pnms_venv_job import isolated_environment
    else:
        from pnms_venv_job import isolated_environment
    selected = Path(evidence["python"])
    require(selected.is_file(), "Fresh venv selected Python is missing")
    ctx.args.python = selected
    ctx.environment = isolated_environment(ctx.environment)
    for key in (
        "PNMS_SERVICE_BIN_DIR",
        "CARGO_TARGET_DIR",
        "NATIVE_WORKFLOW_INCLUDE",
        "NATIVE_WORKFLOW_LIB_DIR",
    ):
        ctx.environment.pop(key, None)
    ctx.environment.update(
        VIRTUAL_ENV=str(selected.parent.parent),
        PATH=str(selected.parent) + os.pathsep + ctx.environment.get("PATH", ""),
        PYO3_PYTHON=str(selected),
        PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE=str(selected),
    )


class Context:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.environment = dict(os.environ)
        self.environment["PNMS_SOURCE_SHA"] = args.expected_source_sha
        self.report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "source_sha": args.expected_source_sha,
            "image_digest": args.image_digest,
            "status": "running",
            "qa_scope": "host-http" if args.mode == "host" else "image-http",
            "targets": [],
            "expected_cases": ["environment"],
            "cases": [],
        }

    def save(self):
        write_json(self.root / "summary.json", self.report)

    def case(self, name, function):
        require(
            name in self.report["expected_cases"]
            and not any(case["name"] == name for case in self.report["cases"]),
            "Unknown or duplicate HTTP QA case",
        )
        case = {"name": name, "status": "running"}
        self.report["cases"].append(case)
        started = time.monotonic()
        self.save()
        try:
            result = function()
            case.update(status="passed", details=result or {})
            return result
        except BaseException as error:
            case.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            case["seconds"] = time.monotonic() - started
            self.save()

    def command(self, name, arguments, *, timeout=60, json_output=True):
        timed_out = False
        try:
            result = subprocess.run(
                [str(value) for value in arguments],
                cwd=self.args.repository,
                env=self.environment,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            timed_out = True
            result = subprocess.CompletedProcess(
                arguments, -1, stdout=error.stdout, stderr=error.stderr
            )
        write_json(
            self.root / (name + ".command.json"),
            {
                "argv": list(map(str, arguments)),
                "returncode": result.returncode,
                "timed_out": timed_out,
            },
        )
        for stream in ("stdout", "stderr"):
            content = getattr(result, stream) or ""
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            (self.root / (name + "." + stream + ".log")).write_text(content)
        require(not timed_out, f"HTTP QA setup command timed out: {name}")
        require(result.returncode == 0, f"HTTP QA setup command failed: {name}")
        return json.loads(result.stdout) if json_output else None


def _wait_ready(http, target, *, process, deployment_id, timeout, launch=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None:
            require(
                process.poll() is None,
                "Owned service exited before readiness; inspect service.log",
            )
        selected = deployment_id
        if launch is not None:
            if not launch.is_file():
                time.sleep(0.2)
                continue
            plan = json.loads(launch.read_text())
            selected = plan["deployment_id"]
        try:
            status, report = http.request("/readyz")
            if status == 200:
                assert_readiness(report, target, deployment_id=selected)
                return report
        except (OSError, URLError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError(
        "Public HTTP service did not become ready under its selected identity"
    )


def _python_result_pending(result, *, run_id, target):
    # The unchanged Python worker persists execution metadata before the results
    # role replaces it with the canonical request/execution/payload envelope.
    if target["runtime"] != "python" or set(result) != {
        "run_id",
        "workflow",
        "status",
        "output_path",
        "error",
        "execution_time_seconds",
    }:
        return False
    duration = result["execution_time_seconds"]
    require(
        result["run_id"] == run_id
        and result["workflow"] == target["workflow"]
        and result["status"] == "succeeded"
        and result["error"] is None
        and result["output_path"] is None
        and type(duration) in (int, float)
        and math.isfinite(duration)
        and duration >= 0,
        "Provisional Python result execution identity/status differs",
    )
    return True


def exercise_request(
    http,
    target,
    *,
    values=None,
    output,
    timeout,
    fixture=None,
    fixture_index=None,
    submitted_barrier=None,
):
    workflow = quote(target["workflow"], safe="")
    operation = quote(target.get("operation", "run"), safe="")
    parameters = (
        fixture["request"]
        if fixture is not None
        else {"values": values}
        if target["runtime"] == "cpp"
        else {"value": values}
    )
    input_evidence = {}
    if (
        fixture is not None
        and target.get("release_acceptance")
        and (
            fixture.get("files")
            or target.get("content_types") == ["multipart/form-data"]
        )
    ):
        files = fixture_files(target, fixture)
        status, submitted = http.request(
            f"/v1/infer/{workflow}/{operation}",
            body={"parameters": parameters},
            files=files,
        )
        input_evidence = {
            "transport": "multipart/form-data",
            "input_files": fixture.get("files", {}),
        }
    elif fixture is None and target.get("ingress") == "multipart/form-data":
        output.mkdir(parents=True, exist_ok=True)
        raw_input = output / "input.f32"
        raw_input.write_bytes(struct.pack("<4f", *values))
        parameters = {
            "values": {
                "artifact": "features",
                "dtype": "float32",
                "shape": [4],
                "byte_order": "little",
                "layout": "contiguous",
                "size_bytes": 16,
            }
        }
        status, submitted = http.request(
            f"/v1/infer/{workflow}/{operation}",
            body=parameters,
            files={"features": raw_input},
        )
        input_evidence = {
            "transport": "multipart/form-data",
            "input_sha256": sha256(raw_input),
            "input_size_bytes": raw_input.stat().st_size,
        }
    else:
        status, submitted = http.request(
            f"/v1/infer/{workflow}/{operation}", body={"parameters": parameters}
        )
    write_json(
        output / "submission.json", {"http_status": status, "response": submitted}
    )
    require(
        status == 202 and submitted.get("workflow") == target["workflow"],
        "Public HTTP submission contract differs",
    )
    if target["runtime"] == "cpp":
        require(
            submitted.get("pipeline") == ["schedule", "execute", "results"],
            "Native request did not use the existing service pipeline",
        )
    run_id = submitted.get("run_id")
    require(isinstance(run_id, str) and run_id, "Submitted request has no run identity")
    base = f"/v1/infer/{workflow}/{quote(run_id, safe='')}"
    deadline = time.monotonic() + timeout
    if submitted_barrier is not None:
        submitted_barrier.wait(timeout=timeout)
    while time.monotonic() < deadline:
        code, state = http.request(base + "/status")
        require(code == 200, "Public status request failed")
        write_json(output / "status.json", state)
        if state.get("status") in {"succeeded", "failed", "cancelled"}:
            require(
                state["status"] == "succeeded",
                "HTTP inference failed: " + json.dumps(state),
            )
            code, result = http.request(base + "/results")
            if code == 200:
                write_json(output / "result.json", result)
                if _python_result_pending(result, run_id=run_id, target=target):
                    write_json(output / "provisional-result.json", result)
                    time.sleep(0.1)
                    continue
                assert_result(
                    result, run_id=run_id, target=target, values=values, fixture=fixture
                )
                if target.get("release_acceptance"):
                    outputs = result.get("execution", {}).get("outputs", [])
                    expected_artifacts = fixture.get("expected_artifacts", {})
                    downloads = {}
                    for name, identity in expected_artifacts.items():
                        downloaded = output / "downloads" / name
                        actual = http.download(
                            base + "/results?artifact=" + quote(name, safe=""),
                            downloaded,
                            expected=identity,
                        )
                        require(
                            actual == identity
                            and sha256(downloaded) == identity["sha256"]
                            and downloaded.stat().st_size == identity["size_bytes"],
                            "artifact download bytes differ from fixture",
                        )
                        downloads[name] = actual
                    input_evidence.update(
                        artifact_downloads=downloads,
                        output_paths=[item["storage_path"] for item in outputs],
                        native_attempt=result["execution"]
                        .get("native", {})
                        .get("attempt"),
                    )
                if fixture is not None:
                    input_evidence["application_fixture"] = {
                        "sha256": target["application_fixtures"]["sha256"],
                        "index": fixture_index,
                        "name": fixture["name"],
                    }
                return {
                    **input_evidence,
                    "run_id": run_id,
                    "expected": fixture["expected"]
                    if fixture is not None
                    else [2 * value + 1 for value in values]
                    if target["runtime"] == "cpp"
                    else 2 * values,
                    "runtime_identity": target.get("runtime_identity"),
                    "result_file": str(output / "result.json"),
                }
            require(code == 404, "Public results request failed")
        time.sleep(0.1)
    raise TimeoutError(
        "HTTP inference/status/result deadline exceeded; the POST was not replayed"
    )


def exercise_overlap(http, target, fixtures, *, output, timeout):
    """Both single POSTs are admitted before either observer polls results."""
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                exercise_request,
                http,
                target,
                fixture=fixtures[min(index, len(fixtures) - 1)],
                fixture_index=min(index, len(fixtures) - 1),
                output=output / str(index),
                timeout=timeout,
                submitted_barrier=barrier,
            )
            for index in range(2)
        ]
        results = [future.result(timeout=timeout + 5) for future in futures]
    require(
        len({item["run_id"] for item in results}) == 2,
        "Overlapping requests reused a run identity",
    )
    attempts = [item.get("native_attempt") for item in results]
    require(
        all(isinstance(value, str) and value for value in attempts)
        and len(set(attempts)) == 2,
        "Overlapping requests reused a native attempt",
    )
    paths = [set(item.get("output_paths", [])) for item in results]
    require(
        not paths[0] & paths[1], "Overlapping request output paths are not isolated"
    )
    return {
        "requests": results,
        "both_admitted_before_poll": True,
        "output_isolation": "passed",
        "scope": "overlapping HTTP requests; single resident native execution thread",
    }


def image_launch_plan(artifact_root, target):
    plans = list(Path(artifact_root).glob(".pnms/service-*/launch.json"))
    require(len(plans) == 1, "Exactly one owned image launch plan is required")
    plan = json.loads(plans[0].read_text())
    require(
        plan.get("runtime") == "cpp"
        and plan.get("bundle_digest") == target["bundle_digest"],
        "Image launch plan differs from sealed target",
    )
    return plans[0], plan


def _redis(plan):
    import redis

    return redis.Redis.from_url(
        plan["environment"]["REDIS_URL"],
        decode_responses=True,
        socket_timeout=5,
        socket_connect_timeout=5,
    )


def _registry(client):
    return {
        key: json.loads(value) for key, value in client.hgetall("gpu:registry").items()
    }


def _worker(registry, deployment):
    selected = [
        worker
        for worker in registry.values()
        if worker.get("deployment_id") == deployment
    ]
    require(len(selected) == 1, "Owned native worker registry inventory differs")
    return selected[0]


def _worker_identity(worker):
    return {
        key: worker.get(key)
        for key in (
            "pid",
            "worker_instance_id",
            "stream",
            "workflow_id",
            "executor_class",
            "device_uuid",
            "device_index",
            "deployment_id",
        )
    }


def reject_native_startup(plan, target, *, kind, output):
    """Adapt the existing release startup-negative contract to pnms launch plans."""
    from pnms_app.service import _process_environment

    require(kind in {"gpu", "model"}, "Unknown startup negative probe")
    native = next(item for item in plan["processes"] if item["name"] == "native-worker")
    deployment = json.loads(Path(native["argv"][1]).read_text())
    if kind == "gpu":
        deployment["target"]["compilation"]["gpu_product"] = (
            "qa-unsupported-other-product-same-sm"
        )
        expected = ("product", "gpu", "device")
    else:
        deployment["model_assets"][0]["sha256"] = "0" * 64
        expected = ("sha", "digest", "integrity", "hash")
    config = output / (kind + "-invalid-deployment.json")
    write_json(config, deployment)
    client = _redis(plan)
    before = _worker(_registry(client), plan["deployment_id"])
    environment = _process_environment(native["env"])
    result = subprocess.run(
        [native["argv"][0], str(config)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    after = _worker(_registry(client), plan["deployment_id"])
    evidence = {
        "returncode": result.returncode,
        "stderr": result.stderr[-8000:],
        "kind": kind,
        "worker_before": _worker_identity(before),
        "worker_after": _worker_identity(after),
        "rejected_before_admission": True,
    }
    write_json(output / (kind + "-startup-rejection.json"), evidence)
    require(
        result.returncode != 0
        and any(word in result.stderr.lower() for word in expected),
        "Invalid native deployment was not rejected before readiness",
    )
    require(
        _worker_identity(before) == _worker_identity(after),
        "Startup-negative probe changed the resident worker",
    )
    return evidence


def _owned_processes(parent):
    processes = {}
    for pid, start in _descendants(parent).items():
        try:
            name = Path(os.readlink(f"/proc/{pid}/exe")).name
        except FileNotFoundError:
            continue
        processes.setdefault(name, []).append({"pid": pid, "start": start})
    return {
        name: sorted(items, key=lambda item: item["pid"])
        for name, items in processes.items()
    }


def recover_worker(
    process,
    http,
    target,
    initial,
    plan,
    fixture,
    *,
    output,
    startup_timeout,
    request_timeout,
):
    """Keep the original durable slot present while API/Redis/scheduler stay alive."""
    client = _redis(plan)
    registry = _registry(client)
    before = _worker(registry, plan["deployment_id"])
    registry_field = next(key for key, item in registry.items() if item == before)
    owned_before = _owned_processes(process.pid)
    require(
        before["pid"]
        in [item["pid"] for item in owned_before.get("native-worker", [])],
        "Refusing to fault a worker not owned by this service",
    )
    require(
        Path(os.readlink(f"/proc/{before['pid']}/exe")).name == "native-worker",
        "Refusing to fault another executable",
    )
    os.kill(before["pid"], signal.SIGKILL)
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        require(
            process.poll() is None, "Service stack exited during worker-only recovery"
        )
        workers = {
            key: item
            for key, item in _registry(client).items()
            if key == registry_field
            or item.get("deployment_id") == plan["deployment_id"]
        }
        require(
            set(workers) == {registry_field}
            and workers[registry_field].get("deployment_id") == plan["deployment_id"],
            "Original durable native worker slot disappeared or changed during recovery",
        )
        worker = workers[registry_field]
        if (
            worker.get("worker_instance_id") != before["worker_instance_id"]
            and worker.get("status") == "available"
        ):
            after = worker
            try:
                code, report = http.request("/readyz")
                require(code == 200, "Replacement worker not ready")
                ready = assert_readiness(
                    report, target, deployment_id=plan["deployment_id"]
                )
                require(
                    ready["worker_instance_id"] == after["worker_instance_id"],
                    "Readiness still exposes old worker",
                )
            except (OSError, URLError, ValueError):
                time.sleep(0.1)
                continue
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("Native worker replacement/readmission deadline exceeded")
    for key in (
        "stream",
        "workflow_id",
        "executor_class",
        "device_uuid",
        "device_index",
        "deployment_id",
    ):
        require(
            before.get(key) == after.get(key),
            "Worker replacement changed stable " + key,
        )
    require(after["pid"] != before["pid"], "Worker replacement did not change PID")
    owned_after = _owned_processes(process.pid)
    essential = {
        name: items for name, items in owned_before.items() if name != "native-worker"
    }
    require(
        essential
        and all(owned_after.get(name) == items for name, items in essential.items()),
        "Worker recovery replaced an essential service process",
    )
    replay = exercise_request(
        http,
        target,
        fixture=fixture,
        fixture_index=0,
        output=output,
        timeout=request_timeout,
    )
    evidence = {
        "worker_before": _worker_identity(before),
        "worker_after": _worker_identity(after),
        "essential_processes_unchanged": essential,
        "post_recovery_request": replay,
        "scope": "idle worker SIGKILL; Redis/API/scheduler retained; no active-kernel claim",
    }
    write_json(output / "worker-recovery.json", evidence)
    return evidence


def _wait_until(function, *, timeout, message):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = function()
        if result:
            return result
        time.sleep(0.05)
    raise TimeoutError(message)


def _event_payloads(client, stream, run_id):
    return [
        json.loads(fields["payload"])
        for _, fields in client.xrange(stream)
        if fields.get("run_id") == run_id
    ]


def validate_fault_completion(kind, *, releases, results, pending, stderr):
    if kind == "deadline":
        require(
            len(releases) == len(results) == 1 and pending == 0,
            "Deadline must settle one terminal result and reservation release",
        )
        outcome = results[0]
        require(
            outcome.get("status") == "failed"
            and "deadline" in outcome.get("execution", {}).get("error", "").lower()
            and outcome.get("execution", {}).get("status") == "failed"
            and releases[0].get("status") == "failed",
            "Deadline published a successful or unrelated result",
        )
    else:
        require(
            not releases and not results and pending == 1,
            "Lease-cancelled stale owner published output or acknowledged the stolen delivery",
        )
        require(
            "ownership" in stderr.lower() or "lease" in stderr.lower(),
            "Lease cancellation lacks ownership-loss evidence",
        )
    return {
        "terminal_results": len(results),
        "reservation_releases": len(releases),
        "pending_count": pending,
        "no_late_success": True,
    }


def isolated_worker_fault(plan, target, fixture, *, kind, output, timeout):
    """Bounded stalled HTTPS input tests deadline/lease loss before native submit.

    This does not introduce an HTTP cancel endpoint or claim cancellation during a
    CUDA kernel. The worker binary, models and sealed bundle remain unchanged;
    only a disposable deployment's input source and deadline are fault controls.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import ssl
    from pnms_app.service import _process_environment

    require(kind in {"deadline", "lease_cancellation"}, "Unknown worker fault probe")
    output.mkdir(parents=True, exist_ok=False)
    certificate, key = output / "ca.pem", output / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    release_download, download_started = threading.Event(), threading.Event()

    class StalledInput(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"x")
            self.wfile.flush()
            download_started.set()
            release_download.wait(timeout + 30)
            try:
                self.wfile.write(b"xxx")
            except (OSError, ssl.SSLError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StalledInput)
    server.daemon_threads = True
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    child = None
    stream_log = None
    client = _redis(plan)
    prefix = "pnms-qa-" + uuid.uuid4().hex + ":"
    try:
        native = next(
            item for item in plan["processes"] if item["name"] == "native-worker"
        )
        config = json.loads(Path(native["argv"][1]).read_text())
        config.update(
            request_timeout_seconds=1 if kind == "deadline" else 60,
            input_cache=str(output / "input-cache"),
            artifact_root=str(output / "artifacts"),
            startup_cases=[],
        )
        config["input_artifacts"]["qa_stall"] = {
            "url": f"https://localhost:{server.server_port}/input",
            "sha256": hashlib.sha256(b"xxxx").hexdigest(),
            "size_bytes": 4,
        }
        config_path = output / "deployment.json"
        write_json(config_path, config)
        write_json(
            output / "release-inventory.json",
            {"bundle_digest": target["bundle_digest"]},
        )
        deployment = "fault-" + uuid.uuid4().hex
        environment = _process_environment(
            {
                **native["env"],
                "NATIVE_DEPLOYMENT_ID": deployment,
                "REDIS_STREAM_PREFIX": prefix,
                "SSL_CERT_FILE": str(certificate),
                "NO_PROXY": "localhost,127.0.0.1",
                "NATIVE_WORKER_SLOT": "0",
            }
        )
        environment.pop("NATIVE_DEPLOYMENT_OPERATION_ID", None)
        stream_log = (output / "worker.log").open("w")
        child = subprocess.Popen(
            [native["argv"][0], str(config_path)],
            env=environment,
            stdout=stream_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def admitted():
            require(
                child.poll() is None, "Isolated worker exited before fault admission"
            )
            return next(
                (
                    (field, worker)
                    for field, worker in _registry(client).items()
                    if worker.get("deployment_id") == deployment
                    and worker.get("status") == "available"
                ),
                None,
            )

        registry_field, worker = _wait_until(
            admitted, timeout=timeout, message="Isolated worker never admitted"
        )
        stream = worker["stream"]
        run_id = "fault-" + uuid.uuid4().hex
        # Original sealed fixture parameters; fixed upload bytes retained under owned ingress.
        artifacts = []
        for name, item in fixture_files(target, fixture).items():
            descriptor = fixture["files"][name]
            artifact_id = str(uuid.uuid4())
            ingress = Path(config["ingress_root"]) / run_id
            ingress.mkdir(parents=True, exist_ok=True)
            path = ingress / (artifact_id + "-" + name)
            path.write_bytes(item["path"].read_bytes())
            artifacts.append(
                {
                    "field_name": name,
                    "artifact_id": artifact_id,
                    "storage_path": str(path),
                    "media_type": item["media_type"],
                    "sha256": descriptor["sha256"],
                    "size_bytes": descriptor["size_bytes"],
                }
            )
        payload = {
            "workflow_id": config["workflow_id"],
            "parameters": fixture["request"],
            "runtime": {"kind": "cpp", "executor_class": config["executor_class"]},
            "resource_id": 0,
            "memory_mb": 1,
            "manifest_version": config.get("manifest_version"),
            "request": {
                "content_type": "multipart/form-data"
                if artifacts
                else "application/json",
                "input_artifacts": artifacts,
            },
        }
        message = client.xadd(
            stream,
            {"run_id": run_id, "payload": json.dumps(payload), "stage": "execute"},
        )
        require(
            download_started.wait(30), "Worker did not reach controlled stalled input"
        )
        groups = client.xinfo_groups(stream)
        require(len(groups) == 1, "Isolated execute group inventory differs")
        group = groups[0]["name"]
        original = client.xpending_range(stream, group, message, message, 1)
        require(len(original) == 1, "Fault injection did not own a pending delivery")
        if kind == "lease_cancellation":
            client.xclaim(
                stream, group, "qa-stolen-consumer", 0, [message], justid=True
            )
        _wait_until(
            lambda: child.poll() is not None,
            timeout=30,
            message="Faulted worker did not retire/drain",
        )
        release_download.set()
        # A held input is now free to finish; verify no success appears after owner retirement.
        time.sleep(0.5)
        results = _event_payloads(client, prefix + "results", run_id)
        releases = _event_payloads(client, prefix + "release", run_id)
        pending = client.xpending(stream, group)["pending"]
        stream_log.flush()
        stderr = (output / "worker.log").read_text()[-12000:]
        selected_registry = {
            field: item
            for field, item in _registry(client).items()
            if field == registry_field or item.get("deployment_id") == deployment
        }
        lease_key = (
            "native:worker-lease:" + hashlib.sha256(registry_field.encode()).hexdigest()
        )
        slot_evidence = {
            "registry_field": registry_field,
            "worker_before": worker,
            "selected_registry": selected_registry,
            "lease_key": lease_key,
            "owner_lease": client.get(lease_key),
        }
        write_json(output / "fault-registry.json", slot_evidence)
        evidence = validate_fault_completion(
            kind, releases=releases, results=results, pending=pending, stderr=stderr
        )
        require(child.returncode != 0, "Faulted native worker did not fail closed")
        require(
            set(selected_registry) == {registry_field},
            "Faulted worker lost or changed its original durable slot",
        )
        retired = selected_registry[registry_field]
        mutable = {"status", "model_cache", "heartbeat_at", "updated_at"}
        require(
            {key: value for key, value in worker.items() if key not in mutable}
            == {key: value for key, value in retired.items() if key not in mutable},
            "Faulted worker changed its original durable slot identity",
        )
        require(
            retired.get("status") == "warming",
            "Faulted worker retained admission",
        )
        require(
            retired.get("model_cache")
            == {
                "schema_version": 1,
                "scope": "process",
                "entries": [],
                "total_entries": 0,
                "warmup": {
                    "workflow_id": worker["workflow_id"],
                    "status": "warming",
                    "reason": "restart_pending",
                },
            },
            "Faulted worker retained usable models or lacks restart-pending warmup",
        )
        require(
            slot_evidence["owner_lease"] is None,
            "Faulted worker retained its live owner lease",
        )
        evidence.update(
            kind=kind,
            run_id=run_id,
            runtime_identity=target["runtime_identity"],
            bundle_digest=target["bundle_digest"],
            returncode=child.returncode,
            controlled_input_started=True,
            retired_slot=slot_evidence,
            scope="isolated native worker, HTTPS input admission before C++ call; no public HTTP cancel API or active-CUDA claim",
        )
        write_json(output / "fault.json", evidence)
        return evidence
    finally:
        release_download.set()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
                raise ValueError("Isolated worker cleanup exceeded deadline")
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if stream_log:
            stream_log.close()
        keys = list(client.scan_iter(match=prefix + "*"))
        if keys:
            client.delete(*keys)


def validate_release_evidence(target, cases):
    details = {
        case["name"].removeprefix(target["name"] + "."): case.get("details", {})
        for case in cases
        if case["name"].startswith(target["name"] + ".")
    }
    count = target["application_fixtures"]["case_count"]
    for index in range(count):
        entry = details.get(f"fixture.{index}", {})
        require(
            isinstance(entry.get("artifact_downloads"), dict)
            and isinstance(entry.get("output_paths"), list)
            and isinstance(entry.get("native_attempt"), str)
            and entry["native_attempt"],
            "Expanded fixture output/download evidence is missing",
        )
    overlap = details.get("overlap", {})
    requests = overlap.get("requests", [])
    require(
        overlap.get("both_admitted_before_poll") is True
        and overlap.get("output_isolation") == "passed"
        and len(requests) == 2
        and len({item.get("run_id") for item in requests}) == 2
        and all(
            isinstance(item.get("native_attempt"), str) and item["native_attempt"]
            for item in requests
        )
        and len({item.get("native_attempt") for item in requests}) == 2
        and all(isinstance(item.get("output_paths"), list) for item in requests)
        and not set(requests[0]["output_paths"]) & set(requests[1]["output_paths"]),
        "Overlapping request/isolation evidence differs",
    )
    for kind in ("gpu", "model"):
        negative = details.get(f"startup_{kind}_rejection", {})
        require(
            type(negative.get("returncode")) is int
            and negative["returncode"] != 0
            and negative.get("rejected_before_admission") is True
            and negative.get("worker_before", {}).get("pid")
            and negative.get("worker_before", {}).get("worker_instance_id")
            and negative.get("worker_before") == negative.get("worker_after"),
            "Startup-negative admission evidence differs",
        )
    for kind, operation in [
        ("deadline", "request_deadline"),
        ("lease_cancellation", "lease_cancellation"),
    ]:
        fault = details.get(operation, {})
        require(
            fault.get("kind") == kind
            and fault.get("bundle_digest") == target["bundle_digest"]
            and fault.get("runtime_identity") == target["runtime_identity"]
            and fault.get("no_late_success") is True
            and fault.get("controlled_input_started") is True
            and type(fault.get("returncode")) is int
            and fault["returncode"] != 0,
            "Native deadline/cancellation evidence differs",
        )
        require(
            (
                fault.get("terminal_results"),
                fault.get("reservation_releases"),
                fault.get("pending_count"),
            )
            == ((1, 1, 0) if kind == "deadline" else (0, 0, 1)),
            "Native deadline/cancellation terminal evidence differs",
        )
    recovery = details.get("worker_recovery", {})
    before, after = recovery.get("worker_before", {}), recovery.get("worker_after", {})
    require(
        before.get("pid")
        and after.get("pid")
        and before["pid"] != after["pid"]
        and before.get("worker_instance_id") != after.get("worker_instance_id")
        and all(
            before.get(key) == after.get(key)
            for key in (
                "stream",
                "workflow_id",
                "executor_class",
                "device_uuid",
                "device_index",
                "deployment_id",
            )
        )
        and recovery.get("essential_processes_unchanged")
        and recovery.get("post_recovery_request", {}).get("run_id"),
        "Worker-only recovery/readmission evidence differs",
    )


def run_target(ctx, target):
    root = ctx.root / target["name"]
    root.mkdir()
    owned = None
    log = None
    process = None
    launch = None
    deployment = ctx.args.deployment_id
    port = _port()
    base_url = ctx.args.base_url or f"http://127.0.0.1:{port}"
    http = HTTP(base_url, timeout=min(ctx.args.request_timeout, 15))
    try:

        def start():
            nonlocal owned, process, log, launch, deployment
            if ctx.args.base_url:
                require(deployment, "Attaching image QA requires --deployment-id")
                return {"owned": False, "base_url": base_url}
            environment = dict(ctx.environment)
            working_directory = ctx.args.repository
            for key in (
                "NATIVE_DEPLOYMENT_CONFIG",
                "RELEASE_IMAGE_DIGEST",
                "PNMS_IMAGE_CONFIG_DIGEST",
                "PNMS_BUNDLE_DIGEST",
                "NATIVE_DEPLOYMENT_OPERATION_ID",
                "PNMS_SERVICE_BIN_DIR",
            ):
                environment.pop(key, None)
            if ctx.args.mode == "image":
                deployment = "qa-" + uuid.uuid4().hex
                identity = target["runtime_identity"]
                environment[
                    "RELEASE_IMAGE_DIGEST"
                    if identity["kind"] == "registry-image"
                    else "PNMS_IMAGE_CONFIG_DIGEST"
                ] = identity["value"]
                environment.update(
                    PORT=str(port),
                    PNMS_OUTPUT_ROOT=str(root / "artifacts"),
                    NATIVE_DEPLOYMENT_ID=deployment,
                    NATIVE_WORKER_SLOT="0",
                    PNMS_SERVICE_BIN_DIR=str(
                        ctx.args.service_bin_dir or "/opt/pnms/bin"
                    ),
                )
                if target["runtime"] == "python":
                    command = [
                        "/usr/bin/dumb-init",
                        "--",
                        "python3",
                        "/opt/pnms/python_release.py",
                        "start",
                    ]
                    working_directory = Path("/app")
                    environment.update(
                        SERVER_PORT=str(port),
                        HEALTH_STUB_ENABLED="false",
                        ARTIFACT_DIR=str(root / "artifacts"),
                        DEFAULT_OUTPUT_DIR=str(root / "outputs"),
                    )
                else:
                    command = [str(ctx.args.python), "-m", "pnms_app.image_entrypoint"]
            else:
                command = [
                    str(ctx.args.python),
                    str(ctx.args.repository / "pnms"),
                    "serve",
                    target["project"],
                    "--repository",
                    str(ctx.args.repository),
                    "--dev",
                    "--port",
                    str(port),
                    "--output",
                    str(root / "service"),
                ]
                if target["runtime"] == "cpp":
                    launch = root / "service/launch.json"
                    if ctx.args.service_bin_dir:
                        environment["PNMS_SERVICE_BIN_DIR"] = str(
                            ctx.args.service_bin_dir
                        )
                        command.append("--skip-build")
                else:
                    # The unchanged Python launcher resolves repository/target/debug.
                    environment.pop("CARGO_TARGET_DIR", None)
            write_json(
                root / "start.json", {"argv": command, "cwd": str(working_directory)}
            )
            log = (root / "service.log").open("w")
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            owned = OwnedService(
                process, timeout=ctx.args.shutdown_timeout, process_group=True
            )
            return {"owned": True, "pid": process.pid, "base_url": base_url}

        ctx.case(target["name"] + ".start", start)
        first_ready = ctx.case(
            target["name"] + ".readiness",
            lambda: _wait_ready(
                http,
                target,
                process=process,
                deployment_id=deployment,
                timeout=ctx.args.startup_timeout,
                launch=launch,
            ),
        )
        initial = first_ready.get("native", {})
        if target["runtime"] == "cpp":
            deployment = initial["deployment_id"]
        fixtures = (
            application_fixture_cases(target)
            if "application_fixtures" in target
            else None
        )
        requests = (
            [(f"fixture.{index}", None) for index in range(len(fixtures))]
            if target.get("release_acceptance")
            else [
                ("first_request", [11, -7, 0.125, 2.75]),
                ("repeat_request", [-4, 0, 32, 0.75]),
            ]
        )
        for label, values in requests:
            if target["runtime"] == "python":
                values = 17 if label == "first_request" else 9
            fixture_index = (
                int(label.split(".")[1])
                if label.startswith("fixture.")
                else 0
                if label == "first_request"
                else 1
            )
            if fixtures:
                fixture_index = min(fixture_index, len(fixtures) - 1)
            ctx.case(
                target["name"] + "." + label,
                lambda values=values, label=label, index=fixture_index: (
                    exercise_request(
                        http,
                        target,
                        values=values,
                        output=root / label,
                        timeout=ctx.args.request_timeout,
                        fixture=fixtures[index] if fixtures else None,
                        fixture_index=index if fixtures else None,
                    )
                ),
            )

        if target.get("release_acceptance"):
            ctx.case(
                target["name"] + ".overlap",
                lambda: exercise_overlap(
                    http,
                    target,
                    fixtures,
                    output=root / "overlap",
                    timeout=ctx.args.request_timeout,
                ),
            )

        def reject_schema():
            body = (
                {"parameters": []}
                if fixtures
                else {"parameters": {"values": [1, 2, 3]}}
                if target["runtime"] == "cpp"
                else {"parameters": {"value": "wrong-type"}}
            )
            operation = quote(target.get("operation", "run"), safe="")
            path = f"/v1/infer/{quote(target['workflow'], safe='')}/{operation}"
            if target.get("release_acceptance") and (
                target.get("input_files")
                or target.get("content_types") == ["multipart/form-data"]
            ):
                code, report = http.request(
                    path, body=body, files=fixture_files(target, fixtures[0])
                )
            elif target.get("ingress") == "multipart/form-data":
                code, report = http.request(
                    path,
                    body=body["parameters"],
                    files={"features": root / "first_request/input.f32"},
                )
            else:
                code, report = http.request(path, body=body)
            write_json(
                root / "schema-rejection.json",
                {"http_status": code, "response": report},
            )
            require(
                code in (400, 422) and "run_id" not in report,
                "Malformed request was admitted or rejected unexpectedly",
            )
            return {"http_status": code}

        def reject_workflow():
            code, report = http.request(
                "/v1/infer/unknown-" + uuid.uuid4().hex + "/run",
                body={"parameters": {}},
            )
            write_json(
                root / "unknown-workflow.json",
                {"http_status": code, "response": report},
            )
            require(
                code == 404, "Unknown workflow was not rejected by the public route"
            )
            return {"http_status": code}

        ctx.case(target["name"] + ".schema_rejection", reject_schema)
        ctx.case(target["name"] + ".unknown_workflow", reject_workflow)

        def resident():
            code, current = http.request("/readyz")
            require(code == 200, "Service lost readiness after repeated requests")
            native = assert_readiness(current, target, deployment_id=deployment)
            if target["runtime"] == "cpp":
                require(
                    native["worker_instance_id"] == initial["worker_instance_id"],
                    "Native resident worker changed between sequential requests",
                )
            write_json(root / "final-readiness.json", current)
            return {
                "worker_instance_id": native.get("worker_instance_id"),
                "native_residency": "passed"
                if target["runtime"] == "cpp"
                else "not_applicable",
            }

        ctx.case(target["name"] + ".resident_worker", resident)
        if target.get("release_acceptance"):
            plan_path, plan = image_launch_plan(root / "artifacts", target)
            for kind in ("gpu", "model"):
                ctx.case(
                    target["name"] + f".startup_{kind}_rejection",
                    lambda kind=kind: reject_native_startup(
                        plan, target, kind=kind, output=root
                    ),
                )
            for kind in ("deadline", "lease_cancellation"):
                operation = "request_deadline" if kind == "deadline" else kind
                ctx.case(
                    target["name"] + "." + operation,
                    lambda kind=kind: isolated_worker_fault(
                        plan,
                        target,
                        fixtures[0],
                        kind=kind,
                        output=root / kind,
                        timeout=ctx.args.startup_timeout,
                    ),
                )
            ctx.case(
                target["name"] + ".worker_recovery",
                lambda: recover_worker(
                    process,
                    http,
                    target,
                    initial,
                    plan,
                    fixtures[0],
                    output=root / "recovery",
                    startup_timeout=ctx.args.startup_timeout,
                    request_timeout=ctx.args.request_timeout,
                ),
            )
    finally:
        try:
            if owned is not None:
                ctx.case(target["name"] + ".shutdown", owned.close)
            elif ctx.args.base_url:
                ctx.case(
                    target["name"] + ".shutdown",
                    lambda: {"owned": False, "external_service_left_running": True},
                )
        finally:
            if log is not None:
                log.close()


def extract_summary(output):
    begin = output.rfind(SUMMARY_BEGIN)
    end = output.find(SUMMARY_END, begin + len(SUMMARY_BEGIN))
    require(begin >= 0 and end > begin, "No complete pnms HTTP QA summary in logs")
    report = json.loads(output[begin + len(SUMMARY_BEGIN) : end].strip())
    require(isinstance(report, dict), "HTTP QA summary must be an object")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("host", "image"), default="host")
    result.add_argument("--image-runtime", choices=("cpp", "python"), default="cpp")
    result.add_argument(
        "--application-fixtures",
        action="store_true",
        help="Native image QA: replay the first two sealed JSON application fixtures",
    )
    result.add_argument(
        "--release-acceptance",
        action="store_true",
        help="Promotable native image gate: all sealed fixtures, concurrency, artifacts and worker fault probes",
    )
    result.add_argument("--python-workflow")
    result.add_argument("--python-plugin-digest")
    result.add_argument("--python-receipt-sha256")
    result.add_argument("--build-receipt", type=Path, action="append", default=[])
    result.add_argument("--local-summary", type=Path)
    result.add_argument("--mount-root", type=Path, default=Path("/outputs"))
    result.add_argument("--repository", type=Path, default=Path("/workspace"))
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--bundle-root", type=Path, default=Path("/opt/physicsnemo-release")
    )
    result.add_argument("--service-bin-dir", type=Path)
    result.add_argument("--base-url")
    result.add_argument("--deployment-id")
    result.add_argument("--include-python", action="store_true")
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--run-id", required=True)
    result.add_argument("--expected-source-sha", required=True)
    result.add_argument("--image-digest", required=True)
    result.add_argument("--expected-runner-sha256")
    result.add_argument("--startup-timeout", type=float, default=1200)
    result.add_argument("--request-timeout", type=float, default=120)
    result.add_argument("--shutdown-timeout", type=float, default=240)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_id), "Invalid QA run ID")
    require(
        re.fullmatch(r"[a-f0-9]{40}", args.expected_source_sha),
        "Full source revision required",
    )
    _expected_identity(args.image_digest)
    require(
        not args.expected_runner_sha256
        or sha256(Path(__file__)) == args.expected_runner_sha256,
        "HTTP runner differs from the selected controller source",
    )
    require(
        all(
            math.isfinite(value) and value > 0
            for value in (
                args.startup_timeout,
                args.request_timeout,
                args.shutdown_timeout,
            )
        ),
        "QA deadlines must be finite and positive",
    )
    require(
        not args.base_url or args.mode == "image",
        "Existing-service mode applies to exact-image QA",
    )
    require(
        not args.include_python or args.mode == "host",
        "Python coverage uses the host development flow",
    )
    require(
        args.mode == "image" or args.image_runtime == "cpp",
        "Image runtime applies to image QA only",
    )
    require(
        not args.application_fixtures
        or args.mode == "image"
        and args.image_runtime == "cpp",
        "Application fixtures apply only to native image QA",
    )
    require(
        not args.release_acceptance
        or args.application_fixtures
        and args.mode == "image"
        and args.image_runtime == "cpp"
        and not args.base_url,
        "Release acceptance requires an owned native image and application fixtures",
    )
    require(
        not args.base_url or args.image_runtime == "cpp",
        "Python image QA owns its release process",
    )
    require(
        args.mode != "image"
        or args.base_url
        or args.bundle_root == Path("/opt/physicsnemo-release"),
        "Owned image QA must use the baked bundle at /opt/physicsnemo-release",
    )
    args.repository = args.repository.resolve()
    if args.mode == "host" and (args.repository / "application-cli/src").is_dir():
        sys.path.insert(0, str(args.repository / "application-cli/src"))
    ctx = Context(args)
    code = 1
    try:
        if args.mode == "host":
            require(
                bool(args.build_receipt) != bool(args.local_summary),
                "Host HTTP QA requires --local-summary or --build-receipt",
            )
            if args.local_summary:
                candidates, evidence = local_candidates(
                    args.local_summary,
                    mount_root=args.mount_root,
                    source_sha=args.expected_source_sha,
                    image_digest=args.image_digest,
                )
                ctx.report["input_local_summary"] = evidence
                configure_local_environment(ctx, evidence)
                receipts = [item["build_receipt"] for item in candidates]
            else:
                candidates = None
                receipts = args.build_receipt
            targets = [
                load_host_target(
                    mounted_path(path, args.mount_root),
                    source_sha=args.expected_source_sha,
                )
                for path in receipts
            ]
            if candidates is not None:
                for target, candidate in zip(targets, candidates, strict=True):
                    require(
                        target["bundle_digest"] == candidate["bundle_digest"]
                        and target["build_id"] == candidate["build_id"],
                        "HTTP target differs from local QA handoff candidate",
                    )
        else:
            require(
                not args.build_receipt and not args.local_summary,
                "Image QA consumes its baked bundle, not a source build receipt",
            )
            targets = [
                _python_image_target(
                    Path("/opt/pnms"),
                    source_sha=args.expected_source_sha,
                    runtime_identity=_expected_identity(args.image_digest),
                    workflow=args.python_workflow,
                    plugin_digest=args.python_plugin_digest,
                    receipt_sha256=args.python_receipt_sha256,
                )
                if args.image_runtime == "python"
                else _bundle_target(
                    args.bundle_root,
                    source_sha=args.expected_source_sha,
                    runtime_identity=_expected_identity(args.image_digest),
                    application_fixtures=args.application_fixtures,
                    release_acceptance=args.release_acceptance,
                )
            ]
        if args.include_python:
            ctx.environment.update(
                PYO3_PYTHON=str(args.python),
                PHYSICSNEMO_SERVE_PYTHON_EXECUTABLE=str(args.python),
            )
            ctx.command(
                "python-bootstrap",
                [
                    args.python,
                    args.repository / "scripts/plugin_dev.py",
                    "bootstrap-python",
                ],
                timeout=args.startup_timeout,
                json_output=False,
            )
            project = ctx.root / "http-python"
            ctx.command(
                "python-init",
                [
                    args.python,
                    args.repository / "pnms",
                    "init",
                    project,
                    "--repository",
                    args.repository,
                    "--runtime",
                    "python",
                ],
            )
            targets.append(
                {
                    "name": "python.generated",
                    "runtime": "python",
                    "workflow": "http-python",
                    "source_sha": args.expected_source_sha,
                    "project": str(project),
                }
            )
        require(
            len({target["name"] for target in targets}) == len(targets),
            "Duplicate HTTP QA target",
        )
        ctx.report.update(targets=targets, expected_cases=expected_cases(targets))

        def environment():
            metadata = Path("/opt/physicsnemo-qa/source.json")
            if args.mode == "host" and metadata.is_file():
                require(
                    json.loads(metadata.read_text())["source_sha"]
                    == args.expected_source_sha,
                    "Prepared image source differs",
                )
            return {
                "runner_sha256": sha256(Path(__file__)),
                "source_sha": args.expected_source_sha,
                "job_image_digest": args.image_digest,
                "nested_docker": False,
                "target_count": len(targets),
                **(
                    {"python": str(args.python), "input_profile": "fresh-venv"}
                    if ctx.report.get("input_local_summary", {}).get("profile")
                    == "fresh-venv"
                    else {}
                ),
                **(
                    {"python_release": targets[0]}
                    if args.mode == "image" and args.image_runtime == "python"
                    else {}
                ),
            }

        ctx.case("environment", environment)
        for target in targets:
            run_target(ctx, target)
        ctx.report["status"] = "passed"
        validate_summary(
            ctx.report,
            run_id=args.run_id,
            source_sha=args.expected_source_sha,
            image_digest=args.image_digest,
            targets=targets,
        )
        code = 0
    except BaseException as error:
        ctx.report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        ctx.save()
        print(SUMMARY_BEGIN, flush=True)
        print(json.dumps(ctx.report, sort_keys=True, allow_nan=False), flush=True)
        print(SUMMARY_END, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
