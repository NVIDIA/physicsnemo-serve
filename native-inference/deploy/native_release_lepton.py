"""Finite Lepton jobs and private, authenticated artifact handoff.

Only resources created by this invocation are removed. Bearer tokens stay in
memory or an owner-only temporary header file and never enter release evidence.
NATIVE_RELEASE_ARTIFACT_PROXY optionally routes only curl artifact HTTPS traffic
through an explicit proxy; Lepton CLI/API traffic keeps its normal route.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from native_release import image_digest, read_json, require, write_json
from native_release_artifacts import extract


BUILD_BEGIN = "NATIVE_RELEASE_BUILD_SUMMARY_BEGIN"
BUILD_END = "NATIVE_RELEASE_BUILD_SUMMARY_END"


def _object(output, start="{", end="}"):
    begin, finish = output.find(start), output.rfind(end)
    require(begin >= 0 and finish >= begin, "Lepton response has no JSON value")
    return json.loads(output[begin : finish + 1])


def extract_summary(output, *, markers=(BUILD_BEGIN, BUILD_END)):
    begin, end = markers
    first = output.rfind(begin)
    last = output.find(end, first + len(begin))
    require(first >= 0 and last >= 0, "Lepton job has no complete release summary")
    result = json.loads(output[first + len(begin) : last].strip())
    require(isinstance(result, dict), "release summary must be an object")
    return result


def validate_summary(
    report, *, required_cases, run_id, source_sha, image_digest, profile
):
    import math

    for key, expected in {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "source_sha": source_sha,
        "image_digest": image_digest,
        "expected_cases": required_cases,
    }.items():
        require(
            type(report.get(key)) is type(expected) and report[key] == expected,
            f"release job {key} differs",
        )
    cases = report.get("cases", [])
    require(
        [case.get("name") for case in cases] == required_cases,
        "release job required gate inventory differs",
    )
    for case in cases:
        seconds = case.get("seconds")
        require(
            case.get("status") == "passed"
            and type(seconds) in (int, float)
            and math.isfinite(seconds)
            and seconds >= 0,
            "every release gate must execute and pass",
        )


def _lifecycle():
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qa.scripts import run_lepton_native_inference_qa

    return run_lepton_native_inference_qa


def _arguments(*, image, source_sha, run_id, artifact_dir, config=None):
    lifecycle = _lifecycle()
    values = [
        "--image",
        image,
        "--expected-source-sha",
        source_sha,
        "--run-id",
        run_id,
        "--artifact-dir",
        str(artifact_dir),
        "--job-timeout",
        "7200",
    ]
    for name, value in (config or {}).items():
        if name in {
            "workspace_id",
            "workspace_url",
            "node_group",
            "pull_secret",
            "resource_shape",
            "nfs_mount_base",
            "lustre_dir",
            "lustre_storage",
            "mount_target",
            "job_timeout",
            "poll_interval",
            "command_timeout",
        }:
            values += ["--" + name.replace("_", "-"), str(value)]
    args = lifecycle.build_parser().parse_args(values)
    lifecycle.validate_args(args)
    require(
        60 < args.job_timeout <= 7200,
        "release job timeout must exceed60 and be at most7200 seconds",
    )
    return args


def validate_controller_config(resolved):
    """Validate location/deadlines without Docker, network calls or a GPU job."""
    args = _arguments(
        image=resolved["images"]["builder"],
        source_sha=resolved["source_sha"],
        run_id="release-config-check",
        artifact_dir=Path.cwd(),
        config=resolved.get("lepton"),
    )
    return {
        key: getattr(args, key)
        for key in (
            "workspace_id",
            "workspace_url",
            "node_group",
            "pull_secret",
            "resource_shape",
            "nfs_mount_base",
            "lustre_dir",
            "lustre_storage",
            "mount_target",
            "job_timeout",
            "poll_interval",
            "command_timeout",
        )
    }


def run_job(
    *,
    image,
    source_sha,
    run_id,
    mode,
    required_cases,
    artifact_dir,
    config=None,
    process_runner=None,
    node_id=None,
):
    lifecycle = _lifecycle()
    args = _arguments(
        image=image,
        source_sha=source_sha,
        run_id=run_id,
        artifact_dir=artifact_dir,
        config=config,
    )
    root = Path(artifact_dir) / run_id

    def command(arguments, job_name):
        result = lifecycle.build_job_command(arguments, job_name)
        output = f"{arguments.mount_target.rstrip('/')}/native-inference/{run_id}"
        script = (
            "native_release_job.py"
            if mode == "compile"
            else "native_release_cfd_job.py"
            if mode == "cfd_compile"
            else "native_release_cfd_http_job.py"
            if mode == "cfd_http"
            else "native_release_http_job.py"
        )
        runner = [
            "exec",
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=60s",
            f"{int(arguments.job_timeout) - 60}s",
            "/opt/nvidia/nvidia_entrypoint.sh",
            "python3",
            "/opt/native-release/" + script,
            "--output",
            output,
            "--run-id",
            run_id,
            "--image",
            image,
        ]
        result[result.index("--command") + 1] = shlex.join(runner)
        result += ["--shared-memory-size", "16384"]
        if node_id is not None:
            require(
                isinstance(node_id, str) and node_id and not node_id.startswith("-"),
                "invalid qualified storage node",
            )
            result += ["--node-id", node_id]
        return result

    def validated(report, **expected):
        validate_summary(report, required_cases=required_cases, **expected)
        controller = read_json(root / "summary.json")
        env = {
            **os.environ,
            "LEPTON_WORKSPACE_ID": args.workspace_id,
            "COLUMNS": "1000000",
            "NO_COLOR": "1",
        }
        if args.workspace_url:
            env["LEPTON_WORKSPACE_URL"] = args.workspace_url
        code, output = lifecycle.lep(
            ["job", "nodes", "-i", controller["job_id"]],
            env=env,
            timeout=min(30, args.command_timeout),
        )
        nodes = _object(output, "[", "]") if code == 0 else []
        require(
            len(set(nodes)) == 1 and isinstance(nodes[0], str),
            "completed build must identify one worker node before cleanup",
        )
        write_json(
            root / "job-placement.json",
            {"job_id": controller["job_id"], "node_id": nodes[0]},
        )

    # The observer runs during summary validation, before existing finally
    # removes the owned job. Component lifecycle defaults are untouched.
    if process_runner is None:
        code = lifecycle.run(
            args,
            command_builder=command,
            summary_extractor=extract_summary,
            summary_validator=validated,
            historical_first=True,
        )
    else:
        # Named process seam is for CPU characterization tests only. It must
        # provide the same lifecycle artifacts; it never qualifies a release.
        code = process_runner.run_lepton_job(args, command, required_cases)
    record = read_json(root / "summary.json")
    require(
        code == 0
        and record.get("status") == "passed"
        and record.get("cleanup_errors") == [],
        "Lepton job or owned cleanup failed",
    )
    summary = read_json(root / "job-summary.json")
    validate_summary(
        summary,
        required_cases=required_cases,
        run_id=run_id,
        source_sha=source_sha,
        image_digest=image,
        profile=args.profile,
    )
    placement = read_json(root / "job-placement.json")
    require(
        placement.get("job_id") == record["job_id"],
        "artifact placement belongs to a different job",
    )
    record["node_id"] = placement["node_id"]
    write_json(root / "summary.json", record)
    return {"controller": record, "summary": summary, "args": args}


class _RetryableArtifactError(ValueError):
    """A transient curl transport failure, without request credentials."""


def _curl(runner, url, output, *, headers=None, timeout=60, deadline=None):
    process_timeout = timeout + 10
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("artifact archive download timed out")
        timeout = min(timeout, remaining)
        process_timeout = min(timeout + 10, remaining)
    argv = [
        "curl",
        "--silent",
        "--show-error",
        "--connect-timeout",
        "15",
        "--max-time",
        str(timeout),
        "--output",
        str(output),
        "--write-out",
        "%{http_code}",
    ]
    proxy = os.environ.get("NATIVE_RELEASE_ARTIFACT_PROXY")
    if proxy:
        require(
            urlsplit(proxy).scheme in {"http", "https", "socks5", "socks5h"}
            and urlsplit(proxy).netloc
            and not any(c.isspace() for c in proxy),
            "artifact proxy requires an explicit HTTP(S) or SOCKS URL",
        )
        argv += ["--proxy", proxy]
    if headers is not None:
        argv += ["--header", "@" + str(headers)]
    try:
        result = runner.run(
            [*argv, url], timeout=process_timeout, sensitive=True, check=False
        )
    except (subprocess.TimeoutExpired, TimeoutError):
        raise _RetryableArtifactError("private artifact transport timed out") from None
    except RuntimeError as error:
        # ProcessRunner deliberately sanitizes sensitive subprocess timeouts.
        # A missing executable or another controller error is not retryable.
        if str(error) == "sensitive release command timed out":
            raise _RetryableArtifactError(
                "private artifact transport timed out"
            ) from None
        raise
    if result.returncode in {5, 6, 7, 18, 28, 35, 52, 55, 56, 92}:
        # DNS/connect, partial transfer, timeout, TLS handshake, empty reply,
        # send/receive and HTTP/2 transport failures. Certificate errors and
        # local configuration/filesystem failures remain permanent.
        raise _RetryableArtifactError("private artifact transport was interrupted")
    require(
        result.returncode == 0 and re.fullmatch(r"[0-9]{3}", result.stdout.strip()),
        "private artifact endpoint is unreachable",
    )
    return int(result.stdout.strip())


def _download_archive(runner, url, output, *, headers, deadline):
    """Restart transient downloads within the original transfer deadline."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("artifact archive download timed out")
        try:
            status = _curl(
                runner,
                url,
                output,
                headers=headers,
                timeout=remaining,
                deadline=deadline,
            )
        except _RetryableArtifactError:
            pass
        else:
            if time.monotonic() >= deadline:
                raise TimeoutError("artifact archive download timed out")
            if status == 200:
                return
            require(
                status in {429, 502, 503, 504},
                f"artifact download failed (HTTP {status})",
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("artifact archive download timed out")
        time.sleep(min(5, remaining))


def fetch_artifact(*, image, job, expected, output, runner, timeout=1200):
    image_digest(image)
    require(0 < timeout <= 1800, "finite artifact export timeout required")
    args = job["args"]
    controller = job["controller"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    name = "pn-native-export-" + secrets.token_hex(6)
    token = secrets.token_urlsafe(48)
    created = False
    result = {
        "schema_version": 1,
        "status": "failed",
        "name": name,
        "cpu_only": True,
        "image": image,
        "source_sha": expected["source_sha"],
        "job_id": controller["job_id"],
        "cleanup_errors": [],
        "proxy_configured": bool(os.environ.get("NATIVE_RELEASE_ARTIFACT_PROXY")),
    }
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_ID": args.workspace_id,
        "NO_COLOR": "1",
        "COLUMNS": "1000000",
        "TERM": "dumb",
    }
    if args.workspace_url:
        env["LEPTON_WORKSPACE_URL"] = args.workspace_url
    workspace_url = runner.run(
        ["lep", "workspace", "url"], env=env, timeout=30
    ).stdout.strip()
    origin = urlsplit(workspace_url)
    require(
        origin.scheme == "https" and origin.netloc,
        "HTTPS workspace API origin required",
    )
    # The API rejects DELETE without a matching Origin on some enterprise hosts.
    # Use the supported CLI header override, never persist an auth-token request.
    env["LEPTON_DEBUG_HEADERS"] = "Origin=" + origin.scheme + "://" + origin.netloc
    node = controller.get("node_id")
    require(
        isinstance(node, str) and node,
        "artifact export requires the worker node captured before job cleanup",
    )
    mount = f"{args.nfs_mount_base.rstrip('/')}/{args.lustre_dir}:{args.mount_target}:node-nfs:{args.lustre_storage}"
    remote = controller["remote_artifacts"]
    command = shlex.join(
        [
            "exec",
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=15s",
            str(timeout) + "s",
            "python3",
            "/opt/native-release/native_release_artifacts.py",
            "--archive",
            remote + "/bundle.tar",
            "--manifest",
            remote + "/manifest.json",
            "--deadline",
            str(timeout),
        ]
    )
    argv = [
        "lep",
        "deployment",
        "create",
        "--name",
        name,
        "--container-image",
        image,
        "--container-port",
        "8080",
        "--container-command",
        command,
        "--resource-shape",
        "cpu.small",
        "--replicas-static",
        "1",
        "--mount",
        mount,
        "--node-group",
        args.node_group,
        "--node-id",
        node,
        "--image-pull-secrets",
        args.pull_secret,
        "--tokens",
        token,
        "--visibility",
        "private",
        "--initial-delay-seconds",
        "600",
    ]
    try:
        # Never log this argv or CLI output: it contains the ephemeral token.
        created = True  # Ambiguous creation outcomes still require owned cleanup.
        runner.run(argv, env=env, timeout=120, sensitive=True)
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryDirectory(prefix="native-release-export-") as temporary:
            scratch = Path(temporary)
            headers = scratch / "headers"
            fd = os.open(headers, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write("Authorization: Bearer " + token + "\n")
            manifest_path = output / "manifest.json"
            endpoint = None
            while time.monotonic() < deadline:
                # The endpoint spec may include tokens. Keep it only in the
                # private temporary directory; stdout is never evidence.
                response = runner.run(
                    ["lep", "deployment", "get", "--name", name],
                    env=env,
                    timeout=30,
                    sensitive=True,
                    check=False,
                )
                if response.returncode == 0:
                    deployment = _object(response.stdout)
                    endpoint = (
                        deployment.get("status", {})
                        .get("endpoint", {})
                        .get("external_endpoint")
                    )
                    if endpoint:
                        endpoint = (
                            endpoint
                            if endpoint.startswith("https://")
                            else "https://" + endpoint
                        )
                        require(
                            urlsplit(endpoint).scheme == "https",
                            "artifact endpoint requires HTTPS",
                        )
                        try:
                            status = _curl(
                                runner,
                                endpoint.rstrip("/") + "/manifest.json",
                                manifest_path,
                                headers=headers,
                                timeout=min(
                                    30, max(1, int(deadline - time.monotonic()))
                                ),
                            )
                            if status == 200:
                                break
                        except ValueError:
                            pass
                time.sleep(min(5, max(0, deadline - time.monotonic())))
            else:
                raise TimeoutError("private CPU artifact export readiness timed out")
            manifest = read_json(manifest_path)
            for key, value in expected.items():
                require(
                    manifest.get(key) == value,
                    f"artifact transport {key} differs from completed job",
                )
            require(
                0 < manifest["archive"]["size_bytes"] <= 16 * 1024**3,
                "artifact transport size is invalid",
            )
            anonymous = _curl(
                runner,
                endpoint.rstrip("/") + "/manifest.json",
                scratch / "anonymous",
                timeout=30,
            )
            require(
                anonymous in {401, 403},
                "artifact endpoint did not reject anonymous access",
            )
            _download_archive(
                runner,
                endpoint.rstrip("/") + "/bundle.tar",
                output / "bundle.tar",
                headers=headers,
                deadline=deadline,
            )
            extracted = extract(output / "bundle.tar", manifest, output / "extracted")
            result.update(
                status="passed",
                archive=manifest["archive"],
                anonymous_status=anonymous,
                extracted=str(extracted),
                node_id=node,
            )
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        if created:
            try:
                runner.run(
                    [
                        "lep",
                        "deployment",
                        "update",
                        "--name",
                        name,
                        "--min-replicas",
                        "0",
                    ],
                    env=env,
                    timeout=60,
                    sensitive=True,
                    check=False,
                )
                deletion = runner.run(
                    ["lep", "deployment", "remove", "--name", name],
                    env=env,
                    timeout=60,
                    sensitive=True,
                    check=False,
                )
                require(
                    deletion.returncode == 0, "CPU exporter deletion was not confirmed"
                )
                result["deployment_deleted"] = True
            except BaseException as error:
                result.update(status="failed", deployment_deleted=False)
                result["cleanup_errors"].append(
                    type(error).__name__ + ": " + str(error)
                )
        write_json(output / "transfer-summary.json", result)
    require(
        result["status"] == "passed" and not result["cleanup_errors"],
        "artifact transfer cleanup failed",
    )
    return result
