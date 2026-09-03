# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_workflow.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def _fake_tools(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    nvidia_smi_pid = tmp_path / "nvidia-smi.pid"

    _write_executable(
        bin_dir / "nvidia-smi",
        r"""
        #!/usr/bin/env bash
        set -eu

        for argument in "$@"; do
            if [ "$argument" = "--query-gpu=index" ]; then
                printf '%b' "${FAKE_GPU_LIST:-0\n}"
                exit 0
            fi
        done

        output_file=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                -f)
                    output_file=$2
                    shift 2
                    ;;
                *)
                    shift
                    ;;
            esac
        done

        printf '%s\n' "$$" > "${NVIDIA_SMI_PID_FILE:?}"
        cat > "$output_file" <<'EOF'
        timestamp, name, pstate, utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]
        2026/09/02 12:00:00, NVIDIA H100, P0, 10 %, 20 %, 1000 MiB, 81920 MiB
        2026/09/02 12:00:01, NVIDIA H100, P0, 30 %, 40 %, 1200 MiB, 81920 MiB
        EOF

        trap 'exit 0' TERM INT
        while true; do
            sleep 1
        done
        """,
    )
    _write_executable(
        bin_dir / "curl",
        r"""
        #!/usr/bin/env bash
        set -eu

        url=""
        for argument in "$@"; do
            case "$argument" in
                http://*|https://*)
                    url=$argument
                    ;;
            esac
            printf '%s\n' "$argument" >> "${CURL_LOG:?}"
        done
        printf '%s\n' "--request--" >> "${CURL_LOG:?}"

        case "$url" in
            */run)
                printf '{"run_id":"run-test-1","status":"queued"}\n202'
                ;;
            */run-test-1/status)
                printf '{"status":"%s"}\n200' "${FAKE_FINAL_STATUS:-succeeded}"
                ;;
            *)
                printf '{"error":"unexpected URL"}\n404'
                ;;
        esac
        """,
    )
    return bin_dir, curl_log, nvidia_smi_pid


def _profile_command(tmp_path: Path, output_dir: Path) -> list[str]:
    request_path = tmp_path / "request.json"
    request_path.write_text('{"models":["domino_surface"]}\n', encoding="utf-8")
    return [
        str(PROFILE_SCRIPT),
        "--wf_name",
        "physicsnemo-cfd-surface-benchmark",
        "--wf_json",
        str(request_path),
        "--output_dir",
        str(output_dir),
        "--poll_interval_seconds",
        "0",
        "--timeout_seconds",
        "5",
    ]


def _profile_environment(
    bin_dir: Path, curl_log: Path, nvidia_smi_pid: Path
) -> dict[str, str]:
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "CURL_LOG": str(curl_log),
        "NVIDIA_SMI_PID_FILE": str(nvidia_smi_pid),
    }
    env.pop("EP_TOKEN", None)
    env.pop("LEPTON_ENDPOINT_TOKEN", None)
    return env


def test_profile_workflow_collects_scheduler_profile(tmp_path: Path):
    bin_dir, curl_log, nvidia_smi_pid = _fake_tools(tmp_path)
    output_dir = tmp_path / "profile-output"

    result = subprocess.run(
        _profile_command(tmp_path, output_dir),
        cwd=tmp_path,
        env=_profile_environment(bin_dir, curl_log, nvidia_smi_pid),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    profile = json.loads(
        (output_dir / "profiles_run-test-1.json").read_text(encoding="utf-8")
    )
    assert profile == {
        "profiles": [
            {
                "workflow": "physicsnemo-cfd-surface-benchmark",
                "gpus.used": 1,
                "average": {
                    "utilization.gpu": "20.00 %",
                    "utilization.memory": "30.00 %",
                },
                "peak": {
                    "utilization.gpu": "30 %",
                    "utilization.memory": "40 %",
                    "memory.used": "1200 MiB",
                    "memory.total": "81920 MiB",
                },
            }
        ]
    }
    assert (output_dir / "profile_run-test-1.csv").is_file()
    assert (output_dir / "outputs_run-test-1.txt").is_file()

    requests = curl_log.read_text(encoding="utf-8")
    assert (
        "http://127.0.0.1:8080/v1/infer/physicsnemo-cfd-surface-benchmark/run"
    ) in requests
    assert (
        "http://127.0.0.1:8080/v1/infer/"
        "physicsnemo-cfd-surface-benchmark/run-test-1/status"
    ) in requests
    assert "Authorization: Bearer" not in requests


def test_profile_workflow_preserves_failure_and_stops_sampler(tmp_path: Path):
    bin_dir, curl_log, nvidia_smi_pid = _fake_tools(tmp_path)
    output_dir = tmp_path / "profile-output"
    env = _profile_environment(bin_dir, curl_log, nvidia_smi_pid)
    env["FAKE_FINAL_STATUS"] = "failed"

    result = subprocess.run(
        _profile_command(tmp_path, output_dir),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert "workflow finished with status failed" in result.stderr
    assert not list(output_dir.glob("profiles_*.json"))

    sampler_pid = int(nvidia_smi_pid.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(sampler_pid, 0)


def test_profile_workflow_requires_one_visible_gpu(tmp_path: Path):
    bin_dir, curl_log, nvidia_smi_pid = _fake_tools(tmp_path)
    output_dir = tmp_path / "profile-output"
    env = _profile_environment(bin_dir, curl_log, nvidia_smi_pid)
    env["FAKE_GPU_LIST"] = "0\n1\n"

    result = subprocess.run(
        _profile_command(tmp_path, output_dir),
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 1
    assert "profiling requires exactly one visible GPU, found 2" in result.stderr
    assert not curl_log.exists()
