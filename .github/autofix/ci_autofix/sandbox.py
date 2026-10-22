# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run commands inside the ci-autofix-env image against a checkout."""

from __future__ import annotations

import os
import posixpath
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ci_autofix.jobs import Command

WORKSPACE = "/workspace"
DEFAULT_TIMEOUT = 30 * 60
OUTPUT_LIMIT = 200_000


@dataclass
class Result:
    returncode: int
    output: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def container_mounts(cache_dir: Path) -> list[str]:
    """Bind mounts shared by triage runs and the agent container.

    Caches live on the host so they persist across ``docker run`` calls and are
    owned by the runner user, which the container runs as.
    """
    mounts = []
    for name, target in (
        ("cargo-registry", "/usr/local/cargo/registry"),
        ("uv", "/cache/uv"),
    ):
        host = cache_dir / name
        host.mkdir(parents=True, exist_ok=True)
        mounts += ["-v", f"{host}:{target}"]
    return mounts


def container_env() -> dict[str, str]:
    return {
        "HOME": "/tmp",
        "UV_CACHE_DIR": "/cache/uv",
        "CARGO_TERM_COLOR": "never",
        "NO_COLOR": "1",
    }


def user_args() -> list[str]:
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


class DockerSandbox:
    """Executes one command per ``docker run``. Holds no credentials."""

    def __init__(self, image: str, cache_dir: Path, executable: str = "docker"):
        self.image = image
        self.cache_dir = cache_dir
        self.executable = executable

    def argv(self, command: Command, checkout: Path) -> list[str]:
        env = container_env() | command.env
        argv = [
            self.executable,
            "run",
            "--rm",
            *user_args(),
            "-v",
            f"{checkout}:{WORKSPACE}",
        ]
        argv += container_mounts(self.cache_dir)
        argv += ["-w", posixpath.normpath(f"{WORKSPACE}/{command.cwd}")]
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        return [*argv, self.image, "bash", "-lc", command.script]

    def run(
        self, command: Command, checkout: Path, timeout: int = DEFAULT_TIMEOUT
    ) -> Result:
        try:
            completed = subprocess.run(
                self.argv(command, checkout),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            output = error.output if isinstance(error.output, str) else ""
            return Result(124, f"{output}\n[timed out after {timeout}s]")
        return Result(completed.returncode, completed.stdout[-OUTPUT_LIMIT:])
