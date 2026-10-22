# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of CI jobs the agent handles and the commands that rerun them."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ci_autofix.failures import Failure

DEFAULT_JOBS_FILE = Path(__file__).resolve().parents[1] / "jobs.yaml"
MODULE_DIR = "{module_dir}"


@dataclass(frozen=True)
class Runner:
    rerun: str
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobSpec:
    name: str
    runners: dict[str, Runner]
    suite: str
    autofix: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    @property
    def kinds(self) -> list[str]:
        return list(self.runners)


@dataclass(frozen=True)
class Command:
    """A shell script run inside the sandbox image, relative to the checkout root."""

    script: str
    cwd: str = "."
    env: dict[str, str] = field(default_factory=dict)


def load_jobs(path: Path = DEFAULT_JOBS_FILE) -> dict[str, JobSpec]:
    raw = yaml.safe_load(path.read_text())["jobs"]
    jobs = {}
    for name, spec in raw.items():
        runners = {kind: Runner(**runner) for kind, runner in spec["runners"].items()}
        jobs[name] = JobSpec(
            name=name,
            runners=runners,
            suite=spec["suite"],
            autofix=spec.get("autofix"),
            env=spec.get("env", {}),
        )
    return jobs


def find_module_dir(workspace: Path, module: str) -> str:
    """Directory (relative to the checkout) holding a unittest module's file."""
    matches = sorted(
        p.parent.relative_to(workspace).as_posix()
        for p in workspace.rglob(f"{module}.py")
        if ".git" not in p.parts and "target" not in p.parts
    )
    if len(matches) != 1:
        raise LookupError(
            f"expected one {module}.py in the checkout, found {len(matches)}: {matches}"
        )
    return matches[0]


def rerun_commands(
    spec: JobSpec, failures: list[Failure], workspace: Path
) -> list[Command]:
    """Commands that rerun exactly the given failures of one job."""
    commands = []
    by_kind: dict[str, list[Failure]] = {}
    for failure in failures:
        by_kind.setdefault(failure.kind, []).append(failure)
    for kind, group in by_kind.items():
        runner = spec.runners[kind]
        env = spec.env | runner.env
        if runner.cwd == MODULE_DIR:
            by_dir: dict[str, list[str]] = {}
            for failure in group:
                module = failure.test_id.split(".")[0]
                by_dir.setdefault(find_module_dir(workspace, module), []).append(
                    failure.test_id
                )
            commands.extend(
                Command(runner.rerun.format(ids=shlex.join(ids)), cwd, env)
                for cwd, ids in by_dir.items()
            )
        else:
            ids = [f.test_id for f in group if f.test_id]
            commands.append(
                Command(runner.rerun.format(ids=shlex.join(ids)), runner.cwd, env)
            )
    return commands


def suite_command(spec: JobSpec) -> Command:
    return Command(spec.suite, ".", spec.env)
