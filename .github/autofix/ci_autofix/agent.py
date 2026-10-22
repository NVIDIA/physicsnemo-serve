# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the agent's task and run mini-swe-agent in the sandbox container."""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass
from pathlib import Path

import yaml

from ci_autofix.failures import JobFailures
from ci_autofix.jobs import JobSpec, rerun_commands
from ci_autofix.sandbox import WORKSPACE, container_env, container_mounts, user_args
from ci_autofix.triage import REPRODUCIBLE, Classified
from ci_autofix.verdict import EXAMPLE

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "agent.yaml"
OUT_MOUNT = "/autofix-out"
MAX_DIFF_CHARS = 60_000
MAX_AGENTS_MD_CHARS = 8_000

logger = logging.getLogger(__name__)


@dataclass
class PullRequest:
    number: int
    title: str
    body: str
    commit_messages: list[str]
    diff: str


@dataclass
class AgentOutcome:
    exit_status: str
    verdict_text: str | None
    trajectory: Path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def build_task(
    pr: PullRequest,
    classified: list[Classified],
    jobs: list[JobFailures],
    specs: dict[str, JobSpec],
    workspace: Path,
    agents_md: str,
) -> str:
    """Markdown task description for the failures triage could not resolve."""
    targets = [c.failure for c in classified if c.status == REPRODUCIBLE]
    sections = [
        f"# CI failures on PR #{pr.number}: {pr.title}",
        "These failures reproduce on the PR head and pass on the merge-base with the "
        "target branch, so the PR most likely caused them.",
        "## Failures",
    ]
    for job in jobs:
        job_targets = [f for f in targets if f.job == job.job]
        if not job_targets:
            continue
        sections.append(f"### Job: {job.job}")
        sections.append(
            "Failure keys:\n" + "\n".join(f"- `{f.key}`" for f in job_targets)
        )
        commands = rerun_commands(specs[job.job], job_targets, workspace)
        lines = []
        for command in commands:
            env = " ".join(f"{k}={v}" for k, v in command.env.items())
            where = f"cd {posixpath.normpath(f'{WORKSPACE}/{command.cwd}')}"
            lines.append(f"{where} && {env + ' ' if env else ''}{command.script}")
        sections.append("Rerun with:\n```bash\n" + "\n".join(lines) + "\n```")
        sections.append("CI log excerpt:\n```text\n" + job.excerpt + "\n```")

    sections += [
        "## Pull request",
        "Description:\n" + (pr.body.strip() or "(empty)"),
        "Commit messages:\n"
        + "\n".join(f"- {m.splitlines()[0]}" for m in pr.commit_messages),
        "Diff against the merge-base with the target branch:\n"
        "```diff\n" + _truncate(pr.diff, MAX_DIFF_CHARS) + "\n```",
        "## Repository instructions (AGENTS.md)",
        _truncate(agents_md, MAX_AGENTS_MD_CHARS) or "(none)",
    ]
    return "\n\n".join(sections)


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def environment_config(
    config: dict, image: str, workspace: Path, out_dir: Path, cache_dir: Path
) -> dict:
    env_config = dict(config["environment"])
    env_config["image"] = image
    env_config["env"] = env_config.get("env", {}) | container_env()
    env_config["run_args"] = [
        "--rm",
        *user_args(),
        "-v",
        f"{workspace}:{WORKSPACE}",
        "-v",
        f"{out_dir}:{OUT_MOUNT}",
        *container_mounts(cache_dir),
    ]
    return env_config


def model_config(config: dict, model_name: str, api_base: str, api_key: str) -> dict:
    model = dict(config["model"])
    model["model_name"] = model_name
    model["model_kwargs"] = model.get("model_kwargs", {}) | {
        "api_base": api_base,
        "api_key": api_key,
    }
    return model


def run_agent(
    task: str,
    *,
    image: str,
    workspace: Path,
    out_dir: Path,
    cache_dir: Path,
    model_name: str,
    api_base: str,
    api_key: str,
    model_class: str = "litellm",
    config: dict | None = None,
) -> AgentOutcome:
    """Run mini-swe-agent to completion and collect verdict.json."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    config = config or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = out_dir / "trajectory.json"
    verdict_file = out_dir / "verdict.json"
    verdict_file.unlink(missing_ok=True)

    model_cfg = model_config(config, model_name, api_base, api_key)
    model_cfg["model_class"] = model_class
    model = get_model(config=model_cfg)
    env = get_environment(
        environment_config(config, image, workspace, out_dir, cache_dir),
        default_type="docker",
    )
    agent = DefaultAgent(model, env, **config["agent"], output_path=trajectory)
    try:
        result = agent.run(task, verdict_example=json.dumps(EXAMPLE, indent=2))
        exit_status = result.get("exit_status", "")
    except Exception as error:  # model endpoint or container failure
        logger.exception("agent run failed")
        exit_status = f"error: {type(error).__name__}: {error}"
    finally:
        if hasattr(env, "cleanup"):  # DockerEnvironment stops its container
            env.cleanup()
    _redact(trajectory, api_key)
    verdict_text = verdict_file.read_text() if verdict_file.is_file() else None
    return AgentOutcome(exit_status, verdict_text, trajectory)


def _redact(path: Path, secret: str) -> None:
    """The trajectory records model kwargs; strip the API key before upload."""
    if secret and path.is_file():
        path.write_text(path.read_text().replace(secret, "***"))
