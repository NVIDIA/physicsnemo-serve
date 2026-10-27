# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run mini-swe-agent in the sandbox container on a failed CI run.

Reads AUTOFIX_MODEL, AUTOFIX_API_BASE and AUTOFIX_API_KEY (and optionally
AUTOFIX_MODEL_CLASS). The key stays in this process; the container gets no
secrets and sees only the git-free working copy at /workspace.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent / "agent.yaml"


def build_task(pr: str, log: str, diff: str) -> str:
    return (
        f"# Failed CI run\n\n```text\n{log}\n```\n\n"
        f"# Pull request\n\n{pr}\n\n"
        f"# Diff against the target branch\n\n```diff\n{diff}\n```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=Path, required=True, help="PR title and body")
    parser.add_argument("--log", type=Path, required=True, help="failed CI job logs")
    parser.add_argument("--diff", type=Path, required=True, help="PR diff")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    config = yaml.safe_load(CONFIG.read_text())
    api_key = os.environ["AUTOFIX_API_KEY"]
    model_config = config["model"] | {
        "model_name": os.environ["AUTOFIX_MODEL"],
        "model_class": os.environ.get("AUTOFIX_MODEL_CLASS") or "litellm",
        "model_kwargs": config["model"].get("model_kwargs", {})
        | {"api_base": os.environ["AUTOFIX_API_BASE"], "api_key": api_key},
    }
    env_config = config["environment"] | {
        "image": args.image,
        "env": config["environment"].get("env", {})
        | {"HOME": "/tmp", "UV_CACHE_DIR": "/tmp/uv", "NO_COLOR": "1"},
        "run_args": [
            "--rm",
            *("--user", f"{os.getuid()}:{os.getgid()}"),
            *("-v", f"{args.workspace.resolve()}:/workspace"),
            *("-v", f"{args.out.resolve()}:/autofix-out"),
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    trajectory = args.out / "trajectory.json"
    task = build_task(
        args.pr.read_text(), args.log.read_text(errors="replace"), args.diff.read_text()
    )
    env = get_environment(env_config, default_type="docker")
    agent = DefaultAgent(
        get_model(config=model_config),
        env,
        **config["agent"],
        output_path=trajectory,
    )
    try:
        status = agent.run(task).get("exit_status", "")
    finally:
        getattr(env, "cleanup", lambda: None)()
        if trajectory.is_file():  # the trajectory records model kwargs
            trajectory.write_text(trajectory.read_text().replace(api_key, "***"))
    print(f"agent exit status: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
