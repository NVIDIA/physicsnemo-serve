from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "scripts" / "entrypoint.sh"


def _entrypoint_runtime_config_env(overrides: dict[str, str]) -> tuple[str, str]:
    source = ENTRYPOINT.read_text(encoding="utf-8")
    configuration = source.split("# Helper Functions", 1)[0]
    script = configuration + (
        '\nprintf "%s\\n%s\\n" "$WORKER_RUNTIME_CONFIG" '
        '"$PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG"\n'
    )
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, **overrides},
    )
    worker_config, server_config = result.stdout.splitlines()
    return worker_config, server_config


def test_entrypoint_worker_runtime_override_replaces_baked_in_server_default():
    worker_config, server_config = _entrypoint_runtime_config_env(
        {
            "WORKER_RUNTIME_CONFIG": "/outputs/custom/worker-runtime.json",
            "PHYSICSNEMO_SERVE_RUNTIME_ENVS_CONFIG": (
                "/app/scripts/worker_runtime_config.json"
            ),
        }
    )

    assert worker_config == "/outputs/custom/worker-runtime.json"
    assert server_config == worker_config
