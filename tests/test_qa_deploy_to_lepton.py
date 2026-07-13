from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deploy_preview_redacts_sensitive_env_names_case_insensitively(tmp_path):
    fake_lep = tmp_path / "lep"
    fake_lep.write_text("#!/bin/sh\nexit 0\n")
    fake_lep.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "LEPTON_WORKSPACE_ID": "workspace",
        "LEPTON_WORKSPACE_TOKEN": "workspace-token-secret-value",
        "LEPTON_ENDPOINT_TOKEN": "endpoint-token-secret-value",
    }
    env.pop("USER", None)

    result = subprocess.run(
        [
            str(REPO_ROOT / "qa" / "scripts" / "deploy-to-lepton.sh"),
            "--source",
            "custom",
            "--skip-build",
            "--skip-push",
            "--image-name",
            "example.invalid/image",
            "--port",
            "8000",
            "--endpoint-name",
            "test-endpoint",
            "--env",
            "database_password=lowercase-secret-value",
            "--env",
            "ApiToken=mixed-case-secret-value",
            "--env",
            "LOG_LEVEL=debug",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "lowercase-secret-value" not in output
    assert "mixed-case-secret-value" not in output
    assert "endpoint-token-secret-value" not in output
    assert "workspace-token-secret-value" not in output
    assert "--tokens <redacted>" in output
    assert "database_password=<redacted>" in output
    assert "ApiToken=<redacted>" in output
    assert "LOG_LEVEL=debug" in output
