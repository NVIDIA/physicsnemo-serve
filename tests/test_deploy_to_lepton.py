from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deploy_to_lepton_rejects_missing_endpoint_token():
    env = {
        **os.environ,
        "LEPTON_WORKSPACE_ID": "workspace",
        "LEPTON_WORKSPACE_TOKEN": "workspace-token",
    }
    env.pop("USER", None)
    env.pop("LEPTON_ENDPOINT_TOKEN", None)

    result = subprocess.run(
        [
            str(REPO_ROOT / "scripts" / "deploy-to-lepton.sh"),
            "--source",
            "custom",
            "--skip-build",
            "--skip-push",
            "--image-name",
            "example.invalid/image:tag",
            "--port",
            "8000",
            "--endpoint-name",
            "test-endpoint",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "endpoint token is required" in result.stderr


def test_deploy_to_lepton_redacts_case_insensitive_secret_env_names(tmp_path):
    fake_lep = tmp_path / "lep"
    fake_lep.write_text("#!/bin/sh\nexit 0\n")
    fake_lep.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "LEPTON_WORKSPACE_ID": "workspace",
        "LEPTON_WORKSPACE_TOKEN": "workspace-token",
        "LEPTON_ENDPOINT_TOKEN": "endpoint-token",
    }
    env.pop("USER", None)

    result = subprocess.run(
        [
            str(REPO_ROOT / "scripts" / "deploy-to-lepton.sh"),
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
            "apiToKeN=mixed-case-secret-value",
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
    assert "database_password=<redacted>" in output
    assert "apiToKeN=<redacted>" in output
