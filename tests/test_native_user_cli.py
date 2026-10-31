from pathlib import Path
import subprocess
import sys
import yaml

import pytest


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["init", "--help"], "--native-image"),
        (["validate", "--help"], "--release-profile"),
        (["check", "--help"], "--model-only"),
        (["check-env", "--help"], "--release-profile"),
        (["run-local", "--help"], "--image"),
        (["release", "--help"], "--promote"),
    ],
)
def test_native_user_commands_are_discoverable(arguments, expected):
    command = Path(__file__).parents[1] / "scripts/plugin_dev.py"
    result = subprocess.run(
        [sys.executable, str(command), *arguments], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_cli_can_scaffold_native_multipart_tensor(tmp_path):
    root = tmp_path / "native-upload"
    command = Path(__file__).parents[1] / "scripts/plugin_dev.py"
    result = subprocess.run(
        [
            sys.executable,
            str(command),
            "init",
            str(root),
            "--runtime",
            "cpp",
            "--template",
            "tensor",
            "--content-type",
            "multipart/form-data",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = yaml.safe_load((root / "plugin.yaml").read_text())
    assert manifest["ingress"]["content_type"] == "multipart/form-data"
    assert manifest["ingress"]["files"][0]["name"] == "features"
