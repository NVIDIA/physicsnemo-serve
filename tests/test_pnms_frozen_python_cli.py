"""CPU subprocess checks for frozen legacy plugin discovery."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def invoke(*arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / "pnms"), *map(str, arguments)],
        env={**os.environ, "PNMS_REPOSITORY": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("manifest_edit", ["invalid", "missing"])
def test_frozen_legacy_run_uses_the_built_manifest(tmp_path, manifest_edit):
    app = tmp_path / "app"
    initialized = invoke("init", app, "--runtime", "python")
    assert initialized.returncode == 0, initialized.stderr
    built = invoke("build", app)
    assert built.returncode == 0, built.stderr
    receipt = Path(json.loads(built.stdout)["root"]) / "build.json"
    original = receipt.read_bytes()
    request = tmp_path / "request.json"
    request.write_text('{"value":7}')
    manifest = app / "plugin.yaml"
    if manifest_edit == "invalid":
        manifest.write_text("metadata: [unfinished edit")
    else:
        manifest.unlink()

    result = invoke("run", app, "--input", request)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["result"]["doubled"] == 14
    assert receipt.read_bytes() == original
    assert not (app / "pnms.yaml").exists()
    development = invoke("run", app, "--input", request, "--dev")
    assert development.returncode != 0


def test_frozen_discovery_verifies_the_receipt_before_dispatch(tmp_path):
    app = tmp_path / "app"
    assert invoke("init", app).returncode == 0
    built = invoke("build", app)
    assert built.returncode == 0, built.stderr
    receipt = Path(json.loads(built.stdout)["root"]) / "build.json"
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    (app / "plugin.yaml").unlink()
    request = tmp_path / "request.json"
    request.write_text('{"value":7}')
    result = invoke("run", app, "--input", request)
    assert result.returncode != 0
    assert "receipt changed" in result.stderr
