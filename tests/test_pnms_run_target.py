"""Local inference admits the current allocated GPU against the build target."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app import admission, build, cli, native  # noqa: E402


@pytest.mark.parametrize(
    "name,driver,accepted",
    [
        ("NVIDIA H100 80GB HBM3", "570.195.03", True),
        ("NVIDIA H200", "570.195.03", False),
        ("NVIDIA H100 80GB HBM3", "580.1", False),
    ],
)
def test_local_run_checks_current_gpu_before_loading_model(
    tmp_path, monkeypatch, name, driver, accepted
):
    project = {"root": tmp_path, "runtime": "cpp", "name": "test", "build": {}}
    built = {
        "root": str(tmp_path),
        "bundle": "bundle",
        "environment": {
            "kind": "local",
            "gpu": {
                "name": "NVIDIA H100 80GB HBM3",
                "architecture": "sm90",
                "driver_version": "570.195.03",
                "mig_mode": "Disabled",
                "binding": "GPU-build",
            },
        },
    }
    monkeypatch.setattr(cli, "load_project", lambda *a, **kw: project)
    monkeypatch.setattr(build, "last_build", lambda *a, **kw: built)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-current")
    calls = []
    provider_checks = []
    monkeypatch.setattr(
        admission,
        "verify_providers",
        lambda *a, **kw: provider_checks.append(
            kw["environment"]["CUDA_VISIBLE_DEVICES"]
        ),
    )

    def run(*a, **kw):
        calls.append(kw["environment"]["CUDA_VISIBLE_DEVICES"])
        return {"status": "ran"}

    monkeypatch.setattr(native, "run_bundle", run)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 0, f"{name}, 9.0, {driver}, Disabled, GPU-current\n", ""
        ),
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"values": [1, 2, 3, 4]}))
    args = cli.parser().parse_args(["run", str(tmp_path), "--input", str(request)])
    if accepted:
        assert cli.execute(args)["status"] == "ran"
        assert calls == ["GPU-current"]
        assert provider_checks == ["GPU-current"]
    else:
        with pytest.raises(ValueError, match="target|driver"):
            cli.execute(args)
        assert not calls
        assert not provider_checks
