"""CPU component checks for release configuration of the existing Python stack."""

from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "application-cli/src"))
from pnms_app.common import legacy_module  # noqa: E402
from pnms_app.python_release import configure, prepare  # noqa: E402


def manifest():
    return {
        "metadata": {"id": "my-python-plugin"},
        "runtime": {
            "kind": "python",
            "entrypoint": "workflow.py",
            "executor_class": "custom.cpu",
        },
        "resources": {
            "defaults": {"device_kind": "cpu", "gpus_required": 0, "memory_mb": 1024}
        },
        "pipeline": {
            "stages": [
                {
                    "id": "prepare",
                    "phase": "prepare",
                    "handler": "plugin_phase",
                    "queue": "prepare",
                    "next": "execute",
                },
                {
                    "id": "execute",
                    "phase": "execute",
                    "handler": "plugin_phase",
                    "queue": "execute.custom.cpu",
                    "next": "results",
                },
                {
                    "id": "results",
                    "phase": "results",
                    "handler": "persist_results",
                    "queue": "results",
                    "next": None,
                },
            ]
        },
    }


def test_python_release_registers_new_cpu_executor_and_isolates_plugin():
    base = json.loads((ROOT / "scripts/worker_runtime_config.json").read_text())
    before = deepcopy(base)
    result = configure(
        manifest(),
        base,
        python="/opt/server/bin/python",
        legacy=legacy_module(ROOT, "plugin_dev"),
    )
    environments = result["runtime_config"]["python_runtime_envs"]
    assert "custom.cpu" in environments, (
        "release must register the declared Python executor"
    )
    assert environments["custom.cpu"]["launch"]["device_kind"] == "cpu"
    assert environments["custom.cpu"]["python_executable"] == "/opt/server/bin/python"
    assert "earth2-gpu" not in environments
    assert result["environment"]["PLUGIN_DIR"] == "/opt/pnms/plugin"
    assert (
        result["environment"]["PHYSICSNEMO_SERVE_ENABLED_PLUGIN_ID"]
        == "my-python-plugin"
    )
    assert result["environment"]["INFERENCE_STREAM"] == "physicsnemo:prepare"
    assert result["runtime_config"]["max_retries"] == base["max_retries"]
    assert base == before


def test_python_release_retains_existing_selected_runtime_and_pipeline_tuning():
    base = json.loads((ROOT / "scripts/worker_runtime_config.json").read_text())
    authored = manifest()
    authored["runtime"]["executor_class"] = "physicsnemo-cfd-gpu"
    authored["runtime"]["prepare_executor_class"] = "earth2-cpu"
    authored["resources"]["defaults"].update(device_kind="gpu", gpus_required=1)
    authored["pipeline"]["stages"][1]["queue"] = "execute.physicsnemo-cfd-gpu"
    result = configure(
        authored,
        base,
        python="/opt/server/bin/python",
        legacy=legacy_module(ROOT, "plugin_dev"),
    )
    environments = result["runtime_config"]["python_runtime_envs"]
    assert (
        environments["physicsnemo-cfd-gpu"]["python_executable"]
        == "/opt/physicsnemo-cfd-venv/bin/python"
    )
    assert environments["physicsnemo-cfd-gpu"]["launch"]["enabled"] is True
    assert (
        environments["physicsnemo-cfd-gpu"]["launch"]["tags"]
        == base["python_runtime_envs"]["physicsnemo-cfd-gpu"]["launch"]["tags"]
    )
    assert (
        environments["earth2-cpu"]["python_executable"]
        == "/opt/physicsnemo-serve-venv/bin/python"
    )
    assert (
        environments["physicsnemo-cfd-gpu"]["env"]
        == base["python_runtime_envs"]["physicsnemo-cfd-gpu"]["env"]
    )
    assert (
        result["runtime_config"]["roles"]["prepare"]["config"]
        == base["roles"]["prepare"]["config"]
    )


def test_image_configuration_does_not_import_user_workflow(tmp_path):
    legacy = legacy_module(ROOT, "plugin_dev")
    legacy.init_plugin(tmp_path / "plugin")
    workflow = tmp_path / "plugin/workflow.py"
    workflow.write_text(
        'raise RuntimeError("image packaging must not import my model")\n'
    )
    result = prepare(
        root=tmp_path,
        scripts=ROOT / "scripts",
        base_config=ROOT / "scripts/worker_runtime_config.json",
        python=sys.executable,
    )
    assert result["workflow_id"] == "plugin"
    assert (
        result["runtime_config"]["python_runtime_envs"]["python.test"]["launch"][
            "device_kind"
        ]
        == "cpu"
    )
    assert "must not import my model" in workflow.read_text()
    assert (
        json.loads((tmp_path / "python-launch.json").read_text())["workflow_id"]
        == "plugin"
    )


def test_python_release_keeps_scheduler_pipeline_and_base_batch_tuning():
    base = json.loads((ROOT / "scripts/worker_runtime_config.json").read_text())
    authored = manifest()
    authored["pipeline"]["stages"].insert(
        1,
        {
            "id": "schedule",
            "phase": "schedule",
            "handler": "schedule",
            "queue": "schedule",
            "next": "execute",
        },
    )
    before = deepcopy(authored)
    result = configure(
        authored, base, python=sys.executable, legacy=legacy_module(ROOT, "plugin_dev")
    )
    assert (
        result["runtime_config"]["roles"]["scheduler"]["config"]
        == base["roles"]["scheduler"]["config"]
    )
    assert "scheduler" in result["environment"]["WORKERS"].split(",")
    assert "collect" in result["environment"]["WORKERS"].split(",")
    assert authored == before
