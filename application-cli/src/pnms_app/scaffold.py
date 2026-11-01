"""Create application inputs without installing or running ML frameworks."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from .common import legacy_module, require, write_json


def init_native(
    root, *, repository, backend="aoti", template="tensor", model_project=None
):
    root = Path(root).resolve()
    require(not root.exists(), f"Application directory already exists: {root}")
    require(
        template in ("tensor", "custom"), "native template must be tensor or custom"
    )
    require(backend in ("aoti", "tensorrt"), "backend must be aoti or tensorrt")
    if model_project:
        model_project = Path(model_project).resolve()
        require(model_project.is_file(), f"model project missing: {model_project}")
    root.mkdir(parents=True)
    try:
        templates = legacy_module(repository, "native_workflow")
        if model_project is None:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(repository) / "native-inference/pnms-model-builder"),
                    "init",
                    str(root / "model"),
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            require(
                process.returncode == 0,
                f"Model Builder init failed: {process.stderr or process.stdout}",
            )
            model_project = root / "model/model-build.json"
            doc = json.loads(model_project.read_text())
            doc.update(
                name=root.name,
                version="0.1.0",
                adapter="build_adapter.py",
                checkpoint="checkpoint.pt",
                source=[],
                config={},
                input_names=["input"],
                output_names=["output"],
                backends=[backend],
                executor="local",
                device="cuda",
                output_root="builds",
            )
            doc.pop("required_gpu_arch", None)
            (root / "model/build_adapter.py").write_text(templates.AFFINE_MODEL)
            shutil.copyfile(
                Path(repository) / "tests/fixtures/native_affine_empty_state.pt",
                root / "model/checkpoint.pt",
            )
            write_json(model_project, doc)
        document = {
            "schema_version": 1,
            "name": root.name,
            "runtime": "cpp",
            "models": {"primary": {"project": os.path.relpath(model_project, root)}},
            "build": {
                "backend": backend,
                "environment": "local",
                "target": "current-gpu",
            },
            "workflow": {"template": template},
            "serve": {"port": 8080},
        }
        if template == "custom":
            native = root / "native"
            native.mkdir()
            (native / "workflow.cpp").write_text(templates.CUSTOM_WORKFLOW)
            (native / "CMakeLists.txt").write_text(templates.CUSTOM_CMAKE)
            document["workflow"].update(
                source_dir="native", target="my_native_workflow"
            )
        (root / "pnms.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
        write_json(root / "examples/request.json", {"values": [0, 1, -1, 4]})
        write_json(
            root / "fixtures/self-test.json",
            {
                "primary": {
                    "inputs": {"input": [0, 1, -1, 4]},
                    "expected_outputs": {"output": [1, 3, -1, 9]},
                    "rtol": 1e-5,
                    "atol": 1e-6,
                }
            },
        )
        write_json(
            root / "fixtures/cases.json",
            {
                "schema_version": 1,
                "cases": [
                    {
                        "name": "default",
                        "request": {"values": [0, 1, -1, 4]},
                        "expected": {"values": [1, 3, -1, 9]},
                    },
                    {
                        "name": "alternate",
                        "request": {"values": [2.5, -8, 0.125, 13]},
                        "expected": {"values": [6, -15, 1.25, 27]},
                    },
                ],
            },
        )
        (root / ".gitignore").write_text(".pnms/\n__pycache__/\nmodel/builds/\n")
        (root / "README.md").write_text(
            "# "
            + root.name
            + '\n\nThis example computes values * 2 + 1. Replace model/build_adapter.py, weights, and independent fixtures for your model. The custom template calls context.infer("primary", inputs).\n\n```sh\npnms check --config-only\npnms build\npnms run --input examples/request.json\n# Optional Docker release with a matched serve.image setting:\npnms build --image '
            + root.name
            + ":0.1\n```\n\nA host build needs a Python venv, CUDA/compiler and complete installed C++ SDK. Set build.python, build.runtime and build.sdk_prefix to existing tools, or build.venv for a managed environment. Docker and Lepton are not needed for local build/inference.\n"
        )
        return {
            "schema_version": 1,
            "status": "initialized",
            "runtime": "cpp",
            "project": str(root),
            "backend": backend,
            "template": template,
        }
    except BaseException:
        shutil.rmtree(root)
        raise
