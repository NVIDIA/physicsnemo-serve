from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from fixtures import write_affine_onnx
from fixtures.affine import create_model, example_inputs
from model_builder.export import export_package


def _add_onnx_artifact(package: Path) -> None:
    artifact_dir = package / "artifacts" / "onnx"
    artifact_dir.mkdir(parents=True)
    write_affine_onnx(
        artifact_dir / "model.onnx", graph_name="pnm-ir-multi-backend-affine"
    )

    manifest_path = package / "model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "backend": "onnxruntime",
            "target": "cpu",
            "precision": "fp32",
            "path": "artifacts/onnx/model.onnx",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_backend(pnmir: Path, package: Path, backend: str) -> str:
    result = subprocess.run(
        [
            str(pnmir),
            "run",
            str(package),
            "--values",
            "1,2,3",
            "--backend",
            backend,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{backend} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    shutil.rmtree(args.output, ignore_errors=True)
    export_package(
        create_model(),
        example_inputs(),
        args.output,
        model_name="multi-backend-affine",
        model_version="0.1.0",
        input_names=("input",),
        output_names=("output",),
    )
    _add_onnx_artifact(args.output)

    manifest = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    assert [artifact["backend"] for artifact in manifest["artifacts"]] == [
        "aoti",
        "onnxruntime",
    ]
    aoti_output = _run_backend(args.pnmir, args.output, "aoti")
    onnxruntime_output = _run_backend(args.pnmir, args.output, "onnxruntime")
    assert aoti_output == "output: 3 5 7", aoti_output
    assert onnxruntime_output == aoti_output, onnxruntime_output
    print(f"aoti: {aoti_output}")
    print(f"onnxruntime: {onnxruntime_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
