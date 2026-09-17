from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path


from fixtures.aoti_identity.model import create_model, example_inputs
from pnmir_export import export_package


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
        model_name="aoti-affine",
        model_version="0.1.0",
        input_names=("input",),
        output_names=("output",),
    )

    manifest = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["backend"] == "aoti"
    assert manifest["producer"]["torch_version"]
    assert manifest["producer"]["torch_export_strict"] is True
    assert manifest["producer"]["aoti_validation"] == "isolated_process"

    result = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(args.output),
            "--values",
            "1,2,3",
            "--backend",
            "aoti",
            "--warmup",
            "2",
            "--iterations",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"PhysicsNeMo Inference failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("benchmark: warmup=2 iterations=3 "), result.stdout
    assert lines[1] == "output: 3 5 7", result.stdout
    print(result.stdout.strip())

    input_file = args.output.parent / "aoti-identity-input.f32"
    output_file = args.output.parent / "aoti-identity-output.f32"
    input_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    output_file.unlink(missing_ok=True)
    file_result = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(args.output),
            "--input-file",
            f"input={input_file}",
            "--output-file",
            str(output_file),
            "--backend",
            "aoti",
        ],
        capture_output=True,
        text=True,
    )
    if file_result.returncode != 0:
        raise AssertionError(
            f"PhysicsNeMo Inference file inference failed with exit {file_result.returncode}\n"
            f"stdout:\n{file_result.stdout}\nstderr:\n{file_result.stderr}"
        )
    assert struct.unpack("=3f", output_file.read_bytes()) == (3.0, 5.0, 7.0)
    print(file_result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
