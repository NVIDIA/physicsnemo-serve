from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path

import torch


from fixtures.aoti_identity.model import create_model, example_inputs
from pnmir_export import export_package


class Int32Gather(torch.nn.Module):
    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.index_select(values, 0, indices.to(torch.int64))


class DynamicIndexedMean(torch.nn.Module):
    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.index_select(values, 0, indices.to(torch.int64)).mean(dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping CUDA AOTI integration test")
        return 77

    shutil.rmtree(args.output, ignore_errors=True)
    export_package(
        create_model(),
        example_inputs(),
        args.output,
        model_name="aoti-affine-cuda",
        model_version="0.1.0",
        input_names=("input",),
        output_names=("output",),
        target="cuda",
    )

    manifest = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["target"] == "cuda"

    input_file = args.output.parent / "aoti-cuda-input.f32"
    output_file = args.output.parent / "aoti-cuda-output.f32"
    input_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    output_file.unlink(missing_ok=True)
    result = subprocess.run(
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
            "--device",
            "cuda",
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
            f"CUDA PhysicsNeMo Inference failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    torch.testing.assert_close(
        torch.tensor(struct.unpack("=3f", output_file.read_bytes())),
        torch.tensor((3.0, 5.0, 7.0)),
    )
    assert "benchmark: warmup=2 iterations=3 " in result.stdout
    print(result.stdout.strip())

    int32_package = args.output.parent / f"{args.output.name}-int32"
    shutil.rmtree(int32_package, ignore_errors=True)
    export_package(
        Int32Gather(),
        (
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            torch.tensor([3, 1, 3], dtype=torch.int32),
        ),
        int32_package,
        model_name="aoti-int32-gather-cuda",
        model_version="0.1.0",
        input_names=("values", "indices"),
        output_names=("selected",),
        target="cuda",
    )
    values_file = args.output.parent / "aoti-cuda-values.f32"
    indices_file = args.output.parent / "aoti-cuda-indices.i32"
    selected_file = args.output.parent / "aoti-cuda-selected.f32"
    values_file.write_bytes(struct.pack("=4f", 10.0, 20.0, 30.0, 40.0))
    indices_file.write_bytes(struct.pack("=3i", 3, 1, 3))
    selected_file.unlink(missing_ok=True)
    int32_result = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(int32_package),
            "--input-file",
            f"values={values_file}",
            "--input-file",
            f"indices={indices_file}",
            "--output-file",
            f"selected={selected_file}",
            "--backend",
            "aoti",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
    )
    if int32_result.returncode != 0:
        raise AssertionError(
            f"CUDA int32 PhysicsNeMo Inference failed with exit {int32_result.returncode}\n"
            f"stdout:\n{int32_result.stdout}\nstderr:\n{int32_result.stderr}"
        )
    assert struct.unpack("=3f", selected_file.read_bytes()) == (40.0, 20.0, 40.0)

    dynamic_package = args.output.parent / f"{args.output.name}-dynamic"
    shutil.rmtree(dynamic_package, ignore_errors=True)
    export_package(
        DynamicIndexedMean(),
        (
            torch.arange(12, dtype=torch.float32).reshape(4, 3),
            torch.tensor([0, 2, 3], dtype=torch.int32),
        ),
        dynamic_package,
        model_name="aoti-dynamic-indexed-mean-cuda",
        model_version="0.1.0",
        input_names=("values", "indices"),
        output_names=("mean",),
        target="cuda",
        dynamic_shapes=(
            None,
            {0: torch.export.Dim("selected_count", min=1, max=32)},
        ),
    )
    dynamic_manifest = json.loads(
        (dynamic_package / "model.json").read_text(encoding="utf-8")
    )
    assert dynamic_manifest["inputs"][1]["shape"] == [-1]
    dynamic_values_file = args.output.parent / "aoti-cuda-dynamic-values.f32"
    dynamic_indices_file = args.output.parent / "aoti-cuda-dynamic-indices.i32"
    dynamic_output_file = args.output.parent / "aoti-cuda-dynamic-mean.f32"
    dynamic_values_file.write_bytes(struct.pack("=12f", *range(12)))
    dynamic_indices_file.write_bytes(struct.pack("=5i", 3, 1, 3, 0, 2))
    dynamic_output_file.unlink(missing_ok=True)
    dynamic_result = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(dynamic_package),
            "--input-file",
            f"values={dynamic_values_file}",
            "--input-file",
            f"indices={dynamic_indices_file}",
            "--input-shape",
            "indices=5",
            "--output-file",
            f"mean={dynamic_output_file}",
            "--backend",
            "aoti",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
    )
    if dynamic_result.returncode != 0:
        raise AssertionError(
            f"CUDA dynamic PhysicsNeMo Inference failed with exit {dynamic_result.returncode}\n"
            f"stdout:\n{dynamic_result.stdout}\nstderr:\n{dynamic_result.stderr}"
        )
    torch.testing.assert_close(
        torch.tensor(struct.unpack("=3f", dynamic_output_file.read_bytes())),
        torch.tensor((5.4, 6.4, 7.4)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
