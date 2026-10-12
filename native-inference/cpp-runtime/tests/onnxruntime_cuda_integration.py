from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto, helper


from fixtures import write_affine_onnx
from model_builder.export import import_onnx_package


def _run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"{description} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.output.parent / "onnxruntime-cuda-affine-source.onnx"
    write_affine_onnx(source, graph_name="pnm-ir-cuda-affine")
    import_onnx_package(
        source,
        args.output,
        model_name="onnx-cuda-affine",
        model_version="0.1.0",
        target="cuda",
        force=True,
    )

    manifest = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["target"] == "cuda"

    input_file = args.output.parent / "onnxruntime-cuda-input.f32"
    output_file = args.output.parent / "onnxruntime-cuda-output.f32"
    input_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    output_file.unlink(missing_ok=True)
    result = _run(
        [
            str(args.pnmir),
            "run",
            str(args.output),
            "--input-file",
            f"input={input_file}",
            "--output-file",
            str(output_file),
            "--backend",
            "onnxruntime",
            "--device",
            "cuda",
            "--warmup",
            "2",
            "--iterations",
            "3",
        ],
        "ONNX Runtime CUDA inference",
    )
    assert "benchmark: warmup=2 iterations=3 " in result.stdout, result.stdout
    assert struct.unpack("=3f", output_file.read_bytes()) == (3.0, 5.0, 7.0)

    unsupported_source = (
        args.output.parent / "onnxruntime-cuda-cpu-fallback-source.onnx"
    )
    unsupported_package = args.output.parent / "onnxruntime-cuda-cpu-fallback.pnmir"
    unsupported_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3])
    unsupported_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None]
    )
    unsupported_graph = helper.make_graph(
        [helper.make_node("Unique", ["input"], ["output"], sorted=1)],
        "pnm-ir-cuda-cpu-fallback",
        [unsupported_input],
        [unsupported_output],
    )
    unsupported_model = helper.make_model(
        unsupported_graph,
        producer_name="pnm-ir-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(unsupported_model)
    onnx.save(unsupported_model, unsupported_source)
    import_onnx_package(
        unsupported_source,
        unsupported_package,
        model_name="onnx-cuda-cpu-fallback",
        model_version="0.1.0",
        target="cuda",
        force=True,
    )
    fallback = subprocess.run(
        [
            str(args.pnmir),
            "run",
            str(unsupported_package),
            "--values",
            "1,2,1",
            "--backend",
            "onnxruntime",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
    )
    assert fallback.returncode != 0, fallback.stdout
    assert "fallback to CPU EP has been explicitly disabled" in fallback.stderr, (
        fallback.stderr
    )
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
