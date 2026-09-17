from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper


from pnmir_export import build_tensorrt_package


def _write_affine_onnx(path: Path) -> None:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["output"]),
        ],
        "pnm-ir-tensorrt-affine",
        [input_info],
        [output_info],
        [scale, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="pnm-ir-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_gather_onnx(path: Path) -> None:
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [4])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT32, [3])
    output = helper.make_tensor_value_info("selected", TensorProto.FLOAT, [3])
    graph = helper.make_graph(
        [helper.make_node("Gather", ["values", "indices"], ["selected"], axis=0)],
        "pnm-ir-tensorrt-gather",
        [values, indices],
        [output],
    )
    model = helper.make_model(
        graph,
        producer_name="pnm-ir-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _run(
    pnmir: Path, package: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(pnmir),
            "run",
            str(package),
            "--backend",
            "tensorrt",
            "--device",
            "cuda",
            *arguments,
        ],
        capture_output=True,
        text=True,
    )


def _assert_success(result: subprocess.CompletedProcess[str], description: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{description} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _copy_with_manifest_change(
    source: Path, destination: Path, change: Callable[[dict], None]
) -> Path:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)
    manifest_path = destination / "model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    change(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination


def _assert_failure(pnmir: Path, package: Path, expected_error: str) -> None:
    result = _run(pnmir, package, "--values", "1,2,3")
    assert result.returncode != 0, result.stdout
    assert expected_error in result.stderr, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping TensorRT integration test")
        return 77

    source = args.output.parent / "tensorrt-affine-source.onnx"
    source.parent.mkdir(parents=True, exist_ok=True)
    _write_affine_onnx(source)
    build_tensorrt_package(
        source,
        args.output,
        model_name="tensorrt-affine",
        model_version="0.1.0",
        force=True,
    )

    manifest = json.loads((args.output / "model.json").read_text(encoding="utf-8"))
    artifact = manifest["artifacts"][0]
    assert artifact["backend"] == "tensorrt"
    assert artifact["target"] == "cuda"
    assert artifact["precision"] == "fp32"
    assert artifact["runtime_version"]
    major, minor = torch.cuda.get_device_capability()
    assert manifest["producer"]["compute_capability"] == f"{major}.{minor}"

    result = _run(
        args.pnmir,
        args.output,
        "--values",
        "1,2,3",
        "--warmup",
        "2",
        "--iterations",
        "3",
    )
    _assert_success(result, "TensorRT affine inference")
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("benchmark: warmup=2 iterations=3 "), result.stdout
    assert lines[1] == "output: 3 5 7", result.stdout

    input_file = args.output.parent / "tensorrt-input.f32"
    output_file = args.output.parent / "tensorrt-output.f32"
    input_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    output_file.unlink(missing_ok=True)
    file_result = _run(
        args.pnmir,
        args.output,
        "--input-file",
        f"input={input_file}",
        "--output-file",
        str(output_file),
    )
    _assert_success(file_result, "TensorRT file inference")
    assert struct.unpack("=3f", output_file.read_bytes()) == (3.0, 5.0, 7.0)

    gather_source = args.output.parent / "tensorrt-gather-source.onnx"
    gather_package = args.output.parent / "tensorrt-gather.pnmir"
    _write_gather_onnx(gather_source)
    build_tensorrt_package(
        gather_source,
        gather_package,
        model_name="tensorrt-gather",
        model_version="0.1.0",
        force=True,
    )
    gather_values = args.output.parent / "tensorrt-gather-values.f32"
    gather_indices = args.output.parent / "tensorrt-gather-indices.i32"
    gather_output = args.output.parent / "tensorrt-gather-output.f32"
    gather_values.write_bytes(struct.pack("=4f", 2.0, 4.0, 6.0, 8.0))
    gather_indices.write_bytes(struct.pack("=3i", 3, 1, 3))
    gather_output.unlink(missing_ok=True)
    gather_result = _run(
        args.pnmir,
        gather_package,
        "--input-file",
        f"values={gather_values}",
        "--input-file",
        f"indices={gather_indices}",
        "--output-file",
        str(gather_output),
    )
    _assert_success(gather_result, "TensorRT int32 gather inference")
    assert struct.unpack("=3f", gather_output.read_bytes()) == (8.0, 4.0, 8.0)

    bad_name_package = _copy_with_manifest_change(
        args.output,
        args.output.parent / "tensorrt-bad-name.pnmir",
        lambda value: value["inputs"][0].update(name="manifest_input"),
    )
    _assert_failure(
        args.pnmir,
        bad_name_package,
        "TensorRT engine has no input named: manifest_input",
    )

    bad_shape_package = _copy_with_manifest_change(
        args.output,
        args.output.parent / "tensorrt-bad-shape.pnmir",
        lambda value: value["inputs"][0].update(shape=[1, 3]),
    )
    _assert_failure(
        args.pnmir,
        bad_shape_package,
        "TensorRT input shape mismatch for tensor: input",
    )

    bad_dtype_package = _copy_with_manifest_change(
        args.output,
        args.output.parent / "tensorrt-bad-dtype.pnmir",
        lambda value: value["inputs"][0].update(dtype="uint8"),
    )
    _assert_failure(
        args.pnmir,
        bad_dtype_package,
        "TensorRT input dtype mismatch for tensor: input",
    )

    bad_version_package = _copy_with_manifest_change(
        args.output,
        args.output.parent / "tensorrt-bad-version.pnmir",
        lambda value: value["artifacts"][0].update(runtime_version="0.0.0"),
    )
    _assert_failure(
        args.pnmir,
        bad_version_package,
        "no compatible backend artifact for: tensorrt",
    )

    corrupt_package = args.output.parent / "tensorrt-corrupt.pnmir"
    shutil.rmtree(corrupt_package, ignore_errors=True)
    shutil.copytree(args.output, corrupt_package)
    corrupt_artifact = (
        corrupt_package
        / json.loads((corrupt_package / "model.json").read_text(encoding="utf-8"))[
            "artifacts"
        ][0]["path"]
    )
    corrupt_artifact.write_bytes(b"not a TensorRT engine")
    _assert_failure(
        args.pnmir,
        corrupt_package,
        "TensorRT could not deserialize engine",
    )

    print(result.stdout.strip())
    print(file_result.stdout.strip())
    print(gather_result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
