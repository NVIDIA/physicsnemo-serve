from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto, helper


from fixtures import write_affine_onnx
from model_builder.export import import_onnx_package


def _write_package(output: Path) -> None:
    shutil.rmtree(output, ignore_errors=True)
    source = output.parent / "onnxruntime-affine-source.onnx"

    write_affine_onnx(source, graph_name="pnm-ir-affine")
    import_onnx_package(
        source,
        output,
        model_name="onnx-affine",
        model_version="0.1.0",
        force=True,
    )


def _write_multi_io_package(output: Path) -> None:
    shutil.rmtree(output, ignore_errors=True)
    artifact_dir = output / "artifacts" / "onnx"
    artifact_dir.mkdir(parents=True)

    left_info = helper.make_tensor_value_info("left", TensorProto.FLOAT, [3])
    right_info = helper.make_tensor_value_info("right", TensorProto.FLOAT, [3])
    sum_info = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [3])
    difference_info = helper.make_tensor_value_info(
        "difference", TensorProto.FLOAT, [3]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["left", "right"], ["sum"]),
            helper.make_node("Sub", ["left", "right"], ["difference"]),
        ],
        "pnm-ir-multi-io",
        [left_info, right_info],
        [sum_info, difference_info],
    )
    model = helper.make_model(
        graph,
        producer_name="pnm-ir-test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, artifact_dir / "model.onnx")

    manifest = {
        "format_version": 1,
        "model": {"name": "onnx-multi-io", "version": "0.1.0"},
        "inputs": [
            {"name": "right", "dtype": "float32", "shape": [3]},
            {"name": "left", "dtype": "float32", "shape": [3]},
        ],
        "outputs": [
            {"name": "difference", "dtype": "float32", "shape": [3]},
            {"name": "sum", "dtype": "float32", "shape": [3]},
        ],
        "artifacts": [
            {
                "backend": "onnxruntime",
                "target": "cpu",
                "precision": "fp32",
                "path": "artifacts/onnx/model.onnx",
            }
        ],
    }
    (output / "model.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _run(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"{description} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _assert_run_failure(
    pnmir: Path,
    package: Path,
    expected_error: str,
) -> None:
    result = subprocess.run(
        [
            str(pnmir),
            "run",
            str(package),
            "--values",
            "1,2,3",
            "--backend",
            "onnxruntime",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert expected_error in result.stderr, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnmir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _write_package(args.output)

    result = _run(
        [
            str(args.pnmir),
            "run",
            str(args.output),
            "--values",
            "1,2,3",
            "--backend",
            "onnxruntime",
            "--warmup",
            "2",
            "--iterations",
            "3",
        ],
        "ONNX Runtime inference",
    )
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("benchmark: warmup=2 iterations=3 "), result.stdout
    assert lines[1] == "output: 3 5 7", result.stdout

    input_file = args.output.parent / "onnxruntime-input.f32"
    output_file = args.output.parent / "onnxruntime-output.f32"
    input_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    output_file.unlink(missing_ok=True)
    file_result = _run(
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
        ],
        "ONNX Runtime file inference",
    )
    assert struct.unpack("=3f", output_file.read_bytes()) == (3.0, 5.0, 7.0)

    bad_name_package = args.output.parent / "onnxruntime-bad-name.pnmir"
    _write_package(bad_name_package)
    bad_name_manifest_path = bad_name_package / "model.json"
    bad_name_manifest = json.loads(bad_name_manifest_path.read_text(encoding="utf-8"))
    bad_name_manifest["inputs"][0]["name"] = "manifest_input"
    bad_name_manifest_path.write_text(
        json.dumps(bad_name_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _assert_run_failure(
        args.pnmir,
        bad_name_package,
        "ONNX Runtime model has no input named: manifest_input",
    )

    bad_shape_package = args.output.parent / "onnxruntime-bad-shape.pnmir"
    _write_package(bad_shape_package)
    bad_shape_manifest_path = bad_shape_package / "model.json"
    bad_shape_manifest = json.loads(bad_shape_manifest_path.read_text(encoding="utf-8"))
    bad_shape_manifest["inputs"][0]["shape"] = [1, 3]
    bad_shape_manifest_path.write_text(
        json.dumps(bad_shape_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _assert_run_failure(
        args.pnmir,
        bad_shape_package,
        "ONNX Runtime input shape mismatch for tensor: input",
    )

    bad_dtype_package = args.output.parent / "onnxruntime-bad-dtype.pnmir"
    _write_package(bad_dtype_package)
    bad_dtype_manifest_path = bad_dtype_package / "model.json"
    bad_dtype_manifest = json.loads(bad_dtype_manifest_path.read_text(encoding="utf-8"))
    bad_dtype_manifest["inputs"][0]["dtype"] = "uint8"
    bad_dtype_manifest_path.write_text(
        json.dumps(bad_dtype_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _assert_run_failure(
        args.pnmir,
        bad_dtype_package,
        "ONNX Runtime input dtype mismatch for tensor: input",
    )

    bad_output_package = args.output.parent / "onnxruntime-bad-output.pnmir"
    _write_package(bad_output_package)
    bad_output_manifest_path = bad_output_package / "model.json"
    bad_output_manifest = json.loads(
        bad_output_manifest_path.read_text(encoding="utf-8")
    )
    bad_output_manifest["outputs"][0]["name"] = "manifest_output"
    bad_output_manifest_path.write_text(
        json.dumps(bad_output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _assert_run_failure(
        args.pnmir,
        bad_output_package,
        "ONNX Runtime model has no output named: manifest_output",
    )

    bad_version_package = args.output.parent / "onnxruntime-bad-version.pnmir"
    _write_package(bad_version_package)
    bad_version_manifest_path = bad_version_package / "model.json"
    bad_version_manifest = json.loads(
        bad_version_manifest_path.read_text(encoding="utf-8")
    )
    bad_version_manifest["artifacts"][0]["runtime_version"] = "0.0.0"
    bad_version_manifest_path.write_text(
        json.dumps(bad_version_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _assert_run_failure(
        args.pnmir,
        bad_version_package,
        "no compatible backend artifact for: onnxruntime",
    )

    multi_io_package = args.output.parent / "onnxruntime-multi-io.pnmir"
    _write_multi_io_package(multi_io_package)
    left_file = args.output.parent / "onnxruntime-left.f32"
    right_file = args.output.parent / "onnxruntime-right.f32"
    left_file.write_bytes(struct.pack("=3f", 5.0, 7.0, 9.0))
    right_file.write_bytes(struct.pack("=3f", 1.0, 2.0, 3.0))
    multi_io_result = _run(
        [
            str(args.pnmir),
            "run",
            str(multi_io_package),
            "--input-file",
            f"left={left_file}",
            "--input-file",
            f"right={right_file}",
            "--backend",
            "onnxruntime",
        ],
        "ONNX Runtime multi-input/multi-output inference",
    )
    assert multi_io_result.stdout.strip().splitlines() == [
        "difference: 4 5 6",
        "sum: 6 9 12",
    ], multi_io_result.stdout

    difference_file = args.output.parent / "onnxruntime-difference.f32"
    sum_file = args.output.parent / "onnxruntime-sum.f32"
    multi_io_file_result = _run(
        [
            str(args.pnmir),
            "run",
            str(multi_io_package),
            "--input-file",
            f"left={left_file}",
            "--input-file",
            f"right={right_file}",
            "--output-file",
            f"difference={difference_file}",
            "--output-file",
            f"sum={sum_file}",
            "--backend",
            "onnxruntime",
        ],
        "ONNX Runtime named output-file inference",
    )
    assert struct.unpack("=3f", difference_file.read_bytes()) == (4.0, 5.0, 6.0)
    assert struct.unpack("=3f", sum_file.read_bytes()) == (6.0, 9.0, 12.0)
    assert multi_io_file_result.stdout.strip().splitlines() == [
        f"difference: wrote 12 bytes to {difference_file}",
        f"sum: wrote 12 bytes to {sum_file}",
    ], multi_io_file_result.stdout

    print(result.stdout.strip())
    print(file_result.stdout.strip())
    print(multi_io_result.stdout.strip())
    print(multi_io_file_result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
