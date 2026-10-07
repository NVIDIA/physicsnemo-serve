"""Exercise device-runner file boundaries and concrete output shapes on CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


class DeviceRunnerTests(unittest.TestCase):
    runner: Path
    backend: str

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pnmir-device-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input_bytes = struct.pack("=3f", 1.0, 2.0, 3.0)
        self.expected_bytes = struct.pack("=3f", 3.0, 5.0, 7.0)

    def fixture(self, root, *, input_name="input", output_name="output", dynamic=False):
        package = root / "package"
        package.mkdir(parents=True)
        inputs = root / "inputs"
        inputs.mkdir()
        outputs = root / "outputs"
        outputs.mkdir()
        artifact = "model.onnx" if self.backend == "onnxruntime" else "model.plan"
        if self.backend == "onnxruntime":
            self.write_onnx(package / artifact, input_name, output_name, dynamic)
        else:
            self.write_tensorrt(package / artifact, input_name, output_name)
        shape = [-1] if dynamic else [3]
        (package / "model.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "model": {"name": "device-runner-affine", "version": "1"},
                    "inputs": [
                        {"name": input_name, "dtype": "float32", "shape": shape}
                    ],
                    "outputs": [
                        {"name": output_name, "dtype": "float32", "shape": shape}
                    ],
                    "artifacts": [
                        {
                            "backend": self.backend,
                            "target": "cuda",
                            "precision": "fp32",
                            "path": artifact,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return package, inputs, outputs

    @staticmethod
    def write_onnx(path, input_name, output_name, dynamic):
        import onnx
        from onnx import TensorProto, helper

        shape = ["elements"] if dynamic else [3]
        graph = helper.make_graph(
            [
                helper.make_node("Mul", [input_name, "scale"], ["scaled"]),
                helper.make_node("Add", ["scaled", "bias"], [output_name]),
            ],
            "device-runner-affine",
            [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, shape)],
            [
                helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
                helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0]),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9
        )
        onnx.checker.check_model(model)
        onnx.save(model, path)

    @staticmethod
    def write_tensorrt(path, input_name, output_name):
        import numpy as np
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        source = network.add_input(input_name, trt.float32, (3,))
        scale = network.add_constant((1,), np.array([2.0], dtype=np.float32))
        bias = network.add_constant((1,), np.array([1.0], dtype=np.float32))
        scaled = network.add_elementwise(
            source, scale.get_output(0), trt.ElementWiseOperation.PROD
        )
        result = network.add_elementwise(
            scaled.get_output(0), bias.get_output(0), trt.ElementWiseOperation.SUM
        )
        result.get_output(0).name = output_name
        network.mark_output(result.get_output(0))
        engine = builder.build_serialized_network(
            network, builder.create_builder_config()
        )
        if engine is None:
            raise RuntimeError("could not build TensorRT affine fixture")
        path.write_bytes(bytes(engine))

    def run_runner(self, fixture, *arguments, preexec_fn=None):
        package, inputs, outputs = fixture
        return subprocess.run(
            [
                str(self.runner),
                str(package),
                self.backend,
                str(inputs),
                str(outputs),
                "1",
                "2",
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            preexec_fn=preexec_fn,
        )

    def assert_success(self, result, outputs):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((outputs / "output.bin").read_bytes(), self.expected_bytes)
        self.assertIn("benchmark: warmup=1 iterations=2 ", result.stdout)

    def test_static_affine(self):
        fixture = self.fixture(self.root)
        (fixture[1] / "input.bin").write_bytes(self.input_bytes)
        self.assert_success(self.run_runner(fixture), fixture[2])

    @unittest.skipUnless(os.name == "posix", "requires POSIX file-size limits")
    def test_buffered_output_write_failure(self):
        import resource
        import signal

        def reject_file_writes():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

        fixture = self.fixture(self.root)
        (fixture[1] / "input.bin").write_bytes(self.input_bytes)
        result = self.run_runner(fixture, preexec_fn=reject_file_writes)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cannot write tensor file:", result.stderr)
        self.assertNotIn("benchmark:", result.stdout)
        self.assertEqual((fixture[2] / "output.bin").stat().st_size, 0)

    def test_dynamic_output_uses_resolved_shape(self):
        if self.backend != "onnxruntime":
            self.skipTest("TensorRT backend currently requires static tensor contracts")
        fixture = self.fixture(self.root, dynamic=True)
        (fixture[1] / "input.bin").write_bytes(self.input_bytes)
        result = self.run_runner(fixture, "--shape", "input=3", "--shape", "output=3")
        self.assert_success(result, fixture[2])

    def test_dynamic_output_requires_shape_override(self):
        if self.backend != "onnxruntime":
            self.skipTest("TensorRT backend currently requires static tensor contracts")
        fixture = self.fixture(self.root, dynamic=True)
        (fixture[1] / "input.bin").write_bytes(self.input_bytes)
        result = self.run_runner(fixture, "--shape", "input=3")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("dynamic tensor requires --shape: output", result.stderr)
        self.assertFalse((fixture[2] / "output.bin").exists())

    def test_input_names_cannot_escape_directory(self):
        self.assert_names_cannot_escape("input")

    def test_output_names_cannot_escape_directory(self):
        self.assert_names_cannot_escape("output")

    def assert_names_cannot_escape(self, role):
        for path_kind in ("traversal", "absolute"):
            with self.subTest(path_kind=path_kind):
                root = self.root / path_kind
                outside = root / "outside.bin"
                name = (
                    "../outside"
                    if path_kind == "traversal"
                    else str(outside.with_suffix(""))
                )
                fixture = self.fixture(root, **{f"{role}_name": name})
                sentinel = (
                    self.input_bytes if role == "input" else b"keep this file intact"
                )
                outside.write_bytes(sentinel)
                if role == "output":
                    (fixture[1] / "input.bin").write_bytes(self.input_bytes)
                result = self.run_runner(fixture)
                self.assertEqual(
                    outside.read_bytes(),
                    sentinel,
                    "runner changed a file outside its output directory",
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("tensor name must be a filename", result.stderr)
                self.assertEqual(list(fixture[2].iterdir()), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("onnxruntime", "tensorrt"), default="onnxruntime"
    )
    args, remaining = parser.parse_known_args()
    if args.backend == "tensorrt":
        import torch

        if not torch.cuda.is_available():
            print("CUDA is not available; skipping TensorRT device-runner tests")
            raise SystemExit(77)
    DeviceRunnerTests.runner = args.runner.resolve()
    DeviceRunnerTests.backend = args.backend
    unittest.main(argv=[__file__, *remaining])
