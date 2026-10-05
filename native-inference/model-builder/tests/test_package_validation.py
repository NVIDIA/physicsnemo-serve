"""Package validation passes concrete inputs and checks observed output shapes."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class PackageValidationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from pnmir_export import validation

        self.torch = torch
        self.validation = validation
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.executable = self.root / "physicsnemo-infer"
        self.executable.touch()

    def manifest(self, input_shapes, output_shapes):
        self.input_specs = [
            {"name": f"input_{index}", "dtype": "float32", "shape": shape}
            for index, shape in enumerate(input_shapes)
        ]
        self.output_specs = [
            {"name": f"output_{index}", "dtype": "float32", "shape": shape}
            for index, shape in enumerate(output_shapes)
        ]
        (self.package / "model.json").write_text(
            json.dumps({"inputs": self.input_specs, "outputs": self.output_specs})
        )

    def native_run(self, outputs, *, input_shapes=(), dynamic_outputs=False):
        def run(command, **kwargs):
            self.assertEqual(
                [command[index + 1] for index, value in enumerate(command)
                 if value == "--input-shape"],
                list(input_shapes),
            )
            output_files = [command[index + 1] for index, value in enumerate(command)
                            if value == "--output-file"]
            for argument, spec, value in zip(
                output_files, self.output_specs, outputs, strict=True
            ):
                name, path = argument.split("=", 1)
                self.assertEqual(name, spec["name"])
                Path(path).write_bytes(value.contiguous().numpy().tobytes())
            if dynamic_outputs:
                self.assertIn("--output-metadata", command)
                metadata = Path(command[command.index("--output-metadata") + 1])
                metadata.write_text(json.dumps({
                    "schema_version": 1,
                    "backend": command[command.index("--backend") + 1],
                    "execution_device": {"type": "cpu", "index": 0},
                    "completed": True,
                    "outputs": [
                        {
                            "name": spec["name"],
                            "dtype": "float32",
                            "shape": list(value.shape),
                            "device": {"type": "cpu", "index": 0},
                            "byte_size": value.numel() * value.element_size(),
                        }
                        for spec, value in zip(self.output_specs, outputs, strict=True)
                    ],
                }))
            else:
                self.assertNotIn("--output-metadata", command)
            return subprocess.CompletedProcess(command, 0, "", "")

        return mock.patch.object(self.validation.subprocess, "run", side_effect=run)

    def validate(self, inputs, outputs):
        return self.validation.validate_pnmir_package(
            self.executable, self.package, inputs, outputs,
            backend="aoti", device="cpu",
        )

    def test_dynamic_inputs_use_each_tensor_shape_and_keep_static_scalar_inputs(self):
        torch = self.torch
        self.manifest([[-1, 4], [2, -1, -1], [4], []], [[1]])
        outputs = (torch.ones(1),)
        inputs = (torch.ones(3, 4), torch.ones(2, 3, 5), torch.ones(4), torch.tensor(2.0))
        with self.native_run(outputs, input_shapes=("input_0=3,4", "input_1=2,3,5")):
            metrics = self.validate(inputs, outputs)
        self.assertEqual(metrics[0].max_abs, 0.0)

    def test_onnxruntime_wrapper_supplies_concrete_dynamic_input_shape(self):
        torch = self.torch
        self.manifest([[-1, 4]], [[4]])
        outputs = (torch.ones(4),)
        with self.native_run(outputs, input_shapes=("input_0=3,4",)) as run:
            metrics = self.validation.validate_onnxruntime_package(
                self.executable, self.package, (torch.ones(3, 4),), outputs,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--backend") + 1], "onnxruntime")
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(metrics[0].max_abs, 0.0)

    def test_dynamic_outputs_decode_multiple_axes_from_native_metadata(self):
        torch = self.torch
        self.manifest([[3, 4]], [[-1, -1], [-1], []])
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        outputs = (value.T, value.sum(dim=1), value.sum())
        with self.native_run(outputs, dynamic_outputs=True):
            metrics = self.validate((value,), outputs)
        self.assertEqual([metric.max_abs for metric in metrics], [0.0, 0.0, 0.0])

    def test_dynamic_output_shape_mismatch_is_not_hidden_by_equal_element_count(self):
        torch = self.torch
        self.manifest([[3, 4]], [[-1, -1]])
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        with self.native_run((value,), dynamic_outputs=True):
            with self.assertRaisesRegex(AssertionError, "shape mismatch"):
                self.validate((value,), (value.reshape(4, 3),))

    def test_static_and_scalar_outputs_keep_manifest_shapes(self):
        torch = self.torch
        self.manifest([[3, 4], []], [[4, 3], []])
        value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        scalar = torch.tensor(2.0)
        outputs = (value.T, scalar)
        with self.native_run(outputs):
            metrics = self.validate((value, scalar), outputs)
        self.assertEqual([metric.max_abs for metric in metrics], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
