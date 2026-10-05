"""AOTI output contracts follow the real exported graph's symbolic shapes."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class AotiDynamicOutputsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from pnmir_export import exporter

        self.torch = torch
        self.exporter = exporter
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "package"
        self.example = (torch.arange(8, dtype=torch.float32).reshape(2, 4),)

    def build(self, model, names, dynamic_shapes):
        def compile_package(exported, *, package_path):
            self.assertIsInstance(exported, self.torch.export.ExportedProgram)
            self.exported = exported
            Path(package_path).write_bytes(b"compiler stand-in")

        def run_package(artifact, inputs, device, validation_dir):
            return self.exporter._tensor_outputs(self.exported.module()(*inputs))

        with (
            mock.patch.object(
                self.torch._inductor,
                "aoti_compile_and_package",
                side_effect=compile_package,
            ),
            mock.patch.object(
                self.exporter, "_run_isolated_aoti_package", side_effect=run_package
            ),
        ):
            self.exporter.export_package(
                model,
                self.example,
                self.output,
                model_name="dynamic-outputs",
                model_version="test",
                input_names=("features",),
                output_names=names,
                dynamic_shapes=dynamic_shapes,
                force=True,
            )
        return json.loads((self.output / "model.json").read_text())

    def test_batch_preserving_output_has_dynamic_batch_contract(self):
        torch = self.torch

        class Double(torch.nn.Module):
            def forward(self, value):
                return value * 2

        manifest = self.build(
            Double(), ("predictions",), ({0: torch.export.Dim("batch", min=2, max=8)},)
        )
        actual = self.exported.module()(torch.ones(3, 4))
        self.assertEqual(list(actual.shape), [3, 4])
        self.assertEqual(manifest["inputs"][0]["shape"], [-1, 4])
        self.assertEqual(
            manifest["outputs"],
            [{"name": "predictions", "dtype": "float32", "shape": [-1, 4]}],
        )

    def test_output_order_and_only_symbolic_axes_are_preserved(self):
        torch = self.torch

        class Transform(torch.nn.Module):
            def __init__(self, as_list):
                super().__init__()
                self.as_list = as_list

            def forward(self, value):
                transposed = value.T
                outputs = (
                    value.sum(dim=0),
                    transposed,
                    torch.cat((value, value), dim=0),
                    value.sum(dim=1),
                    value.reshape(-1),
                    value[:1],
                    value.sum(),
                    transposed,
                )
                return list(outputs) if self.as_list else outputs

        names = ("reduced", "transposed", "doubled", "rows", "flat", "first", "sum", "alias")
        shapes = [[4], [4, -1], [-1, 4], [-1], [-1], [1, 4], [], [4, -1]]
        for as_list in (False, True):
            with self.subTest(as_list=as_list):
                manifest = self.build(
                    Transform(as_list), names, ({0: torch.export.Dim("batch", min=2, max=8)},)
                )
                actual = self.exported.module()(torch.ones(3, 4))
                self.assertEqual(
                    [list(value.shape) for value in actual],
                    [[4], [4, 3], [6, 4], [3], [12], [1, 4], [], [4, 3]],
                )
                self.assertEqual(
                    manifest["outputs"],
                    [
                        {"name": name, "dtype": "float32", "shape": shape}
                        for name, shape in zip(names, shapes)
                    ],
                )

    def test_static_export_keeps_concrete_output_dimensions(self):
        for dynamic_shapes in (None, ({},)):
            with self.subTest(dynamic_shapes=dynamic_shapes):
                manifest = self.build(self.torch.nn.Identity(), ("identity",), dynamic_shapes)
                self.assertEqual(
                    manifest["outputs"],
                    [{"name": "identity", "dtype": "float32", "shape": [2, 4]}],
                )


if __name__ == "__main__":
    unittest.main()
