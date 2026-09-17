"""Freeze parameter gates with native Torch rounding, without touching references."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_export import compat
from pnmir_export.graph_passes import prepare_onnx_program
from pnmir_export.options import ExportOptions


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class SigmoidGatePassTests(unittest.TestCase):
    def transform(self):
        transform = getattr(compat, "FreezeScalarSigmoidGates", None)
        self.assertIsNotNone(
            transform, "Scalar parameter sigmoid gates need an exact export pass"
        )
        return transform()

    def test_freezes_only_scalar_parameter_sigmoids_and_keeps_reference_weights(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Parameter(torch.tensor(-0.31234567))
                self.vector = torch.nn.Parameter(torch.tensor([-0.2, 0.7]))

            def forward(self, left, right):
                gate = torch.sigmoid(self.gate)
                return (
                    gate * left
                    + (1 - gate) * right
                    + torch.sigmoid(left)
                    + torch.sigmoid(self.vector)
                )

        model = Model().eval()
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        args = (torch.tensor([1.7, -2.3]), torch.tensor([-4.5, 3.2]))
        transform = self.transform()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "passes.json"
            exported = prepare_onnx_program(
                model, args, ExportOptions(onnx_passes=(transform,)), report_path
            )
            self.assertEqual(
                json.loads(report_path.read_text())["passes"][0]["rewritten_nodes"], 1
            )
            sigmoid_nodes = [
                n
                for n in exported.graph.nodes
                if n.target == torch.ops.aten.sigmoid.default
            ]
            self.assertEqual(len(sigmoid_nodes), 2)
            self.assertTrue(torch.equal(exported.module()(*args), model(*args)))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]))
                self.assertNotEqual(
                    value.data_ptr(), exported.state_dict[name].data_ptr()
                )
            self.assertEqual(transform(exported.graph_module), 0)

    def test_gate_is_evaluated_on_its_parameter_device(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = torch.nn.Parameter(torch.tensor(0.73))

            def forward(self, value):
                return torch.ops.aten.sigmoid.default(self.gate) * value

        transform = self.transform()
        calls = []
        sigmoid = torch.sigmoid

        def observed(value):
            calls.append(value.device)
            return sigmoid(value)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("torch.sigmoid", side_effect=observed),
        ):
            exported = prepare_onnx_program(
                Model(),
                (torch.ones(2),),
                ExportOptions(onnx_passes=(transform,)),
                Path(directory) / "passes.json",
            )
        self.assertEqual(calls, [torch.device("cpu")])
        self.assertTrue(
            torch.equal(
                exported.module()(torch.ones(2)),
                sigmoid(torch.tensor(0.73)) * torch.ones(2),
            )
        )

    def test_nonfinite_or_non_fp32_scalar_parameters_fail_closed(self):
        import torch

        class Model(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.gate = torch.nn.Parameter(value)

            def forward(self, value):
                return torch.sigmoid(self.gate) * value

        transform = self.transform()
        for value in (
            torch.tensor(float("nan")),
            torch.tensor(0.2, dtype=torch.float64),
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "finite FP32"):
                    prepare_onnx_program(
                        Model(value),
                        (torch.ones(2),),
                        ExportOptions(onnx_passes=(transform,)),
                        Path(directory) / "passes.json",
                    )


if __name__ == "__main__":
    unittest.main()
