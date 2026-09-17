"""Export customization is optional, backend scoped and reference preserving."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import worker


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class AdapterHookTest(unittest.TestCase):
    def test_adapter_hook_is_retained_without_running_before_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "exporter.py").write_text("""
import torch
events = []
class Model(torch.nn.Module):
    def forward(self, value):
        events.append("reference")
        return value * 2
def create_model(): return Model()
def create_cases(): return [(torch.ones(4),)]
def export_options(context):
    assert events == ["reference"]
    return context
""")
            recipe = dict(
                format_version=1,
                adapter="exporter.py",
                factory="create_model",
                cases="create_cases",
                input_names=["input"],
                output_names=["output"],
                dtype="float32",
                shape=[4],
            )
            prepared = worker._prepare_model(recipe, root / "recipe.json", "cpu")
            callback = prepared.get("export_options")
            self.assertTrue(callable(callback), "the optional adapter hook was lost")
            token = object()
            self.assertIs(callback(token), token)
            self.assertEqual(len(prepared["references"]), 1)

    def test_backend_hook_is_optional_and_onnx_passes_cannot_leak_into_aoti(self):
        import torch
        from pnmir_export import ExportOptions
        from pnmir_export.compat import NormalizeClampBounds

        recipe = dict(
            name="affine",
            version="1",
            input_names=["input"],
            output_names=["output"],
            dtype="float32",
            shape=[4],
        )
        prepared = dict(model=torch.nn.Identity(), cases=[(torch.ones(4),)])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("pnmir_export.exporter.export_package") as export:
                worker._build_backend(
                    "aoti", prepared, recipe, "cpu", root / "pkg", root / "plain"
                )
                export.assert_called_once()
            hook = mock.Mock(
                return_value=ExportOptions(onnx_passes=(NormalizeClampBounds(),))
            )
            prepared["export_options"] = hook
            with mock.patch("pnmir_export.exporter.export_package") as export:
                with self.assertRaisesRegex(ValueError, "ONNX graph passes require"):
                    worker._build_backend(
                        "aoti", prepared, recipe, "cpu", root / "pkg", root / "hook"
                    )
                export.assert_not_called()
            self.assertEqual(hook.call_args.args[0].backend, "aoti")

    def test_tensorrt_receives_hook_options_and_restores_cuda_device(self):
        import torch
        from pnmir_export import ExportOptions
        from pnmir_export.compat import NormalizeClampBounds

        options = ExportOptions(onnx_passes=(NormalizeClampBounds(),))
        hook = mock.Mock(return_value=options)
        prepared = dict(
            model=torch.nn.Identity(), cases=[(torch.ones(4),)], export_options=hook
        )
        recipe = dict(
            name="affine", version="1", input_names=["input"], output_names=["output"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch("torch.cuda.current_device", return_value=3),
                mock.patch("torch.cuda.set_device") as set_device,
                mock.patch(
                    "pnmir_export.onnx_exporter.export_onnx_model",
                    return_value=root / "model.onnx",
                ) as export,
                mock.patch("pnmir_export.tensorrt_builder.build_tensorrt_package"),
            ):
                worker._build_backend(
                    "tensorrt", prepared, recipe, "cuda:1", root / "pkg", root / "graph"
                )
                self.assertIs(export.call_args.kwargs["options"], options)
                self.assertEqual(hook.call_args.args[0].backend, "tensorrt")
                self.assertEqual(
                    set_device.call_args_list, [mock.call(1), mock.call(3)]
                )


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires Torch")
class ClampPassTest(unittest.TestCase):
    def test_scalar_float_and_integer_bounds_normalize_before_decomposition(self):
        import torch
        from pnmir_export.compat import NormalizeClampBounds

        class Model(torch.nn.Module):
            def forward(self, value):
                return value.clamp(0.5, 5)

        inputs = (torch.tensor([-2.0, 0.0, 1.0, 10.0]),)
        exported = torch.export.export(Model(), inputs, strict=False)
        self.assertEqual(NormalizeClampBounds()(exported.graph_module), 1)
        exported.validate()
        self.assertTrue(torch.equal(exported.module()(*inputs), Model()(*inputs)))

    def graph(self, *, scalar_lower=True, keywords=False, dtype=None):
        import torch

        dtype = dtype or torch.float32
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        bound = graph.placeholder("bound")
        value.meta["val"] = torch.empty(4, dtype=dtype)
        bound.meta["val"] = torch.empty(4, dtype=dtype)
        lower, upper = (0.5, bound) if scalar_lower else (bound, 0.5)
        clamped = graph.call_function(
            torch.ops.aten.clamp.default,
            (value,) if keywords else (value, lower, upper),
            {"min": lower, "max": upper} if keywords else {},
        )
        clamped.meta["val"] = torch.empty(4, dtype=dtype)
        graph.output(clamped)
        return torch.fx.GraphModule(torch.nn.Module(), graph)

    def test_broadcast_bounds_order_and_tensor_values_are_preserved(self):
        import torch
        from pnmir_export.compat import NormalizeClampBounds

        value = torch.tensor([-2.0, 0.25, 1.0, 10.0])
        for scalar_lower in (True, False):
            for keywords in (True, False):
                with self.subTest(scalar_lower=scalar_lower, keywords=keywords):
                    module = self.graph(scalar_lower=scalar_lower, keywords=keywords)
                    self.assertEqual(NormalizeClampBounds()(module), 1)
                    self.assertEqual(NormalizeClampBounds()(module), 0)
                    for bound in (
                        torch.tensor([-1.0, 0.0, 2.0, 4.0]),
                        torch.tensor([3.0, 4.0, -1.0, 2.0]),
                    ):
                        lower, upper = (
                            (torch.tensor(0.5), bound)
                            if scalar_lower
                            else (bound, torch.tensor(0.5))
                        )
                        expected = torch.minimum(torch.maximum(value, lower), upper)
                        self.assertTrue(torch.equal(module(value, bound), expected))

    def test_unsupported_dtypes_fail_with_an_actionable_error(self):
        import torch
        from pnmir_export.compat import NormalizeClampBounds

        with self.assertRaisesRegex(ValueError, "requires FP32"):
            NormalizeClampBounds()(self.graph(dtype=torch.float64))

    def test_passes_run_before_onnx_on_a_private_program_and_are_recorded(self):
        import torch
        from pnmir_export import ExportOptions
        from pnmir_export.onnx_exporter import export_onnx_model

        model = torch.nn.Linear(4, 4).eval()
        original = model.weight.detach().clone()
        events = []

        def first(module):
            events.append("first")
            return 0

        def second(module):
            events.append("second")
            return 0

        def onnx(program, *args, **kwargs):
            self.assertIsInstance(program, torch.export.ExportedProgram)
            self.assertEqual(events, ["first", "second"])
            self.assertNotEqual(
                program.state_dict["weight"].data_ptr(), model.weight.data_ptr()
            )
            program.state_dict["weight"].zero_()
            return mock.Mock()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            with mock.patch("torch.onnx.export", side_effect=onnx):
                export_onnx_model(
                    model,
                    (torch.ones(1, 4),),
                    path,
                    input_names=("input",),
                    output_names=("output",),
                    device=torch.device("cpu"),
                    options=ExportOptions(onnx_passes=(first, second)),
                )
            report = json.loads(path.with_suffix(".export-options.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual([x["rewritten_nodes"] for x in report["passes"]], [0, 0])
            self.assertTrue(all(len(x["code_sha256"]) == 64 for x in report["passes"]))
        self.assertTrue(torch.equal(model.weight, original))

    def test_failed_pass_stops_onnx_conversion_and_records_failure(self):
        import torch
        from pnmir_export import ExportOptions
        from pnmir_export.onnx_exporter import export_onnx_model

        def broken(module):
            raise ValueError("unsupported graph pattern")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.onnx"
            with mock.patch("torch.onnx.export") as export:
                with self.assertRaisesRegex(ValueError, "unsupported graph pattern"):
                    export_onnx_model(
                        torch.nn.Identity(),
                        (torch.ones(4),),
                        path,
                        input_names=("input",),
                        output_names=("output",),
                        device=torch.device("cpu"),
                        options=ExportOptions(onnx_passes=(broken,)),
                    )
                export.assert_not_called()
            report = json.loads(path.with_suffix(".export-options.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
