"""AOTI output contracts follow the real exported graph's symbolic shapes."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class AotiDynamicOutputsTests(unittest.TestCase):
    def setUp(self):
        import torch
        from model_builder.export import exporter

        self.torch = torch
        self.exporter = exporter
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "package"
        self.example = (torch.arange(8, dtype=torch.float32).reshape(2, 4),)

    def build(
        self,
        model,
        names,
        dynamic_shapes,
        *,
        input_names=("features",),
        required_operators=(),
        compiled=None,
    ):
        def compile_package(exported, *, package_path):
            self.assertIsInstance(exported, self.torch.export.ExportedProgram)
            self.exported = exported
            Path(package_path).write_bytes(b"compiler stand-in")

        def run_compiled(*inputs):
            self.native_inputs = tuple(value.clone() for value in inputs)
            operation = compiled or self.exported.module()
            return self.exporter._tensor_outputs(operation(*inputs))

        def run_package(artifact, inputs, device, validation_dir):
            return run_compiled(*inputs)

        with (
            mock.patch.object(
                self.torch._inductor,
                "aoti_compile_and_package",
                side_effect=compile_package,
            ),
            mock.patch.object(
                self.exporter, "_run_isolated_aoti_package", side_effect=run_package
            ),
            mock.patch.object(
                self.torch._inductor, "aoti_load_package", return_value=run_compiled
            ),
        ):
            self.exporter.export_package(
                model,
                self.example,
                self.output,
                model_name="dynamic-outputs",
                model_version="test",
                input_names=input_names,
                output_names=names,
                dynamic_shapes=dynamic_shapes,
                required_operators=required_operators,
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

        names = (
            "reduced",
            "transposed",
            "doubled",
            "rows",
            "flat",
            "first",
            "sum",
            "alias",
        )
        shapes = [[4], [4, -1], [-1, 4], [-1], [-1], [1, 4], [], [4, -1]]
        for as_list in (False, True):
            with self.subTest(as_list=as_list):
                manifest = self.build(
                    Transform(as_list),
                    names,
                    ({0: torch.export.Dim("batch", min=2, max=8)},),
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
                manifest = self.build(
                    self.torch.nn.Identity(), ("identity",), dynamic_shapes
                )
                self.assertEqual(
                    manifest["outputs"],
                    [{"name": "identity", "dtype": "float32", "shape": [2, 4]}],
                )

    def test_explicit_static_dimensions_keep_concrete_input_contracts(self):
        for dimension in (self.torch.export.Dim.STATIC, None):
            with self.subTest(dimension=dimension):
                manifest = self.build(
                    self.torch.nn.Identity(), ("identity",), ({0: dimension},)
                )
                self.assertEqual(manifest["inputs"][0]["shape"], [2, 4])
                valid = self.torch.ones(2, 4)
                self.assertTrue(self.torch.equal(self.exported.module()(valid), valid))
                with self.assertRaisesRegex(
                    (RuntimeError, AssertionError),
                    r"Guard failed: input\.size\(\)\[0\] == 2|Expected input.*shape.*2",
                ):
                    self.exported.module()(self.torch.ones(3, 4))

    def test_input_shapes_follow_user_order_without_parameters_or_buffers(self):
        torch = self.torch

        class Transform(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4))
                self.register_buffer("offset", torch.ones(4))

            def forward(self, value, row):
                return value * self.weight + self.offset + row

        self.example += (torch.ones(4),)
        manifest = self.build(
            Transform(),
            ("predictions",),
            (
                {
                    0: torch.export.Dim("batch", min=2, max=8),
                    1: torch.export.Dim.STATIC,
                },
                {0: None},
            ),
            input_names=("features", "row"),
        )
        self.assertEqual(
            manifest["inputs"],
            [
                {"name": "features", "dtype": "float32", "shape": [-1, 4]},
                {"name": "row", "dtype": "float32", "shape": [4]},
            ],
        )

    def test_input_mutation_preserves_caller_and_pristine_later_inputs(self):
        torch = self.torch

        class Increment(torch.nn.Module):
            def forward(self, value):
                return value.add_(1)

        original = self.example[0].clone()
        strict_export = self.exporter._strict_export
        captured = []

        def capture(model, inputs, dynamic_shapes):
            captured.append(inputs[0].clone())
            return strict_export(model, inputs, dynamic_shapes)

        for operators in ((), (("test::sidecar", "1"),)):
            with self.subTest(required_operators=operators):
                self.example = (original.clone(),)
                with mock.patch.object(
                    self.exporter, "_strict_export", side_effect=capture
                ):
                    self.build(
                        Increment(),
                        ("incremented",),
                        None,
                        required_operators=operators,
                    )
                torch.testing.assert_close(captured[-1], original)
                torch.testing.assert_close(self.native_inputs[0], original)
                torch.testing.assert_close(self.example[0], original)

    def test_input_aliased_reference_cannot_mask_incorrect_compiled_output(self):
        torch = self.torch

        class Increment(torch.nn.Module):
            def forward(self, value):
                return value.add_(1)

        for operators in ((), (("test::sidecar", "1"),)):
            with self.subTest(required_operators=operators):
                self.example = (torch.zeros(2, 4),)
                with self.assertRaisesRegex(AssertionError, "failed parity"):
                    self.build(
                        Increment(),
                        ("incremented",),
                        None,
                        required_operators=operators,
                        compiled=lambda value: value.add_(2),
                    )


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class ExportedProgramValidationTests(unittest.TestCase):
    def operation_model(self, operation):
        import torch

        class Operation(torch.nn.Module):
            def forward(self, value):
                return operation(value)

        return Operation().eval()

    def test_active_dropout_is_rejected_before_compilation_or_publication(self):
        import torch
        from model_builder.export import exporter

        operations = (
            torch.nn.functional.dropout,
            lambda value: torch.ops.aten.native_dropout.default(
                input=value, p=0.5, train=True
            )[0],
            lambda value: torch.ops.aten.native_dropout.default(value, 0.5, None)[0],
        )
        for operation in operations:
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                destination = Path(tmp) / "package"
                graph_path = Path(tmp) / "exported.pt2"

                def compile_package(exported, *, package_path):
                    Path(package_path).write_bytes(b"compiler stand-in")

                with (
                    mock.patch.object(
                        torch._inductor,
                        "aoti_compile_and_package",
                        side_effect=compile_package,
                    ) as compile_mock,
                    mock.patch.object(
                        exporter,
                        "_run_isolated_aoti_package",
                        return_value=(torch.zeros(32),),
                    ) as run_mock,
                ):
                    with self.assertRaisesRegex(
                        ValueError, "random operators.*dropout"
                    ):
                        exporter.export_package(
                            self.operation_model(operation),
                            (torch.zeros(32),),
                            destination,
                            model_name="dropout",
                            model_version="test",
                            input_names=("features",),
                            output_names=("predictions",),
                            exported_program_path=graph_path,
                        )
                    compile_mock.assert_not_called()
                    run_mock.assert_not_called()
                self.assertFalse(destination.exists())
                self.assertFalse(graph_path.exists())
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_rejects_seeded_rng_outside_the_legacy_operator_list(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        operations = (
            ("binomial", lambda value: torch.binomial(value, value * 0.5)),
            ("rrelu", lambda value: torch.nn.functional.rrelu(value, training=True)),
        )
        for name, operation in operations:
            with self.subTest(operator=name):
                exported = torch.export.export(
                    self.operation_model(operation), (torch.ones(32),), strict=True
                )
                with self.assertRaisesRegex(ValueError, f"random operators.*{name}"):
                    _validate_exported_program(exported, ())

    def test_accepts_dropout_with_proven_deterministic_arguments(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        for native in (False, True):
            for probability, training in ((0.5, False), (0.0, True), (1.0, True)):
                with self.subTest(native=native, p=probability, training=training):

                    def operation(value):
                        if native:
                            return torch.ops.aten.native_dropout.default(
                                input=value, p=probability, train=training
                            )[0]
                        return torch.nn.functional.dropout(
                            value, p=probability, training=training
                        )

                    value = torch.arange(32, dtype=torch.float32)
                    exported = torch.export.export(
                        self.operation_model(operation), (value,), strict=True
                    )
                    self.assertEqual(_validate_exported_program(exported, ()), ())
                    expected = value if not training or probability == 0 else value * 0
                    for _ in range(2):
                        torch.testing.assert_close(exported.module()(value), expected)

    def test_accepts_attention_without_dropout_and_inactive_rrelu(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        operations = (
            lambda value: torch.nn.functional.scaled_dot_product_attention(
                value, value, value
            ),
            lambda value: torch.nn.functional.rrelu(value, training=False),
        )
        value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) - 4
        for operation in operations:
            with self.subTest(operation=operation):
                model = self.operation_model(operation)
                exported = torch.export.export(model, (value,), strict=True)
                self.assertEqual(_validate_exported_program(exported, ()), ())
                torch.testing.assert_close(exported.module()(value), model(value))

    def test_accepts_attention_and_rrelu_with_deterministic_training_arguments(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        operations = (
            lambda value: torch.nn.functional.scaled_dot_product_attention(
                value, value, value, dropout_p=1.0
            ),
            lambda value: torch.ops.aten._scaled_dot_product_attention_math.default(
                value,
                value,
                value,
                dropout_p=0.5,
                dropout_mask=torch.ones((1, 1, 2, 2), dtype=torch.bool),
            )[0],
            lambda value: torch.nn.functional.rrelu(
                value, lower=0.25, upper=0.25, training=True
            ),
        )
        value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) - 4
        for operation in operations:
            with self.subTest(operation=operation), torch.random.fork_rng(devices=[]):
                model = self.operation_model(operation)
                exported = torch.export.export(model, (value,), strict=True)
                expected = model(value)
                for seed in (1, 42):
                    torch.manual_seed(seed)
                    torch.testing.assert_close(exported.module()(value), expected)
                self.assertEqual(_validate_exported_program(exported, ()), ())

    def export_conditional(self, branch):
        import torch

        class Conditional(torch.nn.Module):
            def forward(self, value):
                return torch.cond(
                    value.sum() > 0, lambda value: value + 1, branch, (value,)
                )

        return torch.export.export(Conditional(), (torch.ones(2),), strict=True)

    def test_rejects_random_operators_in_unexecuted_and_nested_cond_branches(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        def random_branch(value):
            return value + torch.rand_like(value)

        def dropout_branch(value):
            return torch.nn.functional.dropout(value)

        def nested_branch(value):
            return torch.cond(
                value.sum() < -1, lambda value: value - 1, random_branch, (value,)
            )

        for branch in (random_branch, nested_branch, dropout_branch):
            with self.subTest(branch=branch.__name__):
                exported = self.export_conditional(branch)
                operator = "dropout" if branch is dropout_branch else "rand_like"
                with self.assertRaisesRegex(
                    ValueError, f"random operators.*{operator}"
                ):
                    _validate_exported_program(exported, ())

    def test_nested_custom_operators_require_declaration_and_report_namespace(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        library = torch.library.Library("pnmir_nested_validation_test", "DEF")
        self.addCleanup(library._destroy)
        library.define("increment(Tensor value) -> Tensor")
        library.impl("increment", lambda value: value + 1, "CPU")
        torch.library.register_fake(
            "pnmir_nested_validation_test::increment",
            lambda value: torch.empty_like(value),
            lib=library,
        )

        def custom_branch(value):
            return torch.ops.pnmir_nested_validation_test.increment(value)

        exported = self.export_conditional(custom_branch)
        with self.assertRaisesRegex(
            ValueError,
            "undeclared custom operator namespaces: pnmir_nested_validation_test",
        ):
            _validate_exported_program(exported, ())
        self.assertEqual(
            _validate_exported_program(exported, (("test::increment", "1"),)),
            ("pnmir_nested_validation_test",),
        )

    def test_accepts_deterministic_nested_cond_branches(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        def nested_branch(value):
            return torch.cond(
                value.sum() < -1,
                lambda value: value - 1,
                lambda value: value * 2,
                (value,),
            )

        exported = self.export_conditional(nested_branch)
        self.assertEqual(_validate_exported_program(exported, ()), ())
        torch.testing.assert_close(
            exported.module()(torch.ones(2)), torch.full((2,), 2.0)
        )
        torch.testing.assert_close(
            exported.module()(-torch.ones(2)), torch.full((2,), -2.0)
        )

    def test_graph_only_export_stand_ins_remain_supported(self):
        import torch
        from model_builder.export.exporter import _validate_exported_program

        exported = torch.export.export(
            torch.nn.Identity(), (torch.ones(2),), strict=True
        )
        for stand_in in (
            SimpleNamespace(graph=exported.graph),
            mock.Mock(graph=exported.graph),
        ):
            with self.subTest(stand_in=type(stand_in).__name__):
                self.assertEqual(_validate_exported_program(stand_in, ()), ())


if __name__ == "__main__":
    unittest.main()
