"""DoMINO exact arithmetic must preserve eager bytes and compiler boundaries."""

from pathlib import Path
from contextlib import contextmanager
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_export import aoti_profiles, compat
from pnmir_export.aoti_options import validate_aoti_options


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class DominoProfileTests(unittest.TestCase):
    def test_builder_loads_captured_sidecar_and_declares_its_abi(self):
        import torch
        from pnmir_build import worker
        from pnmir_export.domino_exact import REQUIRED_OPERATORS

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libpnmir_domino_exact_ops.so"
            prepared = {
                "model": torch.nn.Identity(), "cases": [(torch.ones(2),)],
                "assets": {"domino_exact_ops": library},
            }
            recipe = {
                "name": "domino", "version": "1", "input_names": ["input"],
                "output_names": ["output"], "aoti_profile": "aten-boundary-exact-v3",
            }
            with mock.patch("pnmir_export.domino_exact.load_exact_ops") as loader, mock.patch("pnmir_export.exporter.export_package") as exporter:
                worker._build_backend("aoti", prepared, recipe, "cpu", root / "package", root / "exported")
                loader.assert_called_once_with(library)
                self.assertEqual(exporter.call_args.kwargs["required_operators"], REQUIRED_OPERATORS)
                self.assertEqual(exporter.call_args.kwargs["aoti_profile"], "aten-boundary-exact-v3")
            recipe["aoti_profile"] = "baseline"
            with self.assertRaisesRegex(ValueError, "requires aoti_profile"):
                worker._build_backend("aoti", prepared, recipe, "cpu", root / "bad-package", root / "bad-exported")

    def test_v3_disables_shape_padding_and_restores_configuration(self):
        class Config:
            shape_padding = True
            emulate_divison_rounding = False
            fallback_by_default = False
            selective_decompose = False
            post_grad_custom_pre_pass = None

            @contextmanager
            def patch(self, settings):
                previous = {key: getattr(self, key) for key in settings}
                try:
                    for key, value in settings.items():
                        setattr(self, key, value)
                    yield
                finally:
                    for key, value in previous.items():
                        setattr(self, key, value)

        config = Config()

        self.assertIn(
            "aten-boundary-exact-v3",
            getattr(aoti_profiles, "EXACT_PROFILES", ()),
            "DoMINO needs the qualified v3 compiler profile",
        )
        before = config.shape_padding
        with mock.patch("torch._inductor.config", config), aoti_profiles.compiler_profile("aten-boundary-exact-v3") as metadata:
            self.assertFalse(config.shape_padding)
            self.assertEqual(metadata["version"], 3)
            self.assertIs(metadata["applied_compiler_settings"]["shape_padding"], False)
        self.assertEqual(config.shape_padding, before)
        with self.assertRaisesRegex(ValueError, "not qualified"):
            validate_aoti_options({"shape_padding": True}, "aten-boundary-exact-v3")

    def test_tensor_only_rewrite_preserves_eager_output_and_signature(self):
        import torch

        transform = getattr(compat, "DoMINOExactBoundaryPass", None)
        self.assertIsNotNone(transform, "DoMINO requires tensor-only AOTI boundaries")
        # CPU implementations characterize the graph transform. Native sidecar
        # implementations are separately exercised through the H100 C++ runtime.
        library = torch.library.Library("pnmir_domino", "FRAGMENT")
        definitions = {
            "tensor_scalar_add": ("(Tensor value, Tensor scalar) -> Tensor", lambda a, b: a + b.item()),
            "tensor_scalar_div": ("(Tensor value, Tensor scalar) -> Tensor", lambda a, b: a / b.item()),
            "tensor_scalar_mul": ("(Tensor value, Tensor scalar) -> Tensor", lambda a, b: a * b.item()),
            "tensor_sub": ("(Tensor left, Tensor right) -> Tensor", lambda a, b: a - b),
            "reciprocal": ("(Tensor value) -> Tensor", torch.reciprocal),
            "vector_norm_last_dim": ("(Tensor value) -> Tensor", lambda a: torch.linalg.vector_norm(a, dim=-1, keepdim=True)),
        }
        for name, (schema, implementation) in definitions.items():
            library.define(name + schema)
            library.impl(name, implementation, "CPU")

        class Model(torch.nn.Module):
            def forward(self, left, right):
                shifted = right + 1e-6
                distance = torch.linalg.vector_norm(left - shifted, dim=-1, keepdim=True)
                return (left / 10.0) * 0.5 + torch.reciprocal(distance)

        args = (torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[0.2, 0.4, 0.7]]))
        model = Model()
        expected = model(*args)
        exported = torch.export.export(model, args)
        signature = str(exported.graph_signature)
        self.assertEqual(transform()(exported.graph_module), 6)
        exported.validate()
        self.assertEqual(str(exported.graph_signature), signature)
        actual = exported.module()(*args)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
        self.assertEqual(transform()(exported.graph_module), 0)
        # Tensor/tensor arithmetic and non-default scalar alpha stay untouched.
        class Unsupported(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, 0.3, alpha=2) + a / b
        exported = torch.export.export(Unsupported(), args)
        self.assertEqual(transform()(exported.graph_module), 0)


if __name__ == "__main__":
    unittest.main()
