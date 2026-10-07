from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.export import aoti_profiles
from aoti_test_support import FakeConfig


EXACT = "aten-boundary-exact-v2"
PASS_UUID = "pnmir-aten-boundary-exact-v2"
SETTINGS = {
    "emulate_divison_rounding": True,
    "fallback_by_default": True,
    "selective_decompose": True,
    "post_grad_custom_pre_pass": PASS_UUID,
}
HAS_TORCH = importlib.util.find_spec("torch") is not None


class ProfileFrontendTest(unittest.TestCase):
    def test_baseline_and_validation_work_without_torch(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
from model_builder.export.aoti_profiles import compiler_profile, validate_aoti_profile
assert validate_aoti_profile('baseline') == 'baseline'
assert validate_aoti_profile('aten-boundary-exact-v2') == 'aten-boundary-exact-v2'
with compiler_profile() as profile:
    assert profile['name'] == 'baseline'
    assert profile['applied_compiler_settings'] == {}
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
"""
        for flags in ([], ["-S"]):
            result = subprocess.run(
                [
                    sys.executable,
                    *flags,
                    "-c",
                    code,
                    str(Path(__file__).resolve().parents[1] / "src"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_profiles_fail_before_loading_torch(self):
        for value in ("exact", "aten-boundary-exact-v4", "", None, [], {}, 1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "AOTI profile"),
            ):
                aoti_profiles.validate_aoti_profile(value)


@unittest.skipUnless(HAS_TORCH, "requires the optional Torch dependency")
class CompilerProfileTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.config = FakeConfig()
        self.patch = mock.patch("torch._inductor.config", self.config)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_exact_controls_and_provenance_apply_only_inside_scope(self):
        with aoti_profiles.compiler_profile(EXACT) as profile:
            self.assertTrue(self.config.emulate_divison_rounding)
            self.assertTrue(self.config.fallback_by_default)
            self.assertTrue(self.config.selective_decompose)
            self.assertEqual(self.config.post_grad_custom_pre_pass.uuid(), PASS_UUID)
            self.assertEqual(profile["name"], EXACT)
            self.assertEqual(profile["version"], 2)
            self.assertEqual(profile["requested_compiler_settings"], SETTINGS)
            self.assertEqual(profile["applied_compiler_settings"], SETTINGS)
            self.assertEqual(profile["graph_pass_uuid"], PASS_UUID)
            self.assertEqual(
                profile["compiler"]["torch_version"], self.torch.__version__
            )
            self.assertEqual(len(profile["graph_pass_sha256"]), 64)
            json.dumps(profile, allow_nan=False)
        self.assertFalse(self.config.emulate_divison_rounding)
        self.assertIsNone(self.config.post_grad_custom_pre_pass)

    def test_public_export_function_preserves_import_compatibility(self):
        from model_builder.export import export_package
        from model_builder.export.exporter import export_package as implementation

        self.assertIs(export_package, implementation)

    def test_missing_required_control_fails_before_patching(self):
        for key in SETTINGS:
            with self.subTest(key=key):
                old = getattr(self.config, key)
                delattr(self.config, key)
                try:
                    with self.assertRaisesRegex(ValueError, key):
                        with aoti_profiles.compiler_profile(EXACT):
                            pass
                    self.assertEqual(self.config.patch_calls, 0)
                finally:
                    setattr(self.config, key, old)

    def test_previous_configuration_restored_when_body_raises(self):
        previous_pass = object()
        self.config.fallback_by_default = True
        self.config.post_grad_custom_pre_pass = previous_pass
        with self.assertRaisesRegex(RuntimeError, "intentional compiler failure"):
            with aoti_profiles.compiler_profile(EXACT):
                raise RuntimeError("intentional compiler failure")
        self.assertFalse(self.config.emulate_divison_rounding)
        self.assertTrue(self.config.fallback_by_default)
        self.assertFalse(self.config.selective_decompose)
        self.assertIs(self.config.post_grad_custom_pre_pass, previous_pass)

    def test_pass_identity_describes_loaded_code_when_source_location_changes(self):
        with aoti_profiles.compiler_profile(EXACT) as first:
            expected = first["graph_pass_sha256"]
        with tempfile.TemporaryDirectory() as temporary:
            replacement = Path(temporary) / "aoti_profiles.py"
            replacement.write_text("# different code, not loaded by this process\n")
            with mock.patch.object(aoti_profiles, "__file__", str(replacement)):
                with aoti_profiles.compiler_profile(EXACT) as second:
                    self.assertEqual(second["graph_pass_sha256"], expected)


@unittest.skipUnless(HAS_TORCH, "requires the optional Torch dependency")
class GraphPassTest(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch

    def graph(self, *, bias=True, alpha=1, beta=1, parameter=True):
        torch = self.torch
        root = torch.nn.Module()
        root.register_buffer(
            "weight", torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        root.register_buffer("bias", torch.tensor([0.5, -1.0]))
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        weight = graph.get_attr("weight") if parameter else graph.placeholder("weight")
        transposed = graph.call_function(torch.ops.aten.t.default, (weight,))
        if bias:
            result = graph.call_function(
                torch.ops.aten.addmm.default,
                (graph.get_attr("bias"), value, transposed),
                {"alpha": alpha, "beta": beta},
            )
        else:
            result = graph.call_function(torch.ops.aten.mm.default, (value, transposed))
        graph.output(result)
        return torch.fx.GraphModule(root, graph)

    def test_parameter_linear_restoration_preserves_executable_results(self):
        torch = self.torch
        for bias in (False, True):
            with self.subTest(bias=bias):
                module = self.graph(bias=bias)
                values = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 8
                reference = module(values)
                aoti_profiles._make_exact_pass()(module.graph)
                module.recompile()
                targets = [node.target for node in module.graph.nodes]
                self.assertIn(torch.ops.aten.linear.default, targets)
                self.assertNotIn(torch.ops.aten.addmm.default, targets)
                self.assertNotIn(torch.ops.aten.mm.default, targets)
                self.assertTrue(torch.equal(module(values), reference))

    def test_nonstandard_addmm_and_nonparameter_weights_remain_unchanged(self):
        torch = self.torch
        for kwargs in ({"alpha": 2}, {"beta": 0}, {"parameter": False}):
            with self.subTest(kwargs=kwargs):
                module = self.graph(**kwargs)
                aoti_profiles._make_exact_pass()(module.graph)
                targets = [node.target for node in module.graph.nodes]
                self.assertIn(torch.ops.aten.addmm.default, targets)
                self.assertNotIn(torch.ops.aten.linear.default, targets)

    def test_only_layout_and_scalar_full_nodes_are_forced_to_inductor(self):
        torch = self.torch
        graph = torch.fx.Graph()
        value = graph.placeholder("value")
        clone = graph.call_function(torch.ops.aten.clone.default, (value,))
        view = graph.call_function(torch.ops.aten._unsafe_view.default, (clone, [2, 2]))
        scalar = graph.call_function(torch.ops.aten.full.default, ([], 0.5))
        vector = graph.call_function(torch.ops.aten.full.default, ([2], 0.5))
        arithmetic = graph.call_function(torch.ops.aten.add.Tensor, (view, scalar))
        graph.output((arithmetic, vector))
        aoti_profiles._make_exact_pass()(graph)
        for node in (clone, view, scalar):
            self.assertIn("compile_with_inductor", node.meta.get("custom", {}))
        for node in (value, vector, arithmetic):
            self.assertNotIn("compile_with_inductor", node.meta.get("custom", {}))


@unittest.skipUnless(HAS_TORCH, "requires the optional Torch dependency")
class ExporterProfileTest(unittest.TestCase):
    def setUp(self):
        import torch
        from model_builder.export import exporter

        self.torch = torch
        self.exporter = exporter
        self.config = FakeConfig()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "package.pnmir"
        self.value = torch.tensor([0.0, 1.0])
        self.observed = []

    def compile(self, exported, *, package_path):
        self.observed.append(self.config.emulate_divison_rounding)
        Path(package_path).write_bytes(
            b"test compiler output; not an executable artifact"
        )

    def run_export(self, *, compiled=None, compiler=None, **kwargs):
        with (
            mock.patch("torch._inductor.config", self.config),
            mock.patch.object(self.exporter, "_strict_export", return_value=object()),
            mock.patch.object(
                self.exporter, "_validate_exported_program", return_value=()
            ),
            mock.patch.object(
                self.torch._inductor,
                "aoti_compile_and_package",
                side_effect=compiler or self.compile,
                create=True,
            ),
            mock.patch.object(
                self.exporter,
                "_run_isolated_aoti_package",
                return_value=(self.value if compiled is None else compiled,),
            ),
        ):
            return self.exporter.export_package(
                self.torch.nn.Identity(),
                (self.value,),
                self.output,
                model_name="identity",
                model_version="test",
                input_names=("input",),
                output_names=("output",),
                **kwargs,
            )

    def test_selected_profile_reaches_compiler_and_published_artifact(self):
        self.run_export(aoti_profile=EXACT)
        self.assertEqual(self.observed, [True])
        self.assertFalse(self.config.emulate_divison_rounding)
        manifest = json.loads((self.output / "model.json").read_text())
        self.assertIn("correctness_profile", manifest["artifacts"][0])
        profile = manifest["artifacts"][0]["correctness_profile"]
        self.assertEqual(profile["name"], EXACT)
        self.assertEqual(profile["applied_compiler_settings"], SETTINGS)

    def test_baseline_does_not_apply_precision_controls(self):
        self.run_export()
        self.assertEqual(self.observed, [False])
        self.assertEqual(self.config.patch_calls, 0)
        manifest = json.loads((self.output / "model.json").read_text())
        self.assertNotIn("correctness_profile", manifest["artifacts"][0])

    def test_compiler_failure_and_parity_failure_do_not_publish(self):
        with self.assertRaisesRegex(RuntimeError, "compile intentionally failed"):
            self.run_export(
                aoti_profile=EXACT,
                compiler=mock.Mock(
                    side_effect=RuntimeError("compile intentionally failed")
                ),
            )
        self.assertFalse(self.output.exists())
        self.assertFalse(self.config.emulate_divison_rounding)
        wrong = self.value.clone()
        wrong[0] = 2e-6
        with self.assertRaisesRegex(AssertionError, "failed parity"):
            self.run_export(aoti_profile=EXACT, compiled=wrong)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.config.emulate_divison_rounding)


if __name__ == "__main__":
    unittest.main()
