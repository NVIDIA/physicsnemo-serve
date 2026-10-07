"""AOTI option forwarding/publication; real compilation is qualified on H100."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aoti_test_support import FakeConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires optional Torch")
class AotiOptionExportTests(unittest.TestCase):
    def setUp(self):
        import torch
        from model_builder.export import exporter

        self.torch = torch
        self.module = exporter
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "package"
        self.model = torch.nn.ReLU()
        self.example = (torch.tensor([[-1.0, 0.0, 2.0]]),)
        self.observed = []
        self.export = mock.Mock(return_value=object())
        self.compile = mock.Mock(side_effect=self.compile_package)
        self.native = mock.Mock(return_value=(self.model(*self.example),))
        for name, replacement in (
            ("_strict_export", self.export),
            ("_validate_exported_program", mock.Mock(return_value=())),
            ("_run_isolated_aoti_package", self.native),
        ):
            patch = mock.patch.object(exporter, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(
            torch._inductor, "aoti_compile_and_package", self.compile
        )
        patch.start()
        self.addCleanup(patch.stop)

    def compile_package(self, exported, **kwargs):
        configs = kwargs.get("inductor_configs")
        self.observed.append(None if configs is None else dict(configs))
        # The real API adds package bookkeeping to the supplied dictionary.
        if configs is not None:
            configs["aot_inductor.package"] = True
        Path(kwargs["package_path"]).write_bytes(b"API-double package")

    def build(self, options=None, **kwargs):
        if options is not None:
            kwargs["aoti_options"] = options
        return self.module.export_package(
            self.model,
            self.example,
            self.output,
            model_name="relu",
            model_version="test",
            input_names=("features",),
            output_names=("predictions",),
            **kwargs,
        )

    def test_requested_options_reach_compiler_and_manifest_without_mutation(self):
        options = {"max_autotune": True, "epilogue_fusion": False}
        self.build(options)
        self.assertEqual(self.observed, [options])
        self.assertEqual(options, {"max_autotune": True, "epilogue_fusion": False})
        artifact = json.loads((self.output / "model.json").read_text())["artifacts"][0]
        self.assertEqual(artifact["path"], "model.pt2")
        metadata = artifact["compiler_options"]
        self.assertEqual(metadata["requested"], options)
        self.assertEqual(metadata["applied"], options)
        self.assertTrue(metadata["effective"]["max_autotune"])
        self.assertFalse(metadata["effective"]["epilogue_fusion"])
        self.assertIsInstance(metadata["effective"]["shape_padding"], bool)
        self.assertEqual(metadata["torch_version"], str(self.torch.__version__))
        self.assertEqual(metadata["torch_git_version"], self.torch.version.git_version)
        self.assertEqual(metadata["cuda_version"], self.torch.version.cuda)
        self.native.assert_called_once()

    def test_omitted_and_empty_options_preserve_existing_compile_call(self):
        for options in (None, {}):
            with self.subTest(options=options):
                self.build(options, force=True)
                self.assertIsNone(self.observed[-1])
                artifact = json.loads((self.output / "model.json").read_text())[
                    "artifacts"
                ][0]
                self.assertNotIn("compiler_options", artifact)

    def test_effective_options_include_active_profile_and_explicit_overrides(self):
        config = FakeConfig()
        config.max_autotune = True
        config.shape_padding = True
        observed_shape_padding = []

        def compile_package(exported, **kwargs):
            observed_shape_padding.append(config.shape_padding)
            self.compile_package(exported, **kwargs)

        self.compile.side_effect = compile_package
        for profile, options, profile_padding, effective_padding in (
            ("baseline", {"max_autotune": False, "shape_padding": False}, True, False),
            ("aten-boundary-exact-v2", {"max_autotune": False}, True, True),
            ("aten-boundary-exact-v3", {"max_autotune": False}, False, False),
        ):
            with (
                self.subTest(profile=profile),
                mock.patch("torch._inductor.config", config),
            ):
                self.build(options, aoti_profile=profile, force=True)
                self.assertEqual(observed_shape_padding[-1], profile_padding)
                self.assertEqual(self.observed[-1], options)
                self.assertTrue(config.shape_padding)
                self.assertTrue(config.max_autotune)
                self.assertFalse(config.emulate_divison_rounding)
                self.assertIsNone(config.post_grad_custom_pre_pass)
                artifact = json.loads((self.output / "model.json").read_text())[
                    "artifacts"
                ][0]
                metadata = artifact["compiler_options"]
                self.assertEqual(metadata["requested"], options)
                self.assertEqual(metadata["applied"], options)
                self.assertFalse(metadata["effective"]["max_autotune"])
                self.assertEqual(
                    metadata["effective"]["shape_padding"], effective_padding
                )
                if profile == "aten-boundary-exact-v3":
                    self.assertFalse(
                        artifact["correctness_profile"]["applied_compiler_settings"][
                            "shape_padding"
                        ]
                    )

    def test_invalid_option_is_rejected_before_export_and_output_creation(self):
        with self.assertRaisesRegex(ValueError, "Unsupported AOTI option"):
            self.build({"max_autotune_typo": True})
        self.export.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_unavailable_option_is_rejected_before_export(self):
        with (
            mock.patch("torch._inductor.config", SimpleNamespace(max_autotune=False)),
            mock.patch.object(self.module, "compiler_profile") as profile,
        ):
            with self.assertRaisesRegex(ValueError, "installed Torch.*shape_padding"):
                self.build(
                    {"shape_padding": False}, aoti_profile="aten-boundary-exact-v3"
                )
        profile.assert_not_called()
        self.export.assert_not_called()
        self.compile.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_compile_preserves_previous_package_and_caller_options(self):
        self.output.mkdir()
        (self.output / "model.pt2").write_bytes(b"previous package")
        (self.output / "model.json").write_text('{"previous":true}')
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.compile.side_effect = RuntimeError("intentional compiler failure")
        options = {"max_autotune": False}
        config = FakeConfig()
        config.max_autotune = True
        config.shape_padding = True
        with mock.patch("torch._inductor.config", config):
            with self.assertRaisesRegex(RuntimeError, "intentional compiler failure"):
                self.build(options, force=True, aoti_profile="aten-boundary-exact-v3")
        self.assertTrue(config.shape_padding)
        self.assertTrue(config.max_autotune)
        self.assertFalse(config.emulate_divison_rounding)
        self.assertIsNone(config.post_grad_custom_pre_pass)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.output.iterdir()}, before
        )
        self.assertEqual(options, {"max_autotune": False})

    def test_options_do_not_bypass_native_parity(self):
        self.native.return_value = (self.torch.zeros_like(self.example[0]),)
        with self.assertRaisesRegex(AssertionError, "failed parity"):
            self.build({"max_autotune": True})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
