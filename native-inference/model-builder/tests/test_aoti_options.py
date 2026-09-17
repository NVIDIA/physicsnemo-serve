"""Compiler-option validation remains usable without importing Torch."""

from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_export.aoti_options import validate_aoti_options


class AotiOptionTests(unittest.TestCase):
    def test_supported_booleans_are_copied_without_defaults_or_mutation(self):
        options = {"max_autotune": True, "epilogue_fusion": False,
                   "shape_padding": True, "coordinate_descent_tuning": True}
        validated = validate_aoti_options(options)
        self.assertEqual(validated, options)
        self.assertIsNot(validated, options)
        self.assertEqual(validate_aoti_options({}), {})

    def test_unknown_and_builder_owned_options_are_rejected(self):
        for key in ("max_autotune_typo", "triton.cudagraphs", "fallback_by_default",
                    "aot_inductor.output_path", "post_grad_custom_pre_pass"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unsupported AOTI option"):
                validate_aoti_options({key: True})

    def test_options_must_be_an_object(self):
        for options in (None, [], True, 1, "max-autotune"):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "aoti_options must be an object"):
                validate_aoti_options(options)

    def test_boolean_values_are_strictly_typed(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                validate_aoti_options({"max_autotune": value})

    def test_epilogue_fusion_requires_explicit_autotuning(self):
        for options in ({"epilogue_fusion": True},
                        {"epilogue_fusion": True, "max_autotune": False}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "epilogue_fusion.*max_autotune"):
                validate_aoti_options(options)
        options = {"epilogue_fusion": True, "max_autotune": True}
        self.assertEqual(validate_aoti_options(options), options)
        self.assertEqual(validate_aoti_options({"epilogue_fusion": False}), {"epilogue_fusion": False})

    def test_exact_profile_rejects_enabled_performance_options(self):
        for key in ("max_autotune", "epilogue_fusion", "shape_padding", "coordinate_descent_tuning"):
            options = {key: True}
            if key == "epilogue_fusion":
                options["max_autotune"] = True
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "aten-boundary-exact-v2"):
                validate_aoti_options(options, "aten-boundary-exact-v2")
        self.assertEqual(validate_aoti_options({"max_autotune": False}, "aten-boundary-exact-v2"), {"max_autotune": False})

    def test_validation_is_framework_free(self):
        command = [sys.executable, "-S", "-c",
                   "import sys; sys.path.insert(0, sys.argv[1]); "
                   "from pnmir_export.aoti_options import validate_aoti_options; "
                   "assert validate_aoti_options({'max_autotune': True}) == {'max_autotune': True}; "
                   "assert not {'torch', 'onnx', 'tensorrt'} & sys.modules.keys()",
                   str(Path(__file__).resolve().parents[1] / "src")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
