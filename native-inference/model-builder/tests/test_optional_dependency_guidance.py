"""Optional dependency failures point users to installable distributions."""

from pathlib import Path
import subprocess
import sys
import unittest


class OptionalDependencyGuidanceTests(unittest.TestCase):
    def test_missing_backend_dependencies_have_current_install_commands(self):
        script = """
import sys, types
sys.path.insert(0, sys.argv[1])
# Only dtype names are needed to import these modules; no model executes.
torch = types.ModuleType('torch')
for name in ('float32', 'float16', 'bfloat16', 'int32', 'int64', 'uint8'):
    setattr(torch, name, object())
sys.modules['torch'] = torch
sys.modules['onnx'] = None
sys.modules['tensorrt'] = None
from pnmir_export.onnx_importer import _onnx_module
from pnmir_export.tensorrt_builder import _tensorrt_module
for get_module in (_onnx_module, _tensorrt_module):
    try:
        get_module()
    except RuntimeError as error:
        print(error)
    else:
        raise AssertionError('missing dependency must fail explicitly')
"""
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                script,
                str(Path(__file__).resolve().parents[1] / "src"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        messages = result.stdout.splitlines()
        self.assertEqual(len(messages), 2)
        for message, install in zip(
            messages,
            ("python -m pip install onnx", "physicsnemo-model-builder[tensorrt]"),
            strict=True,
        ):
            with self.subTest(install=install):
                self.assertIn(install, message)


if __name__ == "__main__":
    unittest.main()
