"""Customer initialization reports actionable requirements before execution."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "physicsnemo-model-builder"


class AuthoringCommandTests(unittest.TestCase):
    def invoke(self, *args, python=None):
        result = subprocess.run(
            [python or sys.executable, "-S", str(LAUNCHER), *args, "--json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode, json.loads(result.stdout), result.stderr

    def test_incomplete_init_explains_checkpoint_and_hooks_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 0, result)
            self.assertEqual(result["status"], "initialized")
            original = {p.name: p.read_bytes() for p in project.iterdir()}
            for operation in ("doctor", "check", "build"):
                code, result, error = self.invoke(operation, str(project))
                self.assertEqual(code, 2, result)
                fields = {item.get("field") for item in result["diagnostics"]}
                self.assertIn("checkpoint", fields)
                self.assertIn("adapter", fields)
                self.assertEqual(
                    original, {p.name: p.read_bytes() for p in project.iterdir()}
                )

    def test_checkpoint_can_be_configured_after_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 0, result)
            config_path = project / "model-build.json"
            config = json.loads(config_path.read_text())
            # Doctor must identify bytes without attempting to deserialize them.
            (project / "weights.pt").write_bytes(
                b"opaque checkpoint for metadata validation"
            )
            config.update(checkpoint="weights.pt", executor="local", device="cpu")
            config_path.write_text(json.dumps(config))
            (project / "build_adapter.py").write_text(
                "def create_model(config, assets):\n    return None\n"
                "def create_cases(config, assets):\n    return []\n"
            )
            code, result, error = self.invoke("doctor", str(project))
            self.assertEqual(code, 0, result)
            self.assertEqual(result["status"], "configuration-ok")
            self.assertFalse((project / "model-build.lock.json").exists())
            self.assertFalse((project / "builds").exists())

    def test_existing_adapter_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            adapter = project / "build_adapter.py"
            adapter.write_text("existing customer code\n")
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 2, result)
            self.assertEqual(adapter.read_text(), "existing customer code\n")
            self.assertFalse((project / "model-build.json").exists())


if __name__ == "__main__":
    unittest.main()
