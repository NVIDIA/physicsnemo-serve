import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "physicsnemo-model-builder"


class CustomerCommandTests(unittest.TestCase):
    def invoke(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-S", str(LAUNCHER), *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_help_needs_no_ml_environment(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PhysicsNeMo Model Builder", result.stdout)
        self.assertIn("build", result.stdout)

    def test_list_finds_recipe_from_another_directory(self):
        with tempfile.TemporaryDirectory() as other:
            result = self.invoke("list", "--json", cwd=other)
        self.assertEqual(result.returncode, 0, result.stderr)
        models = json.loads(result.stdout)["models"]
        self.assertEqual(models[0]["name"], "affine")
        self.assertEqual(models[0]["supported_backends"], ["aoti", "tensorrt"])
        self.assertEqual(len(models), 1)

    def test_unknown_backend_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "candidate"
            result = self.invoke(
                "build", "affine", "--backend", "imaginary", "--output", str(output)
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("unsupported backend", result.stderr)
            self.assertFalse(output.exists())

    def test_unreleased_default_image_has_actionable_error(self):
        result = self.invoke("build", "affine")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("no released builder image", result.stderr.lower())
        self.assertIn("--builder-image", result.stderr)

    def test_local_mode_requires_prebuilt_runtime(self):
        result = self.invoke("build", "affine", "--executor", "local")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--runtime", result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_container_rejects_mutable_image_before_docker(self):
        result = self.invoke(
            "build", "affine", "--builder-image", "example/builder:latest"
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("digest", result.stderr.lower())

    def test_external_recipe_rejects_adapter_escape(self):
        with tempfile.TemporaryDirectory() as root:
            recipe = Path(root) / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "name": "custom",
                        "version": "1",
                        "adapter": "../outside.py",
                        "factory": "create_model",
                        "cases": "create_cases",
                        "input_names": ["input"],
                        "output_names": ["output"],
                        "supported_backends": ["aoti"],
                        "default_backend": "aoti",
                        "dtype": "float32",
                        "shape": [4],
                    }
                )
            )
            result = self.invoke("build", "--recipe", str(recipe))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("adapter", result.stderr.lower())
        self.assertIn("within", result.stderr.lower())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "candidate"
            output.mkdir()
            sentinel = output / "user-data.txt"
            sentinel.write_text("preserve")
            result = self.invoke("build", "affine", "--output", str(output))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("already exists", result.stderr.lower())
            self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
