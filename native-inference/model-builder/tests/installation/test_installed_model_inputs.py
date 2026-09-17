"""Validate custom format-2 inputs using an installed wheel without Torch."""

import json
import subprocess
import unittest

import test_installed_wheel


class InstalledModelInputTests(test_installed_wheel.InstalledWheelTests):
    def test_installed_doctor_resolves_custom_inputs_without_torch(self):
        model = self.other / "custom"
        model.mkdir()
        (model / "export.py").write_text(
            "raise RuntimeError('doctor must not import adapter')\n"
        )
        (model / "config.json").write_text('{"offset": 3}')
        checkpoint = model / "weights.pt"
        checkpoint.write_bytes(b"configuration-only check does not deserialize weights")
        recipe = {
            "format_version": 2,
            "name": "custom",
            "version": "1",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "input_names": ["input"],
            "output_names": ["output"],
            "dtype": "float32",
            "shape": [4],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict"},
        }
        (model / "recipe.json").write_text(json.dumps(recipe))
        output = model / "output"
        result = subprocess.run(
            [
                str(self.command),
                "doctor",
                "--recipe",
                str(model / "recipe.json"),
                "--checkpoint",
                str(checkpoint),
                "--executor",
                "local",
                "--runtime",
                str(self.python),
                "--device",
                "cpu",
                "--output",
                str(output),
            ],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "configuration-ok")
        self.assertFalse(output.exists())
        checkpoint.unlink()
        missing = subprocess.run(
            [
                str(self.command),
                "doctor",
                "--recipe",
                str(model / "recipe.json"),
                "--checkpoint",
                str(checkpoint),
                "--executor",
                "local",
                "--runtime",
                str(self.python),
                "--device",
                "cpu",
            ],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(missing.returncode, 2, missing.stderr)
        self.assertIn("checkpoint", missing.stderr)
        self.assertIn("regular file", missing.stderr)


if __name__ == "__main__":
    unittest.main()
