"""External examples use generic recipes without bundled model registration."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli


class ExternalExampleTests(unittest.TestCase):
    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(list(arguments))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_discovery_exposes_recipes_without_model_specific_preparation_commands(
        self,
    ):
        code, stdout, stderr = self.invoke("--help")
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("prepare", stdout)
        code, stdout, stderr = self.invoke("list", "--json")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(
            [item["name"] for item in json.loads(stdout)["models"]], ["affine"]
        )
        self.assertEqual(self.invoke("list", "--preparations", "--json")[0], 2)
        code, stdout, stderr = self.invoke("build", "--help")
        self.assertEqual(code, 0, stderr)
        for flag in ("--points", "--geometry-points", "--producer-dir"):
            self.assertNotIn(flag, stdout)

    def test_geo_name_uses_normal_unknown_recipe_handling(self):
        code, stdout, stderr = self.invoke(
            "doctor", "geotransolver-surface-core", "--json"
        )
        self.assertEqual(code, 2, stdout + stderr)
        message = json.loads(stdout)["diagnostics"][0]["message"]
        self.assertIn("recipe", message.lower())
        self.assertNotIn("checkpoint", message.lower())

    def test_explicit_geo_recipe_resolves_without_importing_model_or_framework(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = json.loads(
                (cli.assets_root() / "models/affine/recipe.json").read_text()
            )
            recipe["name"] = "geotransolver-surface-core"
            recipe["adapter"] = "adapter.py"
            (root / "recipe.json").write_text(json.dumps(recipe))
            (root / "adapter.py").write_text(
                "raise AssertionError('doctor must not import model')\n"
            )
            result = self.invoke(
                "doctor",
                "--recipe",
                str(root / "recipe.json"),
                "--executor",
                "local",
                "--runtime",
                sys.executable,
                "--device",
                "cpu",
                "--output",
                str(root / "output"),
                "--json",
            )
            self.assertEqual(result[0], 0, result[1] + result[2])
            self.assertEqual(json.loads(result[1])["status"], "configuration-ok")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
