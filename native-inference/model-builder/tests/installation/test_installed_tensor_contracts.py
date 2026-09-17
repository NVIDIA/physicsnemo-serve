"""Exercise named tensor recipe checks from an installed wheel without Torch."""

import copy
import json
import subprocess
import unittest

import test_installed_wheel


class InstalledTensorContractTests(test_installed_wheel.InstalledWheelTests):
    def _recipe(self):
        return {
            "format_version": 2,
            "name": "named-stage",
            "version": "1",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "inputs": [
                {"name": "coordinates", "dtype": "float32", "shape": [1, 8, 3]},
                {"name": "features", "dtype": "float32", "shape": [1, 8, 6]},
                {"name": "condition", "dtype": "float32", "shape": [1, 2]},
            ],
            "outputs": [{"name": "prediction", "dtype": "float32", "shape": [1, 8, 4]}],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict", "path": "weights.pt"},
        }

    def _doctor_without_site(self, recipe, directory):
        model = self.other / directory
        model.mkdir()
        (model / "export.py").write_text(
            "raise RuntimeError('doctor must not import adapter')\n"
        )
        (model / "config.json").write_text("{}\n")
        (model / "weights.pt").write_bytes(b"doctor must not deserialize weights")
        recipe_path = model / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        output = model / "output"
        installed_site = subprocess.check_output(
            [
                str(self.python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            env=self.env,
            cwd=self.other,
            text=True,
            timeout=15,
        ).strip()
        # -I -S excludes checkout paths, user site and all automatic site imports.
        # Add only this fresh wheel installation, which has no ML dependencies.
        probe = (
            "import importlib.util, pathlib, sys; "
            "site = pathlib.Path(sys.argv.pop(1)); "
            "sys.path.insert(0, str(site)); "
            "assert sys.flags.no_site and sys.flags.isolated; "
            "assert importlib.util.find_spec('torch') is None; "
            "from pnmir_build import cli; "
            "assert pathlib.Path(cli.__file__).is_relative_to(site); "
            "status = cli.main(sys.argv[1:]); "
            "assert 'torch' not in sys.modules; "
            "raise SystemExit(status)"
        )
        result = subprocess.run(
            [
                str(self.python),
                "-I",
                "-S",
                "-c",
                probe,
                installed_site,
                "doctor",
                "--recipe",
                str(recipe_path),
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
        self.assertFalse(output.exists(), "doctor must not create a build")
        self.assertEqual(json.loads(recipe_path.read_text()), recipe)
        return result

    def test_installed_doctor_accepts_independent_tensor_shapes_without_torch(self):
        result = self._doctor_without_site(self._recipe(), "valid-named-stage")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "configuration-ok")
        self.assertEqual(report["backends"], ["aoti"])
        self.assertEqual(report["device"], "cpu")

    def test_installed_doctor_rejects_malformed_tensor_contracts_without_torch(self):
        original = self._recipe()
        malformed = []
        recipe = copy.deepcopy(original)
        recipe["inputs"][0]["shape"] = [1, -1, 3]
        malformed.append(("dynamic-shape", recipe, "static positive shape"))
        recipe = copy.deepcopy(original)
        recipe["outputs"][0]["dtype"] = "float16"
        malformed.append(("unsupported-dtype", recipe, "float32"))
        recipe = copy.deepcopy(original)
        recipe["inputs"][1]["name"] = recipe["inputs"][0]["name"]
        malformed.append(("duplicate-name", recipe, "unique safe inputs"))
        recipe = copy.deepcopy(original)
        del recipe["outputs"]
        malformed.append(("missing-output", recipe, "outputs tensor descriptors"))
        recipe = copy.deepcopy(original)
        recipe["shape"] = [4]
        malformed.append(("mixed-schemas", recipe, "legacy"))
        for name, recipe, diagnostic in malformed:
            with self.subTest(name=name):
                result = self._doctor_without_site(recipe, name)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("configuration-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
