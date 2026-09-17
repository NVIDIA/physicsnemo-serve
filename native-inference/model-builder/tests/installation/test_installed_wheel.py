"""Validate a built wheel in a fresh environment with no ML dependencies.

Run with PNMIR_TEST_WHEEL=/absolute/path/to/the.whl python test_installed_wheel.py.
"""

import configparser
from email.parser import Parser
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv
import zipfile


class InstalledWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel = Path(os.environ["PNMIR_TEST_WHEEL"]).resolve(strict=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="pnmir-wheel-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.environment = cls.root / "environment"
        venv.EnvBuilder(with_pip=True).create(cls.environment)
        cls.python = cls.environment / "bin/python"
        cls.command = cls.environment / "bin/physicsnemo-model-builder"
        cls.other = cls.root / "unrelated-working-directory"
        cls.other.mkdir()
        cls.env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        cls.env["PYTHONNOUSERSITE"] = "1"
        result = subprocess.run(
            [
                str(cls.python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                str(cls.wheel),
            ],
            env=cls.env,
            cwd=cls.other,
            text=True,
            capture_output=True,
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(
                f"Wheel installation failed: {result.stdout}\n{result.stderr}"
            )

    def _cli_without_ml(self, *arguments):
        site = subprocess.check_output(
            [
                str(self.python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=self.other,
            env=self.env,
            text=True,
            timeout=15,
        ).strip()
        probe = """
import importlib.util, pathlib, sys
site = pathlib.Path(sys.argv.pop(1))
sys.path.insert(0, str(site))
assert sys.flags.isolated and sys.flags.no_site
from pnmir_build import cli
assert pathlib.Path(cli.__file__).is_relative_to(site)
try:
    result = cli.main(sys.argv[1:])
except SystemExit as error:
    result = error.code
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('physicsnemo') is None
assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
assert not any(name == 'physicsnemo' or name.startswith('physicsnemo.') for name in sys.modules)
raise SystemExit(result)
"""
        return subprocess.run(
            [str(self.python), "-I", "-S", "-c", probe, site, *map(str, arguments)],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_installed_help_and_import_do_not_require_torch(self):
        probe = subprocess.run(
            [
                str(self.python),
                "-I",
                "-c",
                "import importlib.util, sys; from pnmir_build import cli; "
                'assert importlib.util.find_spec("torch") is None; '
                'assert "torch" not in sys.modules',
            ],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        result = subprocess.run(
            [str(self.command), "--help"],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PhysicsNeMo Model Builder", result.stdout)

    def test_installed_list_resolves_packaged_recipe_from_unrelated_directory(self):
        result = subprocess.run(
            [str(self.command), "list", "--json"],
            cwd=self.other,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        models = json.loads(result.stdout)["models"]
        self.assertEqual([item["name"] for item in models], ["affine"])
        self.assertEqual(models[0]["supported_backends"], ["aoti", "tensorrt"])
        resources = self.environment / "share/physicsnemo-model-builder"
        recipe = json.loads((resources / "models/affine/recipe.json").read_text())
        self.assertTrue((resources / "models/affine" / recipe["adapter"]).is_file())
        self.assertTrue((resources / "toolchain.lock.json").is_file())

    def test_wheel_contains_both_python_packages_and_canonical_resource_layout(self):
        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
        self.assertIn("pnmir_build/cli.py", names)
        self.assertIn("pnmir_export/exporter.py", names)
        self.assertFalse(any("geotransolver-surface-core" in name for name in names))
        self.assertNotIn("pnmir_build/workflow_cli.py", names)
        self.assertNotIn("pnmir_build/workflows.py", names)
        for suffix in (
            "/data/share/physicsnemo-model-builder/toolchain.lock.json",
            "/data/share/physicsnemo-model-builder/models/affine/recipe.json",
            "/data/share/physicsnemo-model-builder/models/affine/export.py",
        ):
            self.assertTrue(
                any(name.endswith(suffix) for name in names),
                f"missing packaged resource {suffix}",
            )
        self.assertFalse(
            any(
                name.startswith(("tests/", "models/", "cpp-runtime/")) for name in names
            )
        )

    def test_metadata_exposes_only_the_builder_command_and_optional_ml_dependencies(
        self,
    ):
        with zipfile.ZipFile(self.wheel) as archive:
            metadata_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_name).decode())
            entrypoints_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/entry_points.txt")
            )
            entrypoints = configparser.ConfigParser()
            entrypoints.read_string(archive.read(entrypoints_name).decode())
        self.assertEqual(metadata["Name"], "physicsnemo-model-builder")
        requirements = metadata.get_all("Requires-Dist", [])
        self.assertTrue(any(value.startswith("torch") for value in requirements))
        self.assertTrue(
            all("extra ==" in value for value in requirements), requirements
        )
        self.assertEqual(
            dict(entrypoints["console_scripts"]),
            {"physicsnemo-model-builder": "pnmir_build.cli:main"},
        )


if __name__ == "__main__":
    unittest.main()
