"""Installed recipes follow wheel RECORD paths across Python install schemes."""

from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli


class InstalledAssetDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.python_prefix = self.root / "python-prefix"
        self.install_prefix = self.root / "package-install-prefix"
        self.site_packages = self.install_prefix / "lib/python3.12/dist-packages"
        self.site_packages.mkdir(parents=True)
        self.resources = self.install_prefix / "share/pnms-model-builder"
        (self.resources / "models/affine").mkdir(parents=True)
        (self.resources / "toolchain.lock.json").write_text('{"format_version":1}')
        (self.resources / "models/affine/recipe.json").write_text(
            json.dumps({"name": "affine", "adapter": "export.py"})
        )
        (self.resources / "models/affine/export.py").write_text("# canonical adapter\n")
        self.dist_info = self.site_packages / "pnms_model_builder-0.1.0.dist-info"
        self.dist_info.mkdir()
        (self.dist_info / "METADATA").write_text(
            "Metadata-Version: 2.4\nName: pnms-model-builder\nVersion: 0.1.0\n"
        )
        relative_lock = os.path.relpath(
            self.resources / "toolchain.lock.json", self.site_packages
        )
        (self.dist_info / "RECORD").write_text(f"{relative_lock},,\n")
        self.distribution = metadata.PathDistribution(self.dist_info)
        self.module_path = self.site_packages / "model_builder/build/cli.py"

    def resolved_assets(self):
        with (
            mock.patch.object(cli, "__file__", str(self.module_path)),
            mock.patch.object(sys, "prefix", str(self.python_prefix)),
            mock.patch(
                "importlib.metadata.distribution", return_value=self.distribution
            ),
        ):
            try:
                return cli.assets_root()
            except cli.UsageError:
                return None

    def test_installed_recipe_uses_record_when_install_prefix_differs_from_python(self):
        self.assertNotEqual(self.install_prefix, self.python_prefix)
        self.assertEqual(
            self.resolved_assets(),
            self.resources,
            "wheel RECORD must locate recipe data outside sys.prefix",
        )

    def test_installed_record_wins_over_an_unrelated_prefix_resource_directory(self):
        decoy = self.python_prefix / "share/pnms-model-builder"
        (decoy / "models").mkdir(parents=True)
        (decoy / "toolchain.lock.json").write_text('{"unrelated":true}')
        self.assertEqual(
            self.resolved_assets(),
            self.resources,
            "installed recipes must belong to this distribution, not a guessed prefix",
        )

    def test_incomplete_recorded_resources_are_rejected(self):
        (self.resources / "toolchain.lock.json").unlink()
        self.assertIsNone(self.resolved_assets())


if __name__ == "__main__":
    unittest.main()
