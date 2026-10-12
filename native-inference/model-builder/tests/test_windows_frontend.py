"""Native Windows entry points, execution selection and package paths need no ML imports."""

import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from model_builder.build import cli, scaffold, worker  # noqa: E402
import worker_test_support  # noqa: E402


class WindowsFrontendTests(unittest.TestCase):
    def test_module_entry_point_runs_the_frontend(self):
        env = dict(os.environ, PYTHONPATH=str(SOURCE))
        result = subprocess.run(
            [sys.executable, "-S", "-m", "model_builder.build.cli", "list", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"models"', result.stdout)
        self.assertEqual(json.loads(result.stdout)["models"][0]["name"], "affine")

    def test_windows_project_defaults_to_native_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(sys, "platform", "win32"):
                result = scaffold.initialize(Path(temporary))
            document = json.loads((Path(temporary) / "model-build.json").read_text())
            self.assertEqual(document["executor"], "local")
            self.assertEqual(document["backends"], ["aoti"])
            self.assertTrue(any("runtime" in step for step in result["next_steps"]))

    def test_module_entry_point_preserves_setup_configuration_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "model_builder.build.cli",
                    "setup-env",
                    temporary,
                    "--json",
                ],
                env=dict(os.environ, PYTHONPATH=str(SOURCE)),
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["diagnostics"][0]["code"],
            "ENVIRONMENT_CONFIGURATION",
        )

    def test_windows_accepts_both_backends_with_an_exe_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "physicsnemo-infer.exe"
            runtime.write_bytes(b"native executable fixture; never executed")
            runtime.chmod(0o755)
            args = argparse.Namespace(executor="local", runtime=runtime, lock=None)
            plan = {"executor": "local", "backends": ["aoti", "tensorrt"]}
            with mock.patch.object(sys, "platform", "win32"):
                resolved = cli.resolve_executor(args, cli.assets_root(), plan)
            self.assertEqual(resolved["runtime"], runtime.resolve())
            self.assertEqual(resolved["backends"], ["aoti", "tensorrt"])

    def test_windows_container_selection_explains_native_alternative(self):
        args = argparse.Namespace(
            executor="container",
            lock=None,
            runtime=None,
            builder_image="sha256:" + "a" * 64,
        )
        with (
            mock.patch.object(sys, "platform", "win32"),
            mock.patch.object(cli.shutil, "which", return_value="docker.exe"),
        ):
            with self.assertRaisesRegex(cli.UsageError, "--executor local.*WSL"):
                cli.resolve_executor(args, cli.assets_root(), {"executor": "container"})


class WindowsPackagePathTests(unittest.TestCase):
    def setUp(self):
        self.fixture = worker_test_support.WorkerFixture()
        self.fixture.addCleanup = self.addCleanup
        if hasattr(self.fixture, "setUp"):
            self.fixture.setUp()

    def test_windows_inventory_uses_forward_slashes(self):
        path = mock.Mock(wraps=self.fixture.recipe_path)
        path.relative_to.return_value = PureWindowsPath("source/recipe.json")
        record = worker._file_identity(path, self.fixture.root)
        self.assertEqual(record["path"], "source/recipe.json")
        self.assertNotIn("\\", record["path"])

    def test_windows_backend_package_path_matches_inventory_convention(self):
        original_relative_to = Path.relative_to

        def windows_package_relative_to(path, *args, **kwargs):
            relative = original_relative_to(path, *args, **kwargs)
            if path.parent.name == "backends":
                return PureWindowsPath(*relative.parts)
            return relative

        with (
            mock.patch.object(Path, "relative_to", windows_package_relative_to),
            mock.patch.object(worker, "_native_case", return_value={"outputs": []}),
        ):
            receipt = self.fixture.run_build()
        for backend, variant in receipt["variants"].items():
            self.assertEqual(variant["package"], f"backends/{backend}")
            self.assertIn(
                variant["package"] + "/model.json",
                [item["path"] for item in variant["files"]],
            )


if __name__ == "__main__":
    unittest.main()
