"""Native Windows entry points and execution selection need no ML imports."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from pnmir_build import cli, scaffold  # noqa: E402


class WindowsFrontendTests(unittest.TestCase):
    def test_module_entry_point_runs_the_frontend(self):
        env = dict(os.environ, PYTHONPATH=str(SOURCE))
        result = subprocess.run(
            [sys.executable, "-S", "-m", "pnmir_build.cli", "list", "--json"],
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
                    "pnmir_build.cli",
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


if __name__ == "__main__":
    unittest.main()
