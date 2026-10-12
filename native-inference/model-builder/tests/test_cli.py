import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli
from authoring_test_support import linux_container
from model_input_test_support import CliRecipeFixture

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "pnms-model-builder"


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

    def test_config_only_validates_legacy_tensor_contracts_without_execution(self):
        recipe = json.loads(
            (ROOT / "model-builder" / "models" / "affine" / "recipe.json").read_text()
        )
        cases = (
            ({}, None),
            ({"shape": [0]}, "shape"),
            ({"shape": [-1]}, "shape"),
            ({"shape": [True]}, "shape"),
            ({"shape": []}, "shape"),
            ({"dtype": "float64"}, "float32"),
            ({"input_names": ["input", "input"]}, "input_names"),
            ({"output_names": ["../output"]}, "output_names"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "recipe.json"
            output = root / "candidate"
            (root / "export.py").write_text(
                "raise AssertionError('config-only must not import adapter')\n"
            )
            (root / "config.json").write_text("{}")
            (root / "weights.pt").write_bytes(b"must not deserialize weights")
            for version in (1, 2):
                if version == 2:
                    recipe.update(
                        config={"path": "config.json"},
                        checkpoint={"format": "torch-state-dict", "path": "weights.pt"},
                    )
                for changes, diagnostic in cases:
                    with self.subTest(version=version, changes=changes):
                        path.write_text(
                            json.dumps(recipe | {"format_version": version} | changes)
                        )
                        result = self.invoke(
                            "check",
                            "--config-only",
                            "--recipe",
                            str(path),
                            "--executor",
                            "local",
                            "--device",
                            "cpu",
                            "--runtime",
                            sys.executable,
                            "--output",
                            str(output),
                            "--json",
                        )
                        self.assertEqual(
                            result.returncode,
                            2 if diagnostic else 0,
                            result.stdout + result.stderr,
                        )
                        report = json.loads(result.stdout)
                        self.assertEqual(
                            report["status"],
                            "failed" if diagnostic else "configuration-ok",
                        )
                        if diagnostic:
                            self.assertEqual(
                                report["diagnostics"][0]["code"], "INVALID_ARGUMENT"
                            )
                            self.assertIn(
                                diagnostic, report["diagnostics"][0]["message"]
                            )
                        self.assertFalse(output.exists())
                        self.assertFalse((root / "model-build.lock.json").exists())

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


class FrontendReviewTests(CliRecipeFixture, unittest.TestCase):
    @linux_container()
    def test_explicit_recipe_takes_precedence_over_current_authoring_project(self):
        (self.source / "model-build.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "name": "unselected-project",
                    "version": "1",
                    "adapter": "export.py",
                    "source": [],
                    "checkpoint": None,
                }
            )
        )
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.source)

        def launch(plan):
            self.assertEqual(plan["recipe_path"], self.recipe_path.resolve())
            self.assertIsNone(plan.get("project"))
            plan["output"].mkdir()
            return 0

        for operation in ("check", "build"):
            for recipe in (str(self.recipe_path), self.recipe_path.name):
                with self.subTest(operation=operation, recipe=recipe):
                    output = self.root / f"{operation}-{Path(recipe).is_absolute()}"
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        mock.patch.object(
                            cli.shutil, "which", return_value="/usr/bin/docker"
                        ),
                        mock.patch.object(
                            cli, "container_build", side_effect=launch
                        ) as build,
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        code = cli.main(
                            [
                                operation,
                                *(["--config-only"] if operation == "check" else []),
                                "--recipe",
                                recipe,
                                "--output",
                                str(output),
                                "--lock",
                                str(self.lock),
                                "--device",
                                "cpu",
                                "--json",
                            ]
                        )
                    self.assertEqual(code, 0, stdout.getvalue() + stderr.getvalue())
                    result = json.loads(stdout.getvalue())
                    self.assertEqual(
                        result["status"],
                        "configuration-ok" if operation == "check" else "complete",
                    )
                    if operation == "check":
                        build.assert_not_called()
                        self.assertFalse(output.exists())
                    else:
                        build.assert_called_once()
        self.assertFalse((self.source / "model-build.lock.json").exists())

    def test_rejects_absolute_and_lexical_traversal_even_when_source_is_contained(self):
        for adapter in (str(self.source / "export.py"), "../source/export.py"):
            with self.subTest(adapter=adapter):
                self.recipe_path.write_text(
                    json.dumps(dict(self.recipe, adapter=adapter))
                )
                with self.assertRaisesRegex(cli.UsageError, "adapter"):
                    cli.read_recipe(self.recipe_path)

    def test_local_receipt_explicitly_has_no_image_and_survives_worker_error(self):
        runtime = self.root / "physicsnemo-infer"
        runtime.write_text("#!/bin/sh\nexit 0\n")
        runtime.chmod(0o755)
        for failed in (False, True):
            with self.subTest(failed=failed):
                self.output = self.root / f"local-{failed}"

                def execute(*args):
                    self.output.mkdir()
                    if failed:
                        raise RuntimeError("compiler failed")
                    return {"status": "complete"}

                with mock.patch(
                    "model_builder.build.worker.execute_build", side_effect=execute
                ):
                    self.assertEqual(
                        self.invoke("--executor", "local", "--runtime", str(runtime)),
                        int(failed),
                    )
                receipt_path = self.output / "execution.json"
                self.assertTrue(
                    receipt_path.is_file(),
                    "local execution provenance must survive worker errors",
                )
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(receipt["executor"], "local")
                self.assertIsNone(receipt["builder_image"])
                self.assertEqual(receipt["exit_code"], int(failed))
                self.assertEqual(receipt["selection_source"], "argument")


if __name__ == "__main__":
    unittest.main()
