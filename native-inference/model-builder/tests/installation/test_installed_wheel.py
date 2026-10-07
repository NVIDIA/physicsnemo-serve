"""Validate a built wheel in a fresh environment with no ML dependencies.

Run with PNMIR_TEST_WHEEL=/absolute/path/to/the.whl python test_installed_wheel.py.
"""

import configparser
import copy
from email.parser import Parser
import hashlib
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
        scripts = cls.environment / ("Scripts" if os.name == "nt" else "bin")
        cls.python = scripts / ("python.exe" if os.name == "nt" else "python")
        cls.command = scripts / (
            "pnms-model-builder.exe" if os.name == "nt" else "pnms-model-builder"
        )
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

    def setUp(self):
        # Share the installed wheel, while keeping every test's project isolated.
        working = tempfile.TemporaryDirectory(prefix="case-", dir=self.root)
        self.addCleanup(working.cleanup)
        self.other = Path(working.name)

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
from model_builder.build import cli
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
                "import importlib.util, sys; from model_builder.build import cli; "
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
        resources = self.environment / "share/pnms-model-builder"
        recipe = json.loads((resources / "models/affine/recipe.json").read_text())
        self.assertTrue((resources / "models/affine" / recipe["adapter"]).is_file())
        self.assertTrue((resources / "toolchain.lock.json").is_file())

    def test_wheel_contains_both_python_packages_and_canonical_resource_layout(self):
        with zipfile.ZipFile(self.wheel) as archive:
            names = archive.namelist()
        self.assertIn("model_builder/__init__.py", names)
        self.assertIn("model_builder/build/cli.py", names)
        self.assertIn("model_builder/export/exporter.py", names)
        self.assertFalse(any(name.startswith(("build/", "export/")) for name in names))
        self.assertFalse(any("geotransolver-surface-core" in name for name in names))
        self.assertNotIn("model_builder/build/workflow_cli.py", names)
        self.assertNotIn("model_builder/build/workflows.py", names)
        for suffix in (
            "/data/share/pnms-model-builder/toolchain.lock.json",
            "/data/share/pnms-model-builder/models/affine/recipe.json",
            "/data/share/pnms-model-builder/models/affine/export.py",
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
        self.assertEqual(metadata["Name"], "pnms-model-builder")
        requirements = metadata.get_all("Requires-Dist", [])
        self.assertTrue(any(value.startswith("torch") for value in requirements))
        self.assertTrue(
            all("extra ==" in value for value in requirements), requirements
        )
        self.assertEqual(
            dict(entrypoints["console_scripts"]),
            {"pnms-model-builder": "model_builder.build.cli:main"},
        )

    def initialize(self):
        project = self.other / self._testMethodName
        project.mkdir()
        (project / "model.py").write_text(
            "raise AssertionError('the frontend must not import customer code')\n"
        )
        result = self._cli_without_ml("init", project, "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return project, json.loads(result.stdout)

    def test_installed_init_creates_only_configuration_and_adapter_without_checkpoint(
        self,
    ):
        project, result = self.initialize()
        self.assertEqual(result["status"], "initialized")
        self.assertEqual(
            {Path(path).name for path in result["created"]},
            {"model-build.json", "build_adapter.py"},
        )
        document = json.loads((project / "model-build.json").read_text())
        self.assertEqual(document["format_version"], 2)
        self.assertEqual(document["source"], ["model.py"])
        self.assertIsNone(document["checkpoint"])
        self.assertIsNone(document["builder_image"])
        self.assertEqual(
            {path.name for path in project.iterdir()},
            {"model.py", "model-build.json", "build_adapter.py"},
        )
        again = self._cli_without_ml("init", project, "--json")
        self.assertEqual(again.returncode, 2)
        self.assertIn("overwrite", again.stdout)

    def test_installed_check_and_build_describe_missing_settings_without_execution(
        self,
    ):
        project, _ = self.initialize()
        before = {path.name: path.read_bytes() for path in project.iterdir()}
        for command in ("check", "build"):
            with self.subTest(command=command):
                result = self._cli_without_ml(command, project, "--json")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["status"], "incomplete")
                fields = {item["field"] for item in report["diagnostics"]}
                executor = json.loads((project / "model-build.json").read_text())[
                    "executor"
                ]
                expected = {"checkpoint", "adapter"}
                if executor == "container":
                    expected.add("builder_image")
                elif command == "build":
                    expected.add("runtime")
                self.assertEqual(fields, expected)
        self.assertEqual(
            {path.name: path.read_bytes() for path in project.iterdir()}, before
        )

    def test_installed_local_config_only_check_captures_source_without_imports_or_native_sdk(
        self,
    ):
        project, _ = self.initialize()
        path = project / "model-build.json"
        document = json.loads(path.read_text())
        document.update(executor="local", device="cpu", checkpoint="weights.pt")
        path.write_text(json.dumps(document))
        (project / "weights.pt").write_bytes(
            b"check --config-only does not deserialize weights"
        )
        (project / "build_adapter.py").write_text(
            "raise AssertionError('check --config-only must not import the adapter')\n"
            "def create_model(config, assets):\n    return object()\n"
            "def create_cases(config, assets):\n    return []\n"
        )
        result = self._cli_without_ml("check", "--config-only", project, "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "configuration-ok")
        self.assertEqual(report["effective_config"]["source"], ["model.py"])
        self.assertFalse((project / "model-build.lock.json").exists())
        self.assertFalse((project / "builds").exists())

    def test_installed_config_only_check_resolves_custom_inputs_without_torch(self):
        model = self.other / "custom"
        model.mkdir()
        (model / "export.py").write_text(
            "raise RuntimeError('check --config-only must not import adapter')\n"
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
                "check",
                "--config-only",
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
                "check",
                "--config-only",
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

    def _project(self, document):
        project = self.other / self._testMethodName
        project.mkdir()
        runtime = project / "sdk" / "physicsnemo-infer"
        runtime.parent.mkdir()
        runtime.write_text("#!/bin/sh\nexit 99\n")
        runtime.chmod(0o755)
        document = {
            "format_version": 1,
            "executor": "local",
            "runtime": "sdk/physicsnemo-infer",
            "output_root": "builds",
            **document,
        }
        (project / "model-build.json").write_text(json.dumps(document))
        return project

    def _snapshot(self, project):
        return {
            str(path.relative_to(project)): path.read_bytes()
            for path in project.rglob("*")
            if path.is_file()
        }

    def _config_only_check(self, project, *arguments):
        result = self._cli_without_ml(
            "check", "--config-only", project, *arguments, "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["command"], "check")
        self.assertEqual(document["status"], "configuration-ok")
        return document

    def test_installed_project_profile_checks_gpu_requirement_without_probing_host(
        self,
    ):
        project = self._project(
            {
                "model": "affine",
                "device": "cpu",
                "backends": ["aoti", "tensorrt"],
                "default_profile": "h100",
                "profiles": {
                    "h100": {
                        "backends": ["aoti"],
                        "device": "cuda:99",
                        "required_gpu_arch": "sm90",
                    }
                },
            }
        )
        before = self._snapshot(project)
        result = self._config_only_check(project)
        self.assertEqual(result["profile"], "h100")
        effective = result["effective_config"]
        self.assertEqual(effective["model"], "affine")
        self.assertEqual(effective["device"], "cuda:99")
        self.assertEqual(effective["required_gpu_arch"], "sm90")
        self.assertEqual(effective["backends"], ["aoti"])
        self.assertEqual(Path(result["output"]).parent, (project / "builds").resolve())
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())

    def test_installed_project_paths_and_explicit_checkpoint_override_have_distinct_roots(
        self,
    ):
        project = self._project(
            {
                "recipe": "recipe/recipe.json",
                "config": "model-config.json",
                "checkpoint": "weights/trained.pt",
                "assets": {"normalization": "assets/normalization.json"},
                "device": "cpu",
                "backends": ["aoti"],
            }
        )
        recipe_dir = project / "recipe"
        recipe_dir.mkdir()
        (recipe_dir / "model.py").write_text(
            'raise AssertionError("check --config-only must not import the model adapter")\n'
        )
        (recipe_dir / "recipe.json").write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "name": "installed-custom",
                    "version": "0.1.0",
                    "adapter": "model.py",
                    "factory": "create_model",
                    "cases": "create_cases",
                    "config": {},
                    "checkpoint": {"format": "torch-state-dict"},
                    "assets": {"normalization": {"path": "default.json"}},
                    "input_names": ["input"],
                    "output_names": ["output"],
                    "dtype": "float32",
                    "shape": [1],
                    "supported_backends": ["aoti"],
                    "default_backend": "aoti",
                }
            )
        )
        (project / "model-config.json").write_text('{"width": 1}')
        (project / "weights").mkdir()
        (project / "weights/trained.pt").write_bytes(b"opaque checkpoint")
        (project / "assets").mkdir()
        (project / "assets/normalization.json").write_text('{"scale": 1}')
        before = self._snapshot(project)
        result = self._config_only_check(project / "model-build.json")
        effective = result["effective_config"]
        for field, path in (
            ("recipe", recipe_dir / "recipe.json"),
            ("config", project / "model-config.json"),
            ("checkpoint", project / "weights/trained.pt"),
        ):
            with self.subTest(field=field):
                self.assertEqual(Path(effective[field]).resolve(), path.resolve())
        self.assertEqual(
            Path(effective["assets"]["normalization"]).resolve(),
            (project / "assets/normalization.json").resolve(),
        )

        override = self.other / (self._testMethodName + "-override.pt")
        override.write_bytes(b"explicit cwd-relative checkpoint")
        changed = self._config_only_check(project, "--checkpoint", override.name)
        self.assertEqual(
            Path(changed["effective_config"]["checkpoint"]).resolve(),
            override.resolve(),
        )
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())

    def test_installed_recipe_project_hashes_checkpoint_without_loading_tensors(self):
        project = self._project(
            {
                "recipe": "recipe.json",
                "checkpoint": "weights/model.pt",
                "device": "cuda:99",
                "required_gpu_arch": "sm90",
                "backends": ["aoti"],
            }
        )
        recipe = {
            "format_version": 2,
            "name": "external-model",
            "version": "0.1.0",
            "adapter": "adapter.py",
            "factory": "create_model",
            "cases": "create_cases",
            "config": {"path": "config.json"},
            "checkpoint": {"format": "torch-state-dict"},
            "input_names": ["input"],
            "output_names": ["output"],
            "dtype": "float32",
            "shape": [1],
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
        }
        (project / "recipe.json").write_text(json.dumps(recipe))
        (project / "config.json").write_text("{}")
        (project / "adapter.py").write_text("raise AssertionError('must not import')\n")
        checkpoint = project / "weights/model.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(
            b"check --config-only must hash but never deserialize these tensors"
        )
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        before = self._snapshot(project)
        result = self._config_only_check(project)
        self.assertEqual(result["effective_config"]["checkpoint_sha256"], digest)
        self.assertEqual(result["effective_config"]["required_gpu_arch"], "sm90")
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())

    def test_installed_invalid_project_returns_json_diagnostics_without_writing(self):
        project = self._project({"model": "affine", "typo": True})
        before = self._snapshot(project)
        result = self._cli_without_ml("check", "--config-only", project, "--json")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(
            result.stdout.strip(), "invalid project must return a JSON result"
        )
        document = json.loads(result.stdout)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["command"], "check")
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["diagnostics"][0]["code"], "INVALID_PROJECT")
        self.assertIn("typo", document["diagnostics"][0]["message"])
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())

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

    def _config_only_check_without_site(self, recipe, directory):
        model = self.other / directory
        model.mkdir()
        (model / "export.py").write_text(
            "raise RuntimeError('check --config-only must not import adapter')\n"
        )
        (model / "config.json").write_text("{}\n")
        (model / "weights.pt").write_bytes(
            b"check --config-only must not deserialize weights"
        )
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
            "from model_builder.build import cli; "
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
                "check",
                "--config-only",
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
        self.assertFalse(output.exists(), "check --config-only must not create a build")
        self.assertEqual(json.loads(recipe_path.read_text()), recipe)
        return result

    def test_installed_config_only_check_accepts_independent_tensor_shapes_without_torch(
        self,
    ):
        result = self._config_only_check_without_site(
            self._recipe(), "valid-named-stage"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "configuration-ok")
        self.assertEqual(report["backends"], ["aoti"])
        self.assertEqual(report["device"], "cpu")

    def test_installed_config_only_check_rejects_malformed_tensor_contracts_without_torch(
        self,
    ):
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
                result = self._config_only_check_without_site(recipe, name)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn("configuration-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
