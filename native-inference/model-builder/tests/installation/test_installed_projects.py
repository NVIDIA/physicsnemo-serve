"""Installed project commands resolve inputs without a host ML environment."""

import hashlib
import json
from pathlib import Path
import unittest

import test_installed_wheel


class InstalledProjectTests(test_installed_wheel.InstalledWheelTests):
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

    def _doctor(self, project, *arguments):
        result = self._cli_without_ml("doctor", project, *arguments, "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        document = json.loads(result.stdout)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["command"], "doctor")
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
        result = self._doctor(project)
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
            'raise AssertionError("doctor must not import the model adapter")\n'
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
        result = self._doctor(project / "model-build.json")
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
        changed = self._doctor(project, "--checkpoint", override.name)
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
        checkpoint.write_bytes(b"doctor must hash but never deserialize these tensors")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        before = self._snapshot(project)
        result = self._doctor(project)
        self.assertEqual(result["effective_config"]["checkpoint_sha256"], digest)
        self.assertEqual(result["effective_config"]["required_gpu_arch"], "sm90")
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())

    def test_installed_invalid_project_returns_json_diagnostics_without_writing(self):
        project = self._project({"model": "affine", "typo": True})
        before = self._snapshot(project)
        result = self._cli_without_ml("doctor", project, "--json")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertTrue(
            result.stdout.strip(), "invalid project must return a JSON result"
        )
        document = json.loads(result.stdout)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["command"], "doctor")
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["diagnostics"][0]["code"], "INVALID_PROJECT")
        self.assertIn("typo", document["diagnostics"][0]["message"])
        self.assertEqual(self._snapshot(project), before)
        self.assertFalse((project / "builds").exists())


if __name__ == "__main__":
    unittest.main()
