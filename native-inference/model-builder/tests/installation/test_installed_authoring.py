"""Installed authoring onboarding requires neither checkout imports nor host ML."""

import json
from pathlib import Path
import unittest

import test_installed_wheel


class InstalledAuthoringTests(test_installed_wheel.InstalledWheelTests):
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
                self.assertTrue({"checkpoint", "adapter", "builder_image"} <= fields)
        self.assertEqual(
            {path.name: path.read_bytes() for path in project.iterdir()}, before
        )

    def test_installed_local_doctor_captures_source_without_imports_or_native_sdk(self):
        project, _ = self.initialize()
        path = project / "model-build.json"
        document = json.loads(path.read_text())
        document.update(executor="local", device="cpu", checkpoint="weights.pt")
        path.write_text(json.dumps(document))
        (project / "weights.pt").write_bytes(b"doctor does not deserialize weights")
        (project / "build_adapter.py").write_text(
            "raise AssertionError('doctor must not import the adapter')\n"
            "def create_model(config, assets):\n    return object()\n"
            "def create_cases(config, assets):\n    return []\n"
        )
        result = self._cli_without_ml("doctor", project, "--json")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "configuration-ok")
        self.assertEqual(report["effective_config"]["source"], ["model.py"])
        self.assertFalse((project / "model-build.lock.json").exists())
        self.assertFalse((project / "builds").exists())


if __name__ == "__main__":
    unittest.main()
