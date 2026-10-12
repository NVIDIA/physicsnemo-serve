"""Authoring projects resolve inputs without importing a customer's model."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from authoring_test_support import AuthoringConfigFixture

LAUNCHER = Path(__file__).resolve().parents[2] / "pnms-model-builder"


class AuthoringConfigurationTests(AuthoringConfigFixture, unittest.TestCase):
    def test_cli_reports_actionable_incomplete_authoring_project(self):
        self.write()
        result = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "check",
                "--config-only",
                str(self.root),
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        messages = " ".join(item["message"] for item in payload["diagnostics"])
        self.assertIn("checkpoint", messages)
        self.assertNotIn("format_version must be integer 1", messages)

    def test_incomplete_configuration_is_loadable_without_creating_files(self):
        from model_builder.build.authoring_config import load, missing

        document = self.write()
        before = list(self.root.iterdir())
        project = load(self.path)
        self.assertEqual(project["document"], document)
        self.assertEqual(
            project["effective"]["adapter"], str(self.root / "build_adapter.py")
        )
        self.assertIsNone(project["effective"]["checkpoint"])
        self.assertEqual(project["effective"]["config"], {})
        self.assertEqual(
            project["source_identity"],
            {
                "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "size_bytes": len(self.path.read_bytes()),
            },
        )
        self.assertTrue(missing(project["effective"], self.root))
        self.assertEqual(list(self.root.iterdir()), before)

    def test_profile_and_cli_override_paths_without_mutating_arguments_or_document(
        self,
    ):
        from model_builder.build.authoring_config import load

        document = self.write(
            checkpoint="weights.pt",
            config="config.json",
            assets={"mesh": "mesh.bin"},
            profiles={
                "local": {
                    "executor": "local",
                    "device": "cpu",
                    "runtime": "bin/runtime",
                }
            },
            default_profile="local",
        )
        args = argparse.Namespace(
            checkpoint=Path("other.pt"),
            backend=["tensorrt"],
            device="cuda:1",
            asset=["mesh=other-mesh.bin"],
        )
        original = vars(args).copy()
        project = load(self.path, args)
        effective = project["effective"]
        self.assertEqual(project["profile"], "local")
        self.assertEqual(effective["checkpoint"], str(Path("other.pt").absolute()))
        self.assertEqual(effective["config"], str(self.root / "config.json"))
        self.assertEqual(
            effective["assets"]["mesh"], str(Path("other-mesh.bin").absolute())
        )
        self.assertEqual(effective["runtime"], str(self.root / "bin/runtime"))
        self.assertEqual(effective["executor"], "local")
        self.assertEqual(effective["device"], "cuda:1")
        self.assertEqual(effective["backends"], ["tensorrt"])
        self.assertEqual(project["document"], document)
        self.assertEqual(vars(args), original)

    def test_invalid_schema_source_and_profiles_are_rejected(self):
        from model_builder.build.authoring_config import load

        for changes, message in (
            ({"checkpoint": 3}, "checkpoint"),
            ({"config": None}, "config"),
            ({"config": []}, "config"),
            ({"unknown": True}, "Unknown"),
            ({"source": ["../model.py"]}, "relative"),
            ({"source": ["/model.py"]}, "relative"),
            ({"source": ["model.py", "./model.py"]}, "duplicate"),
            ({"source": [False]}, "source"),
            ({"adapter": "../model.py"}, "relative"),
            ({"input_names": ["input", "input"]}, "input_names"),
            ({"output_names": []}, "output_names"),
            ({"profiles": {"gpu": {"checkpoint": "new.pt"}}}, "execution"),
            ({"profiles": {"gpu": {"source": ["new.py"]}}}, "execution"),
            ({"backends": ["unknown"]}, "backends"),
            ({"default_profile": "absent"}, "default_profile"),
        ):
            with self.subTest(changes=changes):
                self.write(**changes)
                with self.assertRaisesRegex(ValueError, message):
                    load(self.path)

    def test_duplicate_and_nonfinite_json_are_rejected(self):
        from model_builder.build.authoring_config import load

        for source, message in (
            ('{"name":"a","name":"b"}', "duplicate"),
            ('{"config":{"value":NaN}}', "non-finite"),
            ('{"config":{"value":1e999}}', "non-finite"),
        ):
            with self.subTest(source=source):
                self.path.write_text(source)
                with self.assertRaisesRegex(ValueError, message):
                    load(self.path)

    def test_aoti_profile_selects_existing_policy_without_changing_omitted_default(
        self,
    ):
        from model_builder.build.authoring_config import load

        self.write()
        self.assertNotIn("aoti_profile", load(self.path)["effective"])
        for profile in ("baseline", "aten-boundary-exact-v2"):
            with self.subTest(profile=profile):
                self.write(aoti_profile=profile)
                self.assertEqual(load(self.path)["effective"]["aoti_profile"], profile)

    def test_invalid_aoti_profile_is_rejected_before_execution(self):
        from model_builder.build.authoring_config import load

        for profile in ("unknown", "", None, [], {}, 1):
            with self.subTest(profile=profile):
                self.write(aoti_profile=profile)
                with self.assertRaisesRegex(ValueError, "AOTI profile"):
                    load(self.path)

    def test_explicit_root_source_capture_is_allowed_but_not_an_adapter_directory(self):
        from model_builder.build.authoring_config import load

        self.write(source=["."])
        try:
            project = load(self.path)
        except ValueError as exc:
            self.fail(f"An explicitly selected source root should be supported: {exc}")
        self.assertEqual(project["effective"]["source"], ["."])
        self.write(adapter="./")
        with self.assertRaisesRegex(ValueError, "relative"):
            load(self.path)

    def test_source_symlink_parent_is_rejected(self):
        from model_builder.build.authoring_config import load

        (self.root / "linked").symlink_to(self.root / "somewhere")
        self.write(source=["linked/model.py"])
        with self.assertRaisesRegex(ValueError, "symlink"):
            load(self.path)

    def test_local_check_does_not_require_runtime_and_toolchain_can_supply_environment(
        self,
    ):
        from model_builder.build.authoring_config import load, missing

        self.write(executor="local")
        effective = load(self.path)["effective"]
        codes = {
            item["code"]
            for item in missing(effective, self.root, require_runtime=False)
        }
        self.assertNotIn("MISSING_RUNTIME", codes)
        self.assertIn(
            "MISSING_RUNTIME", {item["code"] for item in missing(effective, self.root)}
        )
        self.write(toolchain_lock="selected-lock.json")
        codes = {
            item["code"] for item in missing(load(self.path)["effective"], self.root)
        }
        self.assertNotIn("MISSING_BUILDER_IMAGE", codes)

    def test_missing_checks_config_json_and_adapter_without_importing_them(self):
        from model_builder.build.authoring_config import load, missing

        (self.root / "weights.pt").write_bytes(b"weights are not loaded")
        (self.root / "config.json").write_text("[]")
        (self.root / "build_adapter.py").write_text(
            "raise RuntimeError('never import')\n"
            "def create_model(config, assets):\n    return object()\n"
            "def create_cases(config, assets):\n    return []\n"
        )
        self.write(checkpoint="weights.pt", config="config.json", executor="local")
        diagnostics = missing(
            load(self.path)["effective"], self.root, require_runtime=False
        )
        self.assertEqual(
            [item["code"] for item in diagnostics], ["AUTHORING_CONFIG_INVALID"]
        )


if __name__ == "__main__":
    unittest.main()
