"""Initialization and configuration guidance work without an ML environment."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


LAUNCHER = Path(__file__).resolve().parents[2] / "pnms-model-builder"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class InitScaffoldTests(unittest.TestCase):
    def test_cli_initializes_existing_model_without_a_checkpoint_or_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "My CFD Model"
            root.mkdir()
            existing = root / "model.py"
            existing.write_text("raise RuntimeError('model must not be imported')\n")
            result = subprocess.run(
                [sys.executable, str(LAUNCHER), "init", str(root), "--json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            project = json.loads((root / "model-build.json").read_text())
            self.assertEqual(project["format_version"], 2)
            self.assertEqual(project["name"], "my-cfd-model")
            self.assertEqual(project["source"], ["model.py"])
            self.assertIsNone(project["checkpoint"])
            self.assertIsNone(project["builder_image"])
            self.assertEqual(
                {p.name for p in root.iterdir()},
                {"model.py", "model-build.json", "build_adapter.py"},
            )
            self.assertEqual(
                existing.read_text(),
                "raise RuntimeError('model must not be imported')\n",
            )
            self.assertIn("create_model", (root / "build_adapter.py").read_text())

    def test_existing_project_or_adapter_is_preserved_without_partial_scaffold(self):
        from model_builder.build.scaffold import initialize

        for name in ("model-build.json", "build_adapter.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_text("existing customer content")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    initialize(root)
                self.assertEqual({p.name for p in root.iterdir()}, {name})
                self.assertEqual((root / name).read_text(), "existing customer content")

    def test_new_directory_can_be_initialized_with_absent_selected_inputs(self):
        from model_builder.build.scaffold import initialize
        from model_builder.build.authoring_config import load, missing

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new-model"
            result = initialize(root, checkpoint=Path("weights.pt"), source=["network"])
            self.assertEqual(len(result["created"]), 2)
            project = load(root)
            diagnostics = missing(project["effective"], root)
            self.assertTrue(any(item["field"] == "checkpoint" for item in diagnostics))
            self.assertTrue(any(item["field"] == "source" for item in diagnostics))
            self.assertEqual(
                len(
                    [
                        item
                        for item in diagnostics
                        if item["code"] == "AUTHORING_ADAPTER_INCOMPLETE"
                    ]
                ),
                2,
            )
            self.assertFalse((root / "model-build.lock.json").exists())

    def test_second_file_failure_rolls_back_only_own_published_file(self):
        from model_builder.build import scaffold

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("# keep\n")
            original = scaffold._exclusive_write

            def competing_write(path, content):
                if path.name == "build_adapter.py":
                    path.write_text("concurrent owner")
                    raise FileExistsError("concurrent initialization")
                return original(path, content)

            with mock.patch.object(
                scaffold, "_exclusive_write", side_effect=competing_write
            ):
                with self.assertRaisesRegex(ValueError, "initialize"):
                    scaffold.initialize(root)
            self.assertFalse((root / "model-build.json").exists())
            self.assertEqual(
                (root / "build_adapter.py").read_text(), "concurrent owner"
            )
            self.assertEqual((root / "model.py").read_text(), "# keep\n")
            self.assertFalse(list(root.glob(".model-builder-init-*")))

    def test_rollback_preserves_concurrent_replacement_of_own_file(self):
        from model_builder.build import scaffold

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = scaffold._exclusive_write

            def competing_write(path, content):
                if path.name == "build_adapter.py":
                    replacement = root / "replacement"
                    replacement.write_text("concurrent replacement")
                    replacement.replace(root / "model-build.json")
                    raise OSError("simulated publication failure")
                return original(path, content)

            with mock.patch.object(
                scaffold, "_exclusive_write", side_effect=competing_write
            ):
                with self.assertRaisesRegex(ValueError, "initialize"):
                    scaffold.initialize(root)
            self.assertEqual(
                (root / "model-build.json").read_text(), "concurrent replacement"
            )

    def test_symlink_target_or_source_is_rejected_without_writes(self):
        from model_builder.build.scaffold import initialize

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "model-build.json"
            target.symlink_to(root / "absent")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                initialize(root)
            target.unlink()
            (root / "network").symlink_to(root / "outside")
            with self.assertRaisesRegex(ValueError, "symlink"):
                initialize(root, source=["network/model.py"])
            self.assertFalse(target.exists())

    def test_optional_checkpoint_uses_caller_directory_and_is_portable_when_contained(
        self,
    ):
        from model_builder.build.scaffold import initialize

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project"
            initialize(root, checkpoint=Path("weights.pt"))
            document = json.loads((root / "model-build.json").read_text())
            self.assertEqual(document["checkpoint"], str(Path("weights.pt").absolute()))
            other = root.parent / "other"
            initialize(other, checkpoint=other / "weights.pt")
            document = json.loads((other / "model-build.json").read_text())
            self.assertEqual(document["checkpoint"], "weights.pt")


class AuthoringCommandTests(unittest.TestCase):
    def invoke(self, *args, python=None):
        result = subprocess.run(
            [python or sys.executable, "-S", str(LAUNCHER), *args, "--json"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.returncode, json.loads(result.stdout), result.stderr

    def test_incomplete_init_explains_checkpoint_and_hooks_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 0, result)
            self.assertEqual(result["status"], "initialized")
            original = {p.name: p.read_bytes() for p in project.iterdir()}
            for arguments in (("check", "--config-only"), ("check",), ("build",)):
                code, result, error = self.invoke(*arguments, str(project))
                self.assertEqual(code, 2, result)
                fields = {item.get("field") for item in result["diagnostics"]}
                self.assertIn("checkpoint", fields)
                self.assertIn("adapter", fields)
                self.assertEqual(
                    original, {p.name: p.read_bytes() for p in project.iterdir()}
                )

    def test_checkpoint_can_be_configured_after_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 0, result)
            config_path = project / "model-build.json"
            config = json.loads(config_path.read_text())
            # Configuration checks must identify bytes without attempting to deserialize them.
            (project / "weights.pt").write_bytes(
                b"opaque checkpoint for metadata validation"
            )
            config.update(checkpoint="weights.pt", executor="local", device="cpu")
            config_path.write_text(json.dumps(config))
            (project / "build_adapter.py").write_text(
                "def create_model(config, assets):\n    return None\n"
                "def create_cases(config, assets):\n    return []\n"
            )
            code, result, error = self.invoke("check", "--config-only", str(project))
            self.assertEqual(code, 0, result)
            self.assertEqual(result["status"], "configuration-ok")
            self.assertFalse((project / "model-build.lock.json").exists())
            self.assertFalse((project / "builds").exists())

    def test_existing_adapter_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            adapter = project / "build_adapter.py"
            adapter.write_text("existing customer code\n")
            code, result, error = self.invoke("init", str(project))
            self.assertEqual(code, 2, result)
            self.assertEqual(adapter.read_text(), "existing customer code\n")
            self.assertFalse((project / "model-build.json").exists())


if __name__ == "__main__":
    unittest.main()
