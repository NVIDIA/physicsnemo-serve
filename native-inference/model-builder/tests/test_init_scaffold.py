"""Initialization should work in an existing model repository without ML tools."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


LAUNCHER = Path(__file__).resolve().parents[2] / "physicsnemo-model-builder"
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
        from pnmir_build.scaffold import initialize

        for name in ("model-build.json", "build_adapter.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / name).write_text("existing customer content")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    initialize(root)
                self.assertEqual({p.name for p in root.iterdir()}, {name})
                self.assertEqual((root / name).read_text(), "existing customer content")

    def test_new_directory_can_be_initialized_with_absent_selected_inputs(self):
        from pnmir_build.scaffold import initialize
        from pnmir_build.authoring_config import load, missing

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
        from pnmir_build import scaffold

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
        from pnmir_build import scaffold

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
        from pnmir_build.scaffold import initialize

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
        from pnmir_build.scaffold import initialize

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "project"
            initialize(root, checkpoint=Path("weights.pt"))
            document = json.loads((root / "model-build.json").read_text())
            self.assertEqual(document["checkpoint"], str(Path("weights.pt").absolute()))
            other = root.parent / "other"
            initialize(other, checkpoint=other / "weights.pt")
            document = json.loads((other / "model-build.json").read_text())
            self.assertEqual(document["checkpoint"], "weights.pt")


if __name__ == "__main__":
    unittest.main()
