import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli


class FrontendReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.recipe_path = self.source / "recipe.json"
        self.recipe = {
            "format_version": 1,
            "name": "affine",
            "version": "0.1.0",
            "adapter": "export.py",
            "factory": "create_model",
            "cases": "create_cases",
            "supported_backends": ["aoti"],
            "default_backend": "aoti",
        }
        self.recipe_path.write_text(json.dumps(self.recipe))
        (self.source / "export.py").write_text("# recipe\n")
        self.output = self.root / "candidate"
        self.image = "example/builder@sha256:" + "a" * 64
        self.lock = self.root / "toolchain.lock.json"
        self.lock.write_text(
            json.dumps({"format_version": 1, "builder_image": self.image})
        )

    def plan(self, adapter):
        return {
            "recipe_path": self.recipe_path,
            "recipe": dict(self.recipe, adapter=adapter),
            "output": self.output,
            "device": "cpu",
            "image": self.image,
            "backends": ["aoti"],
        }

    def invoke(self, *args):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return cli.main(
                [
                    "build",
                    "--recipe",
                    str(self.recipe_path),
                    "--output",
                    str(self.output),
                    "--lock",
                    str(self.lock),
                    "--device",
                    "cpu",
                    *args,
                ]
            )

    def test_rejects_absolute_and_lexical_traversal_even_when_source_is_contained(self):
        for adapter in (str(self.source / "export.py"), "../source/export.py"):
            with self.subTest(adapter=adapter):
                self.recipe_path.write_text(
                    json.dumps(dict(self.recipe, adapter=adapter))
                )
                with self.assertRaisesRegex(cli.UsageError, "adapter"):
                    cli.read_recipe(self.recipe_path)

    def test_container_staging_rejects_escape_before_any_copy_or_launch(self):
        staged = self.root / "staged"
        staged.mkdir()
        with (
            mock.patch.object(
                cli.tempfile,
                "TemporaryDirectory",
                return_value=contextlib.nullcontext(str(staged)),
            ),
            mock.patch.object(cli.shutil, "copy2") as copy,
            mock.patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as run,
        ):
            with self.assertRaisesRegex(cli.UsageError, "adapter"):
                cli.container_build(self.plan("../source/export.py"))
            copy.assert_not_called()
            run.assert_not_called()

    def test_container_staging_checks_resolved_destination(self):
        staged = self.root / "staged"
        staged.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (staged / "nested").symlink_to(outside, target_is_directory=True)
        (self.source / "nested").mkdir()
        (self.source / "nested" / "export.py").write_text("# adapter\n")
        with (
            mock.patch.object(
                cli.tempfile,
                "TemporaryDirectory",
                return_value=contextlib.nullcontext(str(staged)),
            ),
            mock.patch.object(cli.shutil, "copy2") as copy,
            mock.patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as run,
        ):
            with self.assertRaisesRegex(cli.UsageError, "adapter"):
                cli.container_build(self.plan("nested/export.py"))
            copy.assert_not_called()
            run.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])

    def test_container_receipt_records_host_digest_and_lock_after_success_or_failure(
        self,
    ):
        for exit_code in (0, 17):
            with self.subTest(exit_code=exit_code):
                self.output = self.root / f"candidate-{exit_code}"

                def launch(plan):
                    self.output.mkdir()
                    (self.output / "build.json").write_text('{"worker": "retained"}\n')
                    # The nested local CLI may already have written its own receipt.
                    (self.output / "execution.json").write_text('{"executor":"local"}')
                    return exit_code

                with (
                    mock.patch.object(
                        cli.shutil, "which", return_value="/usr/bin/docker"
                    ),
                    mock.patch.object(cli, "container_build", side_effect=launch),
                ):
                    self.assertEqual(self.invoke(), exit_code)
                receipt_path = self.output / "execution.json"
                self.assertTrue(
                    receipt_path.is_file(),
                    "host execution provenance must survive the command",
                )
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(receipt["executor"], "container")
                self.assertEqual(receipt["builder_image"], self.image)
                self.assertEqual(receipt["exit_code"], exit_code)
                self.assertEqual(
                    receipt["toolchain_lock"]["sha256"],
                    hashlib.sha256(self.lock.read_bytes()).hexdigest(),
                )
                self.assertEqual(receipt["selection_source"], "lock")
                self.assertEqual(
                    (self.output / "build.json").read_text(), '{"worker": "retained"}\n'
                )

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
                    "pnmir_build.worker.execute_build", side_effect=execute
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

    def test_failure_before_candidate_creation_does_not_create_an_output(self):
        with (
            mock.patch.object(cli.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                cli, "container_build", side_effect=OSError("docker unavailable")
            ),
        ):
            self.assertEqual(self.invoke(), 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
