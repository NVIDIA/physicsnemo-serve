"""Checkpoint import remains usable from an unfinished, framework-free project."""

import contextlib
import hashlib
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
from pnmir_build import cli, scaffold


class CheckpointImportCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "customer model"
        scaffold.initialize(self.project)
        self.project_file = self.project / "model-build.json"
        document = json.loads(self.project_file.read_text())
        self.image = "sha256:" + "a" * 64
        document["builder_image"] = self.image
        self.project_file.write_text(json.dumps(document))
        self.original_project = self.project_file.read_bytes()
        self.checkpoint = self.project / "source.mdlus"
        self.checkpoint.write_bytes(b"selected checkpoint bytes")
        self.output = self.project / "imports" / "pretrained"

    def invoke(self, *options, behavior=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch("shutil.which", return_value="/usr/bin/docker"),
            mock.patch("subprocess.run", side_effect=behavior or self.success) as run,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = cli.main(
                [
                    "import-checkpoint",
                    str(self.checkpoint),
                    "--project",
                    str(self.project),
                    "--output",
                    str(self.output),
                    *options,
                    "--json",
                ]
            )
        return code, json.loads(stdout.getvalue()), stderr.getvalue(), run

    def success(self, command, **kwargs):
        if command[0] == "docker":
            mounts = [
                command[i + 1] for i, value in enumerate(command) if value == "--mount"
            ]
            source_mount = next(m for m in mounts if "dst=/inputs" in m)
            output_mount = next(m for m in mounts if "dst=/outputs" in m)
            source = Path(source_mount.split("src=", 1)[1].split(",dst=", 1)[0])
            destination = Path(output_mount.split("src=", 1)[1].split(",dst=", 1)[0])
            output = destination / Path(command[command.index("--output") + 1]).name
            checkpoint = source / Path(command[command.index("--checkpoint") + 1]).name
        else:
            output = Path(command[command.index("--output") + 1])
            checkpoint = Path(command[command.index("--checkpoint") + 1])
        output.mkdir()
        (output / "checkpoint.pt").write_bytes(b"converted tensor state")
        (output / "config.json").write_text('{"width": 4}')

        def identity(path):
            return {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }

        report = {
            "format_version": 1,
            "status": "imported",
            "checkpoint": identity(checkpoint),
            "model": {"module": "physicsnemo.models.example", "name": "Example"},
            "environment": {"torch": "test", "physicsnemo": "test"},
            "verification": {
                "strict_reload": True,
                "tensor_equality": True,
                "source_tensors_unchanged": True,
            },
            "artifacts": {
                "checkpoint": {
                    "path": "checkpoint.pt",
                    **identity(output / "checkpoint.pt"),
                },
                "config": {"path": "config.json", **identity(output / "config.json")},
            },
        }
        self.worker_output = output
        (output / "import.json").write_text(json.dumps(report))
        kwargs["stdout"].write("checkpoint loader diagnostic\n")
        return subprocess.CompletedProcess(command, 0)

    def test_import_uses_project_image_without_requiring_adapter_or_sdk(self):
        code, result, error, run = self.invoke()
        self.assertEqual(code, 0, result)
        self.assertEqual(result["status"], "imported")
        self.assertEqual(
            result["project_settings"],
            {
                "checkpoint": "imports/pretrained/checkpoint.pt",
                "config": "imports/pretrained/config.json",
            },
        )
        command = run.call_args.args[0]
        self.assertIn(self.image, command)
        self.assertIn("/inputs/checkpoint_import_worker.py", command)
        self.assertNotIn("--gpus", command)
        self.assertIn(f"{os.getuid()}:{os.getgid()}", command)
        self.assertIn("readonly", " ".join(command))
        self.assertTrue((self.output / "checkpoint.pt").is_file())
        self.assertIn("checkpoint loader diagnostic", error)
        self.assertEqual(self.project_file.read_bytes(), self.original_project)
        self.assertFalse((self.project / "model-build.lock.json").exists())

    def test_local_import_uses_selected_python_without_a_native_runtime(self):
        code, result, _, run = self.invoke("--executor", "local")
        self.assertEqual(code, 0, result)
        self.assertEqual(run.call_args.args[0][0], sys.executable)
        self.assertNotIn("--runtime", run.call_args.args[0])

    def test_existing_output_preserves_customer_files_without_execution(self):
        self.output.mkdir(parents=True)
        retained = self.output / "customer.txt"
        retained.write_text("keep me")
        code, _, _, run = self.invoke()
        self.assertEqual(code, 2)
        run.assert_not_called()
        self.assertEqual(retained.read_text(), "keep me")

    def test_mutable_image_is_rejected_before_execution(self):
        code, result, _, run = self.invoke("--builder-image", "builder:latest")
        self.assertEqual(code, 2)
        self.assertIn("immutable", result["diagnostics"][0]["message"])
        run.assert_not_called()

    def test_missing_environment_does_not_request_a_cpp_runtime(self):
        document = json.loads(self.project_file.read_text())
        document["builder_image"] = None
        self.project_file.write_text(json.dumps(document))
        code, result, _, run = self.invoke()
        self.assertEqual(code, 2)
        message = result["diagnostics"][0]["message"]
        self.assertIn("builder_image", message)
        self.assertNotIn("--runtime", message)
        run.assert_not_called()

    def test_hash_mismatch_is_rejected_before_execution(self):
        code, result, _, run = self.invoke("--checkpoint-sha256", "b" * 64)
        self.assertEqual(code, 2)
        self.assertIn("SHA-256", result["diagnostics"][0]["message"])
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failed_worker_does_not_publish_an_import(self):
        def failure(command, **kwargs):
            kwargs["stdout"].write("Unsupported constructor\n")
            return subprocess.CompletedProcess(command, 1)

        code, result, error, _ = self.invoke(behavior=failure)
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("Unsupported constructor", error)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.project_file.read_bytes(), self.original_project)

    def test_successful_exit_without_a_receipt_is_not_an_import(self):
        code, result, _, _ = self.invoke(
            behavior=lambda command, **kwargs: subprocess.CompletedProcess(command, 0)
        )
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())

    def test_modified_artifact_is_rejected_before_publication(self):
        def corrupt(command, **kwargs):
            completed = self.success(command, **kwargs)
            (self.worker_output / "checkpoint.pt").write_bytes(b"changed")
            return completed

        code, result, _, _ = self.invoke(behavior=corrupt)
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.output.exists())

    def test_report_for_different_source_is_rejected(self):
        def substitute(command, **kwargs):
            completed = self.success(command, **kwargs)
            path = self.worker_output / "import.json"
            report = json.loads(path.read_text())
            report["checkpoint"]["sha256"] = "c" * 64
            path.write_text(json.dumps(report))
            return completed

        code, _, _, _ = self.invoke(behavior=substitute)
        self.assertEqual(code, 1)
        self.assertFalse(self.output.exists())

    def test_changed_source_tensor_values_cannot_be_reported_as_imported(self):
        def changed(command, **kwargs):
            completed = self.success(command, **kwargs)
            path = self.worker_output / "import.json"
            report = json.loads(path.read_text())
            report["verification"]["source_tensors_unchanged"] = False
            path.write_text(json.dumps(report))
            return completed

        code, _, _, _ = self.invoke(behavior=changed)
        self.assertEqual(code, 1)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
