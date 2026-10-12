"""Container staging, host identity, provenance, and completion validation."""

import contextlib
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import cli
from authoring_test_support import linux_container
from model_input_test_support import CliRecipeFixture
import worker_test_support


class ContainerUserTests(unittest.TestCase):
    @linux_container()
    def test_unlisted_host_uid_has_username_and_writable_compiler_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = {"adapter": "export.py"}
            (root / "recipe.json").write_text(json.dumps(recipe))
            (root / "export.py").write_text("# adapter\n")
            plan = {
                "recipe_path": root / "recipe.json",
                "recipe": recipe,
                "output": root / "candidate",
                "device": "cuda",
                "image": "sha256:" + "a" * 64,
                "backends": ["aoti"],
            }
            with mock.patch.object(
                cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 17)
            ) as run:
                self.assertEqual(cli.container_build(plan), 17)
            command = run.call_args.args[0]
            environment = dict(
                command[index + 1].split("=", 1)
                for index, value in enumerate(command)
                if value == "-e"
            )
            error = None
            username = None
            pwd = ModuleType("pwd")
            pwd.getpwuid = mock.Mock(side_effect=KeyError("uid not found"))
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.dict(sys.modules, {"pwd": pwd}),
            ):
                try:
                    username = getpass.getuser()
                except (KeyError, OSError) as caught:
                    error = str(caught)
            self.assertIsNone(
                error, "compiler username discovery must work without a passwd entry"
            )
            self.assertTrue(username)
            self.assertTrue(
                environment.get("TRITON_CACHE_DIR", "").startswith("/tmp/"),
                "Triton cache must not default to an inaccessible image user's home",
            )
            self.assertEqual(
                command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}"
            )


class ContainerFrontendTests(CliRecipeFixture, unittest.TestCase):
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

    @linux_container()
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

    @linux_container()
    def test_failure_before_candidate_creation_does_not_create_an_output(self):
        with (
            mock.patch.object(cli.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                cli, "container_build", side_effect=OSError("docker unavailable")
            ),
        ):
            self.assertEqual(self.invoke(), 1)
        self.assertFalse(self.output.exists())


class ContainerCompletionTests(unittest.TestCase):
    def setUp(self):
        container = linux_container()
        container.__enter__()
        self.addCleanup(container.__exit__, None, None, None)
        self.fixture = worker_test_support.WorkerFixture()
        self.fixture.addCleanup = self.addCleanup
        if hasattr(self.fixture, "setUp"):
            self.fixture.setUp()
        self.root = self.fixture.root
        self.output = self.fixture.output
        self.plan = {
            "recipe_path": self.fixture.recipe_path,
            "recipe": self.fixture.recipe,
            "output": self.output,
            "device": "cpu",
            "image": "sha256:" + "a" * 64,
            "backends": ["aoti"],
        }

    def completed(self, name="build"):
        self.output = self.root / name
        self.fixture.output = self.output
        self.plan["output"] = self.output
        self.fixture.run_build(["aoti"])
        # The lightweight worker backend fixture omits the real exporter entrypoint.
        self.mutate_receipt(
            lambda receipt: receipt["variants"]["aoti"]["graph"].update(
                entrypoint="program.pt2"
            )
        )

    def container(self, code=0):
        with mock.patch.object(
            cli.subprocess, "run", return_value=subprocess.CompletedProcess([], code)
        ):
            return cli.container_build(self.plan)

    def mutate_receipt(self, change):
        path = self.output / "build.json"
        receipt = json.loads(path.read_text())
        change(receipt)
        path.write_text(json.dumps(receipt))

    def refresh_check(self, change):
        path = self.output / "checks/aoti.json"
        check = json.loads(path.read_text())
        change(check)
        path.write_text(json.dumps(check))
        self.mutate_receipt(
            lambda receipt: receipt["variants"]["aoti"]["checks"].update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
        )

    def test_zero_without_build_outputs_is_rejected(self):
        with self.assertRaisesRegex((ValueError, RuntimeError), "build|output|receipt"):
            self.container()
        self.assertFalse(self.output.exists())

    def test_complete_hash_verified_worker_output_is_accepted(self):
        self.completed()
        before = (self.output / "build.json").read_bytes()
        self.assertEqual(self.container(), 0)
        self.assertEqual((self.output / "build.json").read_bytes(), before)

    def test_nonzero_container_exit_keeps_original_code_and_diagnostics(self):
        self.output.mkdir()
        path = self.output / "build.json"
        path.write_text('{"status":"failed","error":"compiler failed"}')
        self.assertEqual(self.container(17), 17)
        self.assertIn("compiler failed", path.read_text())

    def test_wrong_completion_coverage_or_device_is_rejected(self):
        changes = {
            "failed": lambda r: r.update(status="failed"),
            "device": lambda r: r.update(device="cuda"),
            "requested": lambda r: r.update(requested_backends=["tensorrt"]),
            "missing": lambda r: r.update(variants={}),
            "duplicate": lambda r: r.update(requested_backends=["aoti", "aoti"]),
            "zero-cases": lambda r: r.update(case_count=0),
            "variant-failed": lambda r: r["variants"]["aoti"].update(status="failed"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.completed(name)
                self.mutate_receipt(change)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.container()

    def test_missing_or_corrupt_referenced_files_cannot_pass(self):
        mutations = {
            "release": lambda: (self.output / "model/model-release.json").write_text(
                "{}"
            ),
            "package": lambda: (
                self.output / "model/backends/aoti/model.pt2"
            ).write_bytes(b"corrupt"),
            "graph": lambda: (self.output / "exported/aoti/program.pt2").unlink(),
            "check": lambda: (self.output / "checks/aoti.json").write_text("{}"),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                self.completed(name)
                change()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.container()

    def test_self_consistently_hashed_but_failed_or_empty_checks_cannot_pass(self):
        changes = {
            "failed": lambda c: c.update(passed=False),
            "empty": lambda c: c.update(cases=[]),
            "short": lambda c: c.update(cases=c["cases"][:1]),
            "case-failed": lambda c: c["cases"][0].update(passed=False),
            "case-backend": lambda c: c["cases"][0]["metadata"].update(backend="wrong"),
            "case-incomplete": lambda c: c["cases"][0]["metadata"].update(
                completed=False
            ),
            "case-device": lambda c: c["cases"][0]["metadata"].update(
                execution_device={"type": "cuda", "index": 0}
            ),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.completed(name)
                self.refresh_check(change)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.container()

    def test_release_inventory_must_agree_with_build_receipt(self):
        self.completed()
        path = self.output / "model/model-release.json"
        release = json.loads(path.read_text())
        release["variants"]["aoti"]["files"] = []
        path.write_text(json.dumps(release))
        self.mutate_receipt(
            lambda receipt: receipt["release"].update(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
        )
        with self.assertRaises((ValueError, RuntimeError)):
            self.container()

    def test_completion_rejects_symlinked_evidence_directories(self):
        for relative in ("source", "model", "checks/aoti", "checks/aoti/case-0"):
            with self.subTest(directory=relative):
                name = relative.replace("/", "-")
                self.completed("linked-" + name)
                directory = self.output / relative
                outside = self.root / ("outside-" + name)
                directory.rename(outside)
                directory.symlink_to(outside, target_is_directory=True)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.container()

    def test_claimed_graph_paths_cannot_escape_output_or_follow_symlinks(self):
        for name in ("traversal", "symlink"):
            with self.subTest(name=name):
                self.completed(name)
                if name == "traversal":
                    self.mutate_receipt(
                        lambda receipt: receipt["variants"]["aoti"]["graphs"][0].update(
                            path="../outside"
                        )
                    )
                else:
                    path = self.output / "exported/aoti/program.pt2"
                    other = self.root / "outside"
                    other.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(other)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.container()


if __name__ == "__main__":
    unittest.main()
