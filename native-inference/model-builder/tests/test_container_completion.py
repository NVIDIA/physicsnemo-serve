"""The container exit code is insufficient proof of a completed model build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import cli
import test_worker


class ContainerCompletionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_worker.WorkerTest(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
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
