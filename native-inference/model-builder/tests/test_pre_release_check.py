"""The additional qualification gate executes before candidate publication."""

import json
import unittest
from unittest import mock

import test_worker as worker_tests
from pnmir_build import worker


class PreReleaseCheckTest(unittest.TestCase):
    def setUp(self):
        self.fixture = worker_tests.WorkerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def build(self, check):
        f = self.fixture
        with (
            mock.patch.object(worker, "_prepare_model", return_value=f.prepared),
            mock.patch.object(worker, "_build_backend", side_effect=f.backend),
        ):
            return worker.execute_build(
                f.recipe_path,
                f.output,
                ["aoti"],
                "cpu",
                f.runtime,
                pre_release_check=check,
            )

    def qualification(self, output, receipt):
        self.assertFalse((output / "model/model-release.json").exists())
        self.assertEqual(receipt["status"], "building")
        self.assertEqual(receipt["variants"]["aoti"]["status"], "complete")
        report = output / "model/qualification/full-model.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"passed": True}))
        return {
            "passed": True,
            "report": worker._file_identity(report, output / "model"),
        }

    def test_qualification_precedes_release_and_moves_with_model(self):
        callback = mock.Mock(side_effect=self.qualification)
        receipt = self.build(callback)
        self.assertEqual(callback.call_count, 1)
        self.assertIn("qualification", receipt)
        release = json.loads(
            (self.fixture.output / "model/model-release.json").read_text()
        )
        self.assertEqual(release["qualification"], receipt["qualification"])
        self.assertEqual(
            receipt["qualification"]["report"]["path"], "qualification/full-model.json"
        )

    def test_failed_qualification_preserves_diagnostics_without_release(self):
        def fail(output, receipt):
            self.qualification(output, receipt)
            raise ValueError("full-model parity failed")

        with self.assertRaisesRegex(ValueError, "full-model parity failed"):
            self.build(fail)
        output = self.fixture.output
        self.assertFalse((output / "model/model-release.json").exists())
        self.assertTrue((output / "model/qualification/full-model.json").is_file())
        receipt = json.loads((output / "build.json").read_text())
        self.assertEqual(receipt["status"], "failed")

    def test_qualification_cannot_mutate_retained_source_then_publish(self):
        def mutate(output, receipt):
            result = self.qualification(output, receipt)
            path = output / "source/recipe.json"
            path.write_text(path.read_text() + "\n")
            return result

        with self.assertRaisesRegex(ValueError, "retained source changed"):
            self.build(mutate)
        self.assertFalse((self.fixture.output / "model/model-release.json").exists())
        receipt = json.loads((self.fixture.output / "build.json").read_text())
        self.assertEqual(receipt["status"], "failed")

    def test_false_success_and_unbound_reports_cannot_publish(self):
        for mode in ("false", "missing", "wrong-hash", "escape", "absolute", "symlink"):
            with self.subTest(mode=mode):
                self.fixture.output = self.fixture.root / mode

                def callback(output, receipt):
                    result = self.qualification(output, receipt)
                    if mode == "false":
                        result["passed"] = 1
                    elif mode == "missing":
                        result["report"]["path"] = "missing.json"
                    elif mode == "wrong-hash":
                        result["report"]["sha256"] = "0" * 64
                    elif mode == "escape":
                        result["report"]["path"] = "../build.json"
                    elif mode == "absolute":
                        result["report"]["path"] = str(
                            output / "model/qualification/full-model.json"
                        )
                    else:
                        link = output / "model/link.json"
                        link.symlink_to(output / "model/qualification/full-model.json")
                        result["report"]["path"] = "link.json"
                    return result

                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.build(callback)
                self.assertFalse(
                    (self.fixture.output / "model/model-release.json").exists()
                )

    def test_qualification_cannot_leave_stale_artifact_or_check_inventories(self):
        for kind in ("package", "graph", "checks"):
            with self.subTest(kind=kind):
                self.fixture.output = self.fixture.root / ("mutate-" + kind)

                def mutate(output, receipt):
                    result = self.qualification(output, receipt)
                    variant = receipt["variants"]["aoti"]
                    if kind == "package":
                        path = output / "model" / variant["files"][0]["path"]
                    elif kind == "graph":
                        path = output / variant["graphs"][0]["path"]
                    else:
                        path = output / variant["checks"]["path"]
                    path.write_bytes(path.read_bytes() + b"\n")
                    return result

                with self.assertRaisesRegex(ValueError, "changed"):
                    self.build(mutate)
                self.assertFalse(
                    (self.fixture.output / "model/model-release.json").exists()
                )


if __name__ == "__main__":
    unittest.main()
