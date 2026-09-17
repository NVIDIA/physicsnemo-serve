"""Required architecture gates precede backend compilation and candidate release."""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pnmir_build import worker
import test_worker


class BuildTargetGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_worker.WorkerTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.set_runtime(
            "device"
        )  # Explicit native double reports CUDA execution.
        self.fixture.prepared["environment"]["compute_capability"] = [9, 0]

    def build(self, *, device="cuda", required="sm90", hook=None):
        f = self.fixture
        with (
            mock.patch.object(
                worker, "_prepare_model", return_value=f.prepared
            ) as prepare,
            mock.patch.object(
                worker, "_build_backend", side_effect=f.backend
            ) as compile,
        ):
            self.prepare = prepare
            self.compile = compile
            return worker.execute_build(
                f.recipe_path,
                f.output,
                ["aoti"],
                device,
                f.runtime,
                required_gpu_arch=required,
                pre_release_check=hook,
            )

    def assert_no_candidate(self):
        output = self.fixture.output
        self.assertFalse((output / "model/model-release.json").exists())
        self.assertEqual(
            json.loads((output / "build.json").read_text())["status"], "failed"
        )

    def test_matching_architecture_retains_complete_native_case_coverage(self):
        receipt = self.build()
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["environment"]["compute_capability"], [9, 0])
        checks = json.loads((self.fixture.output / "checks/aoti.json").read_text())
        self.assertTrue(checks["passed"])
        self.assertEqual(len(checks["cases"]), 2)
        self.assertTrue((self.fixture.output / "model/model-release.json").exists())

    def test_mismatched_architecture_fails_before_backend_compilation(self):
        self.fixture.prepared["environment"]["compute_capability"] = [8, 6]
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.build()
        self.compile.assert_not_called()
        self.assert_no_candidate()

    def test_unknown_or_malformed_measured_architecture_cannot_pass(self):
        for capability in (
            None,
            [],
            [9],
            [True, 0],
            [9.0, 0],
            [9, "0"],
            "sm90",
            [9, 0, 1],
        ):
            with self.subTest(capability=capability):
                self.fixture.output = (
                    self.fixture.root / f"bad-{len(list(self.fixture.root.iterdir()))}"
                )
                self.fixture.prepared["environment"]["compute_capability"] = capability
                with self.assertRaisesRegex(ValueError, "architecture"):
                    self.build()
                self.compile.assert_not_called()
                self.assert_no_candidate()

    def test_syntax_and_cpu_gpu_conflict_fail_before_model_loading(self):
        for device, required in (("cpu", "sm90"), ("cuda", "h100"), ("cuda", True)):
            with self.subTest(device=device, required=required):
                self.fixture.output = self.fixture.root / f"invalid-{device}-{required}"
                with self.assertRaisesRegex(ValueError, "architecture"):
                    self.build(device=device, required=required)
                self.prepare.assert_not_called()
                self.assertFalse(self.fixture.output.exists())

    def test_release_hook_cannot_change_measured_environment_then_publish(self):
        def hook(output, receipt):
            receipt["environment"]["compute_capability"] = [8, 6]
            report = output / "model/qualification/test.json"
            worker._write_json(report, {"passed": True})
            return {
                "passed": True,
                "report": worker._file_identity(report, output / "model"),
            }

        with self.assertRaisesRegex(ValueError, "architecture"):
            self.build(hook=hook)
        self.compile.assert_called_once()
        self.assert_no_candidate()


if __name__ == "__main__":
    unittest.main()
