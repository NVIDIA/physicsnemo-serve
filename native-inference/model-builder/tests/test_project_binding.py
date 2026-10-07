"""A published project identity remains bound to actual producer/runtime bytes."""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_builder.build import worker
import worker_test_support


def identity(path):
    return {
        key: value
        for key, value in worker._file_identity(path).items()
        if key != "path"
    }


class ProjectBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = worker_test_support.WorkerFixture()
        self.fixture.addCleanup = self.addCleanup
        if hasattr(self.fixture, "setUp"):
            self.fixture.setUp()
        f = self.fixture
        self.expected = {
            "recipe": identity(f.recipe_path),
            "adapter": identity(f.recipe_path.parent / "export.py"),
            "runtime": identity(f.runtime),
            "model_inputs": None,
        }

    def build(self, expected=None, **kwargs):
        f = self.fixture
        with (
            mock.patch.object(
                worker, "_prepare_model", return_value=f.prepared
            ) as prepare,
            mock.patch.object(worker, "_build_backend", side_effect=f.backend),
        ):
            self.prepare = prepare
            return worker.execute_build(
                f.recipe_path,
                f.output,
                ["aoti"],
                "cpu",
                f.runtime,
                expected_identity=self.expected if expected is None else expected,
                **kwargs,
            )

    def assert_failed(self):
        f = self.fixture
        self.assertFalse((f.output / "model/model-release.json").exists())
        self.assertEqual(
            json.loads((f.output / "build.json").read_text())["status"], "failed"
        )

    def test_matching_expected_identity_preserves_complete_native_coverage(self):
        receipt = self.build()
        self.assertEqual(receipt["status"], "complete")
        check = json.loads((self.fixture.output / "checks/aoti.json").read_text())
        self.assertTrue(check["passed"])
        self.assertEqual(len(check["cases"]), len(self.fixture.prepared["cases"]))

    def test_changed_adapter_is_rejected_before_model_loading(self):
        (self.fixture.recipe_path.parent / "export.py").write_text("# new adapter\n")
        with self.assertRaisesRegex(ValueError, "adapter"):
            self.build()
        self.prepare.assert_not_called()
        self.assert_failed()

    def test_changed_recipe_is_rejected_before_model_loading(self):
        path = self.fixture.recipe_path
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "recipe"):
            self.build()
        self.prepare.assert_not_called()
        self.assert_failed()

    def test_changed_runtime_is_rejected_before_model_loading(self):
        path = self.fixture.runtime
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.build()
        self.prepare.assert_not_called()
        self.assert_failed()

    def test_model_input_expectation_cannot_be_ignored(self):
        expected = dict(
            self.expected,
            model_inputs={"checkpoint": {"sha256": "a" * 64, "size_bytes": 1}},
        )
        with self.assertRaisesRegex(ValueError, "model_inputs"):
            self.build(expected)
        self.prepare.assert_not_called()
        self.assert_failed()

    def test_runtime_swap_between_native_cases_fails_before_next_case(self):
        original = worker._native_case
        calls = []

        def run(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            runtime = self.fixture.runtime
            runtime.write_text(runtime.read_text() + "\n")
            return result

        with mock.patch.object(worker, "_native_case", side_effect=run):
            with self.assertRaisesRegex(ValueError, "runtime"):
                self.build()
        self.assertEqual(len(calls), 1)
        self.assert_failed()

    def test_runtime_swap_in_release_hook_prevents_candidate(self):
        def qualify(output, receipt):
            runtime = self.fixture.runtime
            runtime.write_text(runtime.read_text() + "\n")
            report = output / "model/qualification/test.json"
            worker._write_json(report, {"passed": True})
            return {
                "passed": True,
                "report": worker._file_identity(report, output / "model"),
            }

        with self.assertRaisesRegex(ValueError, "runtime"):
            self.build(pre_release_check=qualify)
        self.assert_failed()

    def test_external_validator_checks_subset_and_rejects_unknown_expectations(self):
        receipt = self.build()
        worker.verify_expected_identity(receipt, {"adapter": self.expected["adapter"]})
        worker.verify_expected_identity(receipt, None)
        for expected in (
            {"runtime": {"sha256": "0" * 64, "size_bytes": 1}},
            {"unexpected": {}},
            [],
            {
                "adapter": {
                    "sha256": self.expected["adapter"]["sha256"],
                    "size_bytes": True,
                }
            },
        ):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    worker.verify_expected_identity(receipt, expected)


if __name__ == "__main__":
    unittest.main()
